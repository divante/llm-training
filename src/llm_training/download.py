"""Download base models and datasets from HuggingFace.

Usage:
    python scripts/download.py --model code
    python scripts/download.py --base qwen3.5-9b
    python scripts/download.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_training.common import ROOT, load_experiments_config, load_models_config, log, append_run_log


def download_base_model(hf_id: str, local_dir: Path) -> None:
    """Download a base model from HuggingFace Hub."""
    if local_dir.exists() and any(local_dir.iterdir()):
        log.info(f"Base model already exists at {local_dir}, skipping download")
        return

    from huggingface_hub import snapshot_download

    log.info(f"Downloading base model: {hf_id} -> {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(hf_id, local_dir=str(local_dir))
    log.info(f"Downloaded {hf_id}")


def download_datasets(model_id: str, model_cfg: dict) -> int:
    """Download HuggingFace datasets for a model. Returns count downloaded."""
    from datasets import load_dataset

    downloaded = 0
    for ds in model_cfg.get("datasets", []):
        if ds["source"] != "huggingface":
            continue

        out_path = ROOT / "datasets" / "raw" / ds["name"]
        if out_path.exists() and any(out_path.iterdir()):
            log.info(f"[{model_id}] Dataset {ds['name']} already exists, skipping")
            downloaded += 1
            continue

        hf_id = ds["hf_id"]
        log.info(f"[{model_id}] Downloading {ds['name']} from {hf_id}...")

        try:
            dataset = load_dataset(hf_id, split="train")
            sample_size = ds.get("sample_size")
            if sample_size:
                actual_size = min(sample_size, len(dataset))
                dataset = dataset.shuffle(seed=42).select(range(actual_size))

            out_path.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(out_path))
            log.info(f"[{model_id}] Saved {len(dataset)} examples to {out_path}")
            downloaded += 1
        except Exception as e:
            log.error(f"[{model_id}] Failed to download {ds['name']}: {e}")
            log.error("This may be a gated dataset requiring HF approval.")

    return downloaded


def download_for_model(model_id: str) -> None:
    """Download base model + datasets for a model config entry."""
    models_cfg = load_models_config()
    model_cfg = models_cfg["models"].get(model_id)
    if not model_cfg:
        log.error(f"Model '{model_id}' not found in models.yaml")
        sys.exit(1)

    if not model_cfg.get("enabled", False):
        log.warning(f"Model '{model_id}' is disabled in config, skipping")
        return

    # Download base model
    base = model_cfg["base"]
    if model_cfg.get("use_fallback"):
        base = model_cfg.get("fallback_base", base)

    local_dir = ROOT / "models" / "bases" / model_id
    download_base_model(base, local_dir)

    # Download pre-quantized base for MoE adapter serving
    if model_cfg.get("moe_base_quantized"):
        moe_base_dir = ROOT / "models" / "gptq" / f"{model_id}-base"
        download_base_model(model_cfg["moe_base_quantized"], moe_base_dir)

    # Download datasets
    count = download_datasets(model_id, model_cfg)

    append_run_log({
        "model": model_id,
        "phase": "download",
        "base": base,
        "datasets_downloaded": count,
    })


def download_for_experiment_base(base_name: str) -> None:
    """Download a base model referenced in experiments.yaml."""
    exp_cfg = load_experiments_config()
    base_cfg = exp_cfg.get("bases", {}).get(base_name)
    if not base_cfg:
        log.error(f"Base '{base_name}' not found in experiments.yaml")
        sys.exit(1)

    hf_id = base_cfg["hf_id"]
    local_dir = ROOT / "models" / "bases" / base_name
    download_base_model(hf_id, local_dir)

    # Pre-quantized base for MoE
    if base_cfg.get("pre_quantized_base"):
        moe_base_dir = ROOT / "models" / "gptq" / f"{base_name}-base"
        download_base_model(base_cfg["pre_quantized_base"], moe_base_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download models and datasets")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Download base + datasets for a model config entry")
    group.add_argument("--base", help="Download a base model from experiments.yaml")
    group.add_argument("--all", action="store_true", help="Download all enabled models")
    args = parser.parse_args()

    if args.model:
        download_for_model(args.model)
    elif args.base:
        download_for_experiment_base(args.base)
    elif args.all:
        models_cfg = load_models_config()
        for model_id, model_cfg in models_cfg["models"].items():
            if model_cfg.get("enabled", False):
                download_for_model(model_id)


if __name__ == "__main__":
    main()
