"""Merge LoRA adapter weights into base model.

For dense models only — MoE models use adapter-based serving.

Usage:
    python scripts/merge.py --base qwen3.5-9b --specialization chat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import ROOT, Timer, append_run_log, load_experiments_config, log, training_cache_key


def merge(
    base_name: str,
    specialization: str,
    output_dir: Path | None = None,
) -> Path:
    """Merge LoRA adapter into base model. Returns path to merged model."""
    exp_cfg = load_experiments_config()
    base_cfg = exp_cfg["bases"][base_name]

    if base_cfg["architecture"] == "moe":
        log.info(
            f"MoE model ({base_name}) — skipping merge. "
            f"Use adapter-based serving instead."
        )
        return ROOT / "models" / "lora" / training_cache_key(base_name, specialization)

    cache_key = training_cache_key(base_name, specialization)
    adapter_dir = ROOT / "models" / "lora" / cache_key
    merged_dir = output_dir or (ROOT / "models" / "merged" / f"{base_name}_{specialization}")

    if merged_dir.exists() and any(merged_dir.glob("*.safetensors")):
        log.info(f"Merged model already exists at {merged_dir}, skipping")
        return merged_dir

    if not adapter_dir.exists() or not (adapter_dir / "adapter_config.json").exists():
        log.error(f"Adapter not found at {adapter_dir}. Run train.py first.")
        sys.exit(1)

    base_model_path = ROOT / "models" / "bases" / base_name
    if not base_model_path.exists():
        log.error(f"Base model not found at {base_model_path}")
        sys.exit(1)

    merged_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Merging adapter into base:")
    log.info(f"  Base: {base_model_path}")
    log.info(f"  Adapter: {adapter_dir}")
    log.info(f"  Output: {merged_dir}")

    with Timer() as timer:
        _run_merge(base_model_path, adapter_dir, merged_dir)

    log.info(f"Merge completed in {timer.hours:.2f} hours")

    append_run_log({
        "model": f"{base_name}_{specialization}",
        "phase": "merge",
        "output": str(merged_dir),
        "duration_hours": round(timer.hours, 2),
    })

    return merged_dir


def _run_merge(base_path: Path, adapter_path: Path, output_path: Path) -> None:
    """Execute the merge."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        torch_dtype=torch.float16,  # merge in fp16 to save RAM
        device_map="cpu",  # merge on CPU to avoid GPU memory issues
        trust_remote_code=True,
    )

    log.info("Loading adapter...")
    model = PeftModel.from_pretrained(model, str(adapter_path))

    log.info("Merging weights...")
    model = model.merge_and_unload()

    log.info(f"Saving merged model to {output_path}...")
    model.save_pretrained(str(output_path))

    tokenizer = AutoTokenizer.from_pretrained(str(base_path), trust_remote_code=True)
    tokenizer.save_pretrained(str(output_path))

    log.info("Merge complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base", required=True, help="Base model name")
    parser.add_argument("--specialization", required=True, help="Specialization")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    args = parser.parse_args()

    merge(args.base, args.specialization, args.output_dir)


if __name__ == "__main__":
    main()
