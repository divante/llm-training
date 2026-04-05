"""Dataset curation: clean, filter, dedup, format, and split.

Usage:
    python scripts/curate.py --model code
    python scripts/curate.py --model chat
    python scripts/curate.py --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from common import ROOT, load_models_config, log, append_run_log


def load_raw_dataset(ds_config: dict) -> list[dict] | None:
    """Load a raw dataset from disk. Returns None if unavailable."""
    if ds_config["source"] == "huggingface":
        raw_path = ROOT / "datasets" / "raw" / ds_config["name"]
        if not raw_path.exists():
            log.warning(f"Raw dataset not found: {raw_path}")
            return None
        from datasets import load_from_disk
        ds = load_from_disk(str(raw_path))
        return [dict(row) for row in ds]

    elif ds_config["source"] == "local":
        local_path = ROOT / ds_config["path"]
        if not local_path.exists() or not any(local_path.iterdir()):
            if ds_config.get("custom"):
                log.warning(
                    f"Custom dataset not available yet: {ds_config['name']} "
                    f"(expected at {local_path}). Skipping."
                )
                return None
            log.error(f"Local dataset missing: {local_path}")
            return None

        # Load JSONL files from the directory
        examples = []
        for f in sorted(local_path.glob("*.jsonl")):
            for line in f.read_text().strip().splitlines():
                if line.strip():
                    examples.append(json.loads(line))
        for f in sorted(local_path.glob("*.json")):
            data = json.loads(f.read_text())
            if isinstance(data, list):
                examples.extend(data)
        return examples

    return None


def quality_filter(examples: list[dict], max_seq_length: int = 8192) -> list[dict]:
    """Remove low-quality examples."""
    filtered = []
    for ex in examples:
        # Get text content for length checking
        text = _extract_text(ex)
        if not text:
            continue

        # Too short
        if len(text.split()) < 20:  # rough proxy for 50 tokens
            continue

        # Nonsensical (>50% non-alphanumeric)
        alnum = sum(c.isalnum() or c.isspace() for c in text)
        if len(text) > 0 and alnum / len(text) < 0.5:
            continue

        filtered.append(ex)

    removed = len(examples) - len(filtered)
    if removed:
        log.info(f"Quality filter removed {removed} examples")
    return filtered


def _extract_text(example: dict) -> str:
    """Extract text from an example regardless of format."""
    # ShareGPT format
    if "conversations" in example:
        return " ".join(
            turn.get("value", "") for turn in example["conversations"]
        )
    # Alpaca format
    if "instruction" in example:
        parts = [example.get("instruction", ""), example.get("input", ""),
                 example.get("output", "")]
        return " ".join(p for p in parts if p)
    # Raw text
    if "text" in example:
        return example["text"]
    if "content" in example:
        return example["content"]
    # Fallback: concat all string values
    return " ".join(str(v) for v in example.values() if isinstance(v, str))


def exact_dedup(examples: list[dict]) -> list[dict]:
    """Remove exact duplicates based on normalized text hash."""
    seen: set[str] = set()
    unique = []
    for ex in examples:
        text = _extract_text(ex)
        # Normalize: lowercase, collapse whitespace
        normalized = re.sub(r"\s+", " ", text.lower().strip())
        h = hashlib.md5(normalized.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(ex)

    removed = len(examples) - len(unique)
    if removed:
        log.info(f"Exact dedup removed {removed} examples")
    return unique


def near_dedup(examples: list[dict], threshold: float = 0.85) -> list[dict]:
    """Near-duplicate removal using MinHash (datasketch)."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        log.warning("datasketch not installed, skipping near-dedup")
        return examples

    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    kept = []
    removed = 0

    for i, ex in enumerate(examples):
        text = _extract_text(ex)
        words = text.lower().split()
        if not words:
            continue

        mh = MinHash(num_perm=128)
        for w in words:
            mh.update(w.encode("utf-8"))

        key = str(i)
        if lsh.query(mh):
            removed += 1
            continue

        try:
            lsh.insert(key, mh)
            kept.append(ex)
        except ValueError:
            # Duplicate key, skip
            removed += 1

    if removed:
        log.info(f"Near-dedup removed {removed} examples")
    return kept


def convert_to_sharegpt(example: dict) -> dict | None:
    """Convert any format to ShareGPT multi-turn format."""
    if "conversations" in example:
        # Already ShareGPT
        return example

    if "instruction" in example:
        # Alpaca → ShareGPT
        user_msg = example["instruction"]
        if example.get("input"):
            user_msg += "\n\n" + example["input"]
        return {
            "conversations": [
                {"from": "human", "value": user_msg},
                {"from": "gpt", "value": example.get("output", "")},
            ]
        }

    if "text" in example or "content" in example:
        # Completion format — wrap as single turn
        text = example.get("text", example.get("content", ""))
        return {
            "conversations": [
                {"from": "human", "value": "Continue the following:"},
                {"from": "gpt", "value": text},
            ]
        }

    return None


def curate_model(model_id: str) -> None:
    """Run full curation pipeline for a model."""
    models_cfg = load_models_config()
    model_cfg = models_cfg["models"].get(model_id)
    if not model_cfg:
        log.error(f"Model '{model_id}' not found in models.yaml")
        sys.exit(1)

    all_examples: list[dict] = []
    dataset_counts: dict[str, int] = {}

    for ds_config in model_cfg.get("datasets", []):
        log.info(f"[{model_id}] Loading dataset: {ds_config['name']}")
        raw = load_raw_dataset(ds_config)
        if raw is None:
            continue

        log.info(f"[{model_id}] {ds_config['name']}: {len(raw)} raw examples")

        # Quality filter
        filtered = quality_filter(raw)

        # Convert to unified format
        converted = []
        for ex in filtered:
            c = convert_to_sharegpt(ex)
            if c:
                converted.append(c)

        dataset_counts[ds_config["name"]] = len(converted)
        all_examples.extend(converted)

    if not all_examples:
        log.error(f"[{model_id}] No examples after processing. Check dataset availability.")
        return

    log.info(f"[{model_id}] Total before dedup: {len(all_examples)}")

    # Dedup
    all_examples = exact_dedup(all_examples)
    all_examples = near_dedup(all_examples)

    log.info(f"[{model_id}] Total after dedup: {len(all_examples)}")

    # Shuffle with fixed seed
    random.seed(42)
    random.shuffle(all_examples)

    # Split off calibration and eval
    calibration_size = 256
    eval_size = 500

    if len(all_examples) < calibration_size + eval_size + 100:
        log.error(
            f"[{model_id}] Only {len(all_examples)} examples — not enough for "
            f"calibration ({calibration_size}) + eval ({eval_size}) + training"
        )
        return

    eval_split = all_examples[-eval_size:]
    all_examples = all_examples[:-eval_size]

    cal_split = all_examples[-calibration_size:]
    all_examples = all_examples[:-calibration_size]

    train_split = all_examples

    # Write outputs
    out_dir = ROOT / "datasets" / "processed" / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(out_dir / "train.jsonl", train_split)
    _write_jsonl(ROOT / "datasets" / "calibration" / f"{model_id}.jsonl", cal_split)
    _write_jsonl(ROOT / "datasets" / "eval" / f"{model_id}.jsonl", eval_split)

    log.info(f"[{model_id}] Output:")
    log.info(f"  Train: {len(train_split)} examples -> {out_dir / 'train.jsonl'}")
    log.info(f"  Calibration: {len(cal_split)} examples")
    log.info(f"  Eval: {len(eval_split)} examples")
    log.info(f"  Per-dataset: {dataset_counts}")

    append_run_log({
        "model": model_id,
        "phase": "curate",
        "train_examples": len(train_split),
        "calibration": len(cal_split),
        "eval": len(eval_split),
        "per_dataset": dataset_counts,
    })


def _write_jsonl(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate datasets")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Curate datasets for a specific model")
    group.add_argument("--all", action="store_true", help="Curate all enabled models")
    args = parser.parse_args()

    if args.model:
        curate_model(args.model)
    elif args.all:
        models_cfg = load_models_config()
        for model_id, model_cfg in models_cfg["models"].items():
            if model_cfg.get("enabled", False):
                curate_model(model_id)


if __name__ == "__main__":
    main()
