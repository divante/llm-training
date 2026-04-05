"""Shared utilities for the training pipeline."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("LLM_TRAINING_ROOT", Path(__file__).resolve().parent.parent.parent))
MODELS_CONFIG = ROOT / "config" / "models.yaml"
EXPERIMENTS_CONFIG = ROOT / "config" / "experiments.yaml"
STATE_FILE = ROOT / "logs" / "experiment_state.jsonl"
RUNS_LOG = ROOT / "logs" / "runs.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("llm-training")


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_models_config() -> dict:
    return load_yaml(MODELS_CONFIG)


def load_experiments_config() -> dict:
    return load_yaml(EXPERIMENTS_CONFIG)


def resolve_base_template(architecture: str, train_method: str) -> Path:
    """Return the correct base training config template."""
    if architecture == "moe":
        return ROOT / "config" / "train" / "base_moe.yaml"
    if train_method == "bf16_lora":
        return ROOT / "config" / "train" / "base_dense_lora.yaml"
    return ROOT / "config" / "train" / "base_qlora.yaml"


def merge_training_config(
    base_template: dict, specialization_config: dict, base_name: str
) -> dict:
    """Deep-merge base template with per-model overrides from specialization config."""
    merged = deep_merge({}, base_template)
    overrides = specialization_config.get("overrides", {}).get(base_name, {})
    return deep_merge(merged, overrides)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def append_run_log(entry: dict) -> None:
    """Append one JSON line to the run log."""
    entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    with open(RUNS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def make_experiment_id(
    base: str, quant: str, specialization: str, finetuned: bool
) -> str:
    variant = "finetuned" if finetuned else "vanilla"
    return f"{base}_{quant}_{specialization}_{variant}"


def training_cache_key(base: str, specialization: str) -> str:
    """Training depends on (base, specialization), not quant level."""
    return f"{base}_{specialization}"


# --- Experiment state management ---


def load_experiment_states() -> dict[str, dict]:
    """Load all experiment states from the JSONL file. Last entry per ID wins."""
    states: dict[str, dict] = {}
    if STATE_FILE.exists():
        for line in STATE_FILE.read_text().strip().splitlines():
            if line.strip():
                entry = json.loads(line)
                states[entry["id"]] = entry
    return states


def save_experiment_state(state: dict) -> None:
    """Append experiment state (last entry per ID wins on reload)."""
    state.setdefault("updated_at", datetime.now(timezone.utc).isoformat())
    with open(STATE_FILE, "a") as f:
        f.write(json.dumps(state) + "\n")


def make_initial_state(experiment_id: str) -> dict:
    return {
        "id": experiment_id,
        "status": "pending",
        "completed_steps": [],
        "current_step": None,
        "results": {},
        "retry_count": 0,
        "error": None,
        "started_at": None,
        "duration_hours": None,
    }


class Timer:
    """Simple context manager for timing steps."""

    def __init__(self) -> None:
        self.start_time: float = 0
        self.elapsed: float = 0

    def __enter__(self) -> Timer:
        self.start_time = time.monotonic()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed = time.monotonic() - self.start_time

    @property
    def hours(self) -> float:
        return self.elapsed / 3600
