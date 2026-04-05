"""Fire-and-forget experiment runner.

Processes the experiment matrix, tracks state for resumability, handles failures.
Each experiment is a unique (base, quant, specialization, vanilla|finetuned) combo.

Usage:
    python scripts/run_experiments.py --resume
    python scripts/run_experiments.py --filter "qwen3.5-9b*chat*"
    python scripts/run_experiments.py --dry-run
    python scripts/run_experiments.py --filter "qwen3.5-9b_gptq-int4_chat*" --resume
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Ensure scripts/ is on the path for sibling imports
sys.path.insert(0, str(Path(__file__).parent))

from common import (
    ROOT,
    Timer,
    load_experiment_states,
    load_experiments_config,
    log,
    make_experiment_id,
    make_initial_state,
    save_experiment_state,
    training_cache_key,
)


def expand_experiment_matrix(config: dict) -> list[dict]:
    """Expand the experiment matrix into individual experiment entries.

    Each (base x quant x specialization) produces TWO experiments:
    vanilla (no fine-tune) and finetuned.
    """
    experiments = []
    bases = config.get("bases", {})
    quant_levels = config.get("quant_levels", {})

    for entry in config.get("experiments", []):
        base_name = entry["base"]
        if base_name not in bases:
            log.warning(f"Base '{base_name}' not found in config, skipping")
            continue

        base_cfg = bases[base_name]

        for quant_name in entry.get("quants", []):
            if quant_name not in quant_levels:
                log.warning(f"Quant '{quant_name}' not found in config, skipping")
                continue

            quant_cfg = quant_levels[quant_name]

            for spec in entry.get("specializations", []):
                for finetuned in [False, True]:
                    exp_id = make_experiment_id(base_name, quant_name, spec, finetuned)
                    experiments.append({
                        "id": exp_id,
                        "base_name": base_name,
                        "base_cfg": base_cfg,
                        "quant_name": quant_name,
                        "quant_cfg": quant_cfg,
                        "specialization": spec,
                        "finetuned": finetuned,
                    })

    return experiments


def determine_steps(experiment: dict) -> list[str]:
    """Determine which steps this experiment needs."""
    steps = ["download"]

    if experiment["finetuned"]:
        steps.append("train")
        # Dense models need merge; MoE models skip it (adapter-based)
        if experiment["base_cfg"]["architecture"] != "moe":
            steps.append("merge")

    steps.append("quantize")
    steps.append("benchmark")

    return steps


def run_step(experiment: dict, step: str) -> None:
    """Execute a single pipeline step."""
    base_name = experiment["base_name"]
    quant_name = experiment["quant_name"]
    spec = experiment["specialization"]
    finetuned = experiment["finetuned"]

    if step == "download":
        from download import download_for_experiment_base
        download_for_experiment_base(base_name)

        # Also download datasets for curate step
        # The curate step handles its own dataset loading, so we just ensure
        # the base model is present
        _ensure_datasets(spec)

    elif step == "train":
        from train import train
        train(base_name, spec)

    elif step == "merge":
        from merge import merge
        merge(base_name, spec)

    elif step == "quantize":
        from quantize import quantize
        quantize(base_name, quant_name, spec, finetuned)

    elif step == "benchmark":
        from eval import evaluate
        exp_id = experiment["id"]
        evaluate(exp_id)


def _ensure_datasets(specialization: str) -> None:
    """Ensure curated datasets exist for a specialization."""
    processed = ROOT / "datasets" / "processed" / specialization / "train.jsonl"
    if processed.exists():
        return

    log.info(f"Curated data for '{specialization}' not found. Running curate...")

    # Check if this specialization matches a model ID in models.yaml
    from common import load_models_config
    models_cfg = load_models_config()

    if specialization in models_cfg.get("models", {}):
        model_cfg = models_cfg["models"][specialization]
        if model_cfg.get("enabled", False):
            # Download datasets first
            from download import download_datasets
            download_datasets(specialization, model_cfg)

            # Then curate
            from curate import curate_model
            curate_model(specialization)
            return

    log.warning(
        f"No model config for specialization '{specialization}'. "
        f"Provide curated data at {processed} manually."
    )


def run_experiment(experiment: dict, max_retries: int = 2) -> dict:
    """Run a single experiment through all its steps. Returns final state."""
    exp_id = experiment["id"]
    states = load_experiment_states()
    state = states.get(exp_id, make_initial_state(exp_id))

    if state["status"] == "done":
        log.info(f"[{exp_id}] Already done, skipping")
        return state

    steps = determine_steps(experiment)
    completed = set(state.get("completed_steps", []))

    state["status"] = "running"
    state["started_at"] = state.get("started_at") or datetime.now(timezone.utc).isoformat()
    save_experiment_state(state)

    with Timer() as total_timer:
        for step in steps:
            if step in completed:
                log.info(f"[{exp_id}] Step '{step}' already done, skipping")
                continue

            log.info(f"[{exp_id}] Running step: {step}")
            state["current_step"] = step
            save_experiment_state(state)

            try:
                with Timer() as step_timer:
                    run_step(experiment, step)

                completed.add(step)
                state["completed_steps"] = list(completed)
                state["current_step"] = None
                state["error"] = None
                save_experiment_state(state)
                log.info(f"[{exp_id}] Step '{step}' done ({step_timer.hours:.2f}h)")

            except Exception as e:
                state["retry_count"] = state.get("retry_count", 0) + 1
                state["error"] = str(e)
                state["failed_step"] = step

                log.error(f"[{exp_id}] Step '{step}' failed: {e}")
                log.error(traceback.format_exc())

                if state["retry_count"] >= max_retries:
                    state["status"] = "failed"
                    save_experiment_state(state)
                    log.error(
                        f"[{exp_id}] Max retries ({max_retries}) reached. "
                        f"Marking as failed."
                    )
                    return state

                # OOM handling: halve batch size and retry
                if "out of memory" in str(e).lower() or "OOM" in str(e):
                    log.warning(f"[{exp_id}] OOM detected, will retry with lower batch size")
                    # The train.py script should handle this via config overrides
                    # For now, just log and retry

                state["status"] = "pending"  # will retry
                save_experiment_state(state)
                log.info(f"[{exp_id}] Will retry (attempt {state['retry_count']}/{max_retries})")

                # Retry the failed step
                try:
                    run_step(experiment, step)
                    completed.add(step)
                    state["completed_steps"] = list(completed)
                    state["current_step"] = None
                    state["error"] = None
                    save_experiment_state(state)
                except Exception as e2:
                    state["retry_count"] = state.get("retry_count", 0) + 1
                    state["error"] = str(e2)
                    if state["retry_count"] >= max_retries:
                        state["status"] = "failed"
                        save_experiment_state(state)
                        return state

    state["status"] = "done"
    state["duration_hours"] = round(total_timer.hours, 2)
    save_experiment_state(state)

    # Load eval results into state
    eval_file = ROOT / "logs" / "eval" / f"{exp_id}.json"
    if eval_file.exists():
        import json
        state["results"] = json.loads(eval_file.read_text())
        save_experiment_state(state)

    log.info(f"[{exp_id}] Experiment complete ({total_timer.hours:.1f}h)")

    # Cleanup
    _cleanup(experiment)

    return state


def _cleanup(experiment: dict) -> None:
    """Clean up intermediate artifacts if configured."""
    from common import load_experiments_config
    config = load_experiments_config()
    cleanup_cfg = config.get("cleanup", {})

    if cleanup_cfg.get("delete_merged_after_quantize", False):
        merged_dir = (
            ROOT / "models" / "merged"
            / f"{experiment['base_name']}_{experiment['specialization']}"
        )
        if merged_dir.exists():
            import shutil
            log.info(f"Cleaning up merged model: {merged_dir}")
            shutil.rmtree(merged_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fire-and-forget experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate pipeline with smallest experiment:
  python scripts/run_experiments.py --filter "qwen3.5-9b_gptq-int4_chat*" --resume

  # Run all experiments (resumable):
  python scripts/run_experiments.py --resume

  # Dry run to see what would execute:
  python scripts/run_experiments.py --dry-run

  # Filter to specific base model:
  python scripts/run_experiments.py --filter "qwen3.5-9b*" --resume
        """,
    )
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip completed experiments (default: True)")
    parser.add_argument("--filter", help="Glob pattern to filter experiment IDs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run without executing")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Max retries per failed experiment")
    args = parser.parse_args()

    config = load_experiments_config()
    experiments = expand_experiment_matrix(config)

    # Apply filter
    if args.filter:
        experiments = [
            e for e in experiments
            if fnmatch.fnmatch(e["id"], args.filter)
        ]

    if not experiments:
        log.warning("No experiments match the filter. Nothing to do.")
        return

    # Load existing states for resume
    states = load_experiment_states()

    # Sort: vanilla before finetuned (vanilla is faster, good for early feedback)
    experiments.sort(key=lambda e: (e["finetuned"], e["base_name"], e["quant_name"]))

    log.info(f"Experiment matrix: {len(experiments)} experiments")

    if args.dry_run:
        for exp in experiments:
            existing = states.get(exp["id"], {})
            status = existing.get("status", "new")
            steps = determine_steps(exp)
            completed = existing.get("completed_steps", [])
            remaining = [s for s in steps if s not in completed]
            log.info(
                f"  {exp['id']}: status={status}, "
                f"steps={steps}, remaining={remaining}"
            )
        return

    # Run experiments
    completed = 0
    failed = 0
    skipped = 0

    for i, exp in enumerate(experiments, 1):
        existing = states.get(exp["id"], {})
        if existing.get("status") == "done":
            log.info(f"[{i}/{len(experiments)}] {exp['id']} — already done, skipping")
            skipped += 1
            continue

        log.info(f"\n{'='*60}")
        log.info(f"[{i}/{len(experiments)}] Starting: {exp['id']}")
        log.info(f"{'='*60}\n")

        result = run_experiment(exp, max_retries=args.max_retries)

        if result["status"] == "done":
            completed += 1
        elif result["status"] == "failed":
            failed += 1

    # Generate report
    log.info("\n" + "=" * 60)
    log.info("All experiments processed. Generating report...")
    log.info("=" * 60 + "\n")

    from generate_report import generate_report
    generate_report()

    log.info(f"\nFinal: {completed} completed, {failed} failed, {skipped} skipped")


if __name__ == "__main__":
    main()
