"""Benchmark models before and after fine-tuning.

Uses lm-eval-harness for standard benchmarks and custom eval on held-out data.

Usage:
    python scripts/eval.py --experiment-id qwen3.5-9b_gptq-int4_chat_vanilla
    python scripts/eval.py --base qwen3.5-9b --quant gptq_int4 --specialization chat --finetuned
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import (
    ROOT,
    Timer,
    append_run_log,
    load_experiments_config,
    log,
    make_experiment_id,
)


# Eval suite mapping
EVAL_SUITES = {
    "humaneval": "humaneval",
    "mbpp": "mbpp",
    "multipl_e": "multiple_e",
    "mt_bench": "mt_bench",
    "mt_bench_writing": "mt_bench",
    "alpaca_eval": "alpaca_eval",
    "alpaca_eval_writing": "alpaca_eval",
    "mmlu_subset": "mmlu",
    "arc_challenge": "arc_challenge",
}


def evaluate(experiment_id: str) -> dict:
    """Run evaluation for an experiment. Returns results dict."""
    # Parse experiment ID
    parts = experiment_id.rsplit("_", 3)
    if len(parts) < 4:
        log.error(f"Cannot parse experiment ID: {experiment_id}")
        sys.exit(1)

    variant = parts[-1]  # vanilla or finetuned
    specialization = parts[-2]
    quant = parts[-3]
    base_name = "_".join(parts[:-3])

    exp_cfg = load_experiments_config()
    spec_cfg = exp_cfg.get("specializations", {}).get(specialization, {})
    eval_suite = spec_cfg.get("eval_suite", [])
    custom_eval_path = spec_cfg.get("custom_eval")

    # Find model path
    model_path = ROOT / "models" / "gptq" / experiment_id
    if not model_path.exists():
        log.error(f"Model not found at {model_path}. Run quantize.py first.")
        sys.exit(1)

    # Check if this is an MoE model with adapter config
    moe_config_path = model_path / "moe_config.json"
    is_moe = moe_config_path.exists()

    results: dict = {}

    log.info(f"Evaluating: {experiment_id}")
    log.info(f"  Specialization: {specialization}")
    log.info(f"  Eval suite: {eval_suite}")
    log.info(f"  MoE: {is_moe}")

    with Timer() as timer:
        # Run standard benchmarks via lm-eval-harness
        if eval_suite:
            bench_results = _run_benchmarks(model_path, eval_suite, is_moe)
            results.update(bench_results)

        # Run custom eval
        if custom_eval_path:
            custom_path = ROOT / custom_eval_path
            if custom_path.exists():
                custom_results = _run_custom_eval(model_path, custom_path, is_moe)
                results["custom_eval"] = custom_results
            else:
                log.warning(f"Custom eval data not found at {custom_path}")

        # Measure inference performance
        perf_results = _measure_performance(model_path, is_moe)
        results.update(perf_results)

    # Save results
    eval_dir = ROOT / "logs" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_file = eval_dir / f"{experiment_id}.json"
    eval_file.write_text(json.dumps(results, indent=2))

    log.info(f"Evaluation complete in {timer.hours:.2f} hours")
    log.info(f"Results: {json.dumps(results, indent=2)}")

    append_run_log({
        "model": experiment_id,
        "phase": "eval",
        "stage": variant,
        "results": results,
        "duration_hours": round(timer.hours, 2),
    })

    return results


def _run_benchmarks(model_path: Path, eval_suite: list[str], is_moe: bool) -> dict:
    """Run lm-eval-harness benchmarks."""
    results = {}

    try:
        import lm_eval

        # Map our eval names to lm-eval task names
        tasks = []
        for name in eval_suite:
            lm_eval_name = EVAL_SUITES.get(name, name)
            tasks.append(lm_eval_name)

        log.info(f"Running lm-eval-harness tasks: {tasks}")

        eval_results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_path},trust_remote_code=True",
            tasks=tasks,
            batch_size="auto",
        )

        if eval_results and "results" in eval_results:
            for task_name, task_results in eval_results["results"].items():
                # Extract the primary metric for each task
                for metric, value in task_results.items():
                    if isinstance(value, (int, float)):
                        results[f"{task_name}_{metric}"] = value

    except ImportError:
        log.warning("lm-eval-harness not installed. Skipping standard benchmarks.")
    except Exception as e:
        log.error(f"Benchmark failed: {e}")

    return results


def _run_custom_eval(model_path: Path, eval_data_path: Path, is_moe: bool) -> dict:
    """Run custom evaluation on held-out data."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    log.info(f"Running custom eval from {eval_data_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), device_map="auto", trust_remote_code=True,
    )
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)

    # Load eval examples
    examples = []
    with open(eval_data_path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    # Generate responses and collect basic stats
    total = len(examples)
    generated = 0
    errors = 0
    total_tokens = 0

    for ex in examples[:100]:  # cap at 100 for speed
        try:
            if "conversations" in ex:
                prompt = ex["conversations"][0]["value"]
            elif "instruction" in ex:
                prompt = ex["instruction"]
            else:
                continue

            result = gen(prompt, max_new_tokens=256)
            output = result[0]["generated_text"][len(prompt):]
            total_tokens += len(output.split())
            generated += 1
        except Exception as e:
            errors += 1

    return {
        "total_examples": total,
        "evaluated": generated,
        "errors": errors,
        "avg_output_tokens": total_tokens / max(generated, 1),
    }


def _measure_performance(model_path: Path, is_moe: bool) -> dict:
    """Measure inference performance (tok/s, VRAM)."""
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    results = {}

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), device_map="auto", trust_remote_code=True,
        )

        prompt = "Write a detailed explanation of how neural networks work, including backpropagation."
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Warmup
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=10)

        # Timed generation
        gen_tokens = 100
        start = time.monotonic()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=gen_tokens)
        elapsed = time.monotonic() - start

        actual_tokens = output.shape[1] - inputs["input_ids"].shape[1]
        results["tok_per_sec"] = round(actual_tokens / elapsed, 1)
        results["ttft_ms"] = None  # would need streaming to measure properly

        # VRAM usage
        if torch.cuda.is_available():
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            results["vram_gb"] = round(vram_gb, 1)

        log.info(f"Performance: {results['tok_per_sec']} tok/s, VRAM: {results.get('vram_gb', 'N/A')} GB")

    except Exception as e:
        log.error(f"Performance measurement failed: {e}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument("--experiment-id", help="Full experiment ID")
    parser.add_argument("--base", help="Base model name")
    parser.add_argument("--quant", help="Quant level")
    parser.add_argument("--specialization", help="Specialization")
    parser.add_argument("--finetuned", action="store_true", help="Evaluate finetuned version")
    args = parser.parse_args()

    if args.experiment_id:
        evaluate(args.experiment_id)
    elif args.base and args.quant and args.specialization:
        exp_id = make_experiment_id(args.base, args.quant, args.specialization, args.finetuned)
        evaluate(exp_id)
    else:
        parser.error("Provide --experiment-id or --base + --quant + --specialization")


if __name__ == "__main__":
    main()
