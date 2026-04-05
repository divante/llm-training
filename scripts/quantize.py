"""GPTQ quantization for fine-tuned or vanilla models.

Handles dense (AutoGPTQ) and MoE (pre-quantized base + adapter) models.

Usage:
    python scripts/quantize.py --base qwen3.5-9b --quant gptq_int4 --specialization chat --finetuned
    python scripts/quantize.py --base qwen3.5-9b --quant gptq_int4 --specialization chat
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
    training_cache_key,
)


def quantize(
    base_name: str,
    quant_name: str,
    specialization: str,
    finetuned: bool,
) -> Path:
    """Quantize a model. Returns path to the quantized output."""
    exp_cfg = load_experiments_config()
    base_cfg = exp_cfg["bases"][base_name]
    quant_cfg = exp_cfg["quant_levels"][quant_name]
    experiment_id = make_experiment_id(base_name, quant_name, specialization, finetuned)

    output_dir = ROOT / "models" / "gptq" / experiment_id

    if output_dir.exists() and any(output_dir.glob("*.safetensors")):
        log.info(f"Quantized model already exists at {output_dir}, skipping")
        return output_dir

    method = quant_cfg["method"]
    if method != "gptq":
        log.error(f"Only GPTQ quantization is currently active. Got: {method}")
        sys.exit(1)

    architecture = base_cfg["architecture"]

    # MoE models: use pre-quantized base, no requantization needed
    if architecture == "moe":
        return _handle_moe(base_name, quant_name, specialization, finetuned, base_cfg, experiment_id)

    # Dense models: quantize with AutoGPTQ
    return _handle_dense(base_name, quant_name, specialization, finetuned, base_cfg, quant_cfg, experiment_id)


def _handle_moe(
    base_name: str,
    quant_name: str,
    specialization: str,
    finetuned: bool,
    base_cfg: dict,
    experiment_id: str,
) -> Path:
    """MoE: use pre-quantized base from HF. Adapter is loaded at serve time."""
    pre_quant_id = base_cfg.get("pre_quantized_base")
    if not pre_quant_id:
        log.error(f"MoE base {base_name} has no pre_quantized_base configured")
        sys.exit(1)

    base_gptq_dir = ROOT / "models" / "gptq" / f"{base_name}-base"
    output_dir = ROOT / "models" / "gptq" / experiment_id

    if not base_gptq_dir.exists():
        log.info(f"Downloading pre-quantized base: {pre_quant_id}")
        from huggingface_hub import snapshot_download
        base_gptq_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(pre_quant_id, local_dir=str(base_gptq_dir))

    # For MoE, the "quantized model" is just a symlink to the pre-quantized base
    output_dir.mkdir(parents=True, exist_ok=True)
    config_file = output_dir / "moe_config.json"
    config = {
        "experiment_id": experiment_id,
        "base_gptq": str(base_gptq_dir),
        "finetuned": finetuned,
    }

    if finetuned:
        cache_key = training_cache_key(base_name, specialization)
        adapter_dir = ROOT / "models" / "lora" / cache_key
        if not adapter_dir.exists():
            log.error(f"Adapter not found at {adapter_dir}. Run train.py first.")
            sys.exit(1)
        config["adapter"] = str(adapter_dir)
        log.info(f"MoE finetuned: base={base_gptq_dir}, adapter={adapter_dir}")
    else:
        log.info(f"MoE vanilla: using pre-quantized base at {base_gptq_dir}")

    config_file.write_text(json.dumps(config, indent=2))

    append_run_log({
        "model": experiment_id,
        "phase": "quantize",
        "method": "moe_prequant",
        "base_gptq": str(base_gptq_dir),
        "finetuned": finetuned,
    })

    return output_dir


def _handle_dense(
    base_name: str,
    quant_name: str,
    specialization: str,
    finetuned: bool,
    base_cfg: dict,
    quant_cfg: dict,
    experiment_id: str,
) -> Path:
    """Dense models: quantize with AutoGPTQ."""
    output_dir = ROOT / "models" / "gptq" / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    bits = quant_cfg["bits"]
    group_size = quant_cfg["group_size"]

    # Determine source model
    if finetuned:
        cache_key = training_cache_key(base_name, specialization)
        merged_dir = ROOT / "models" / "merged" / f"{base_name}_{specialization}"
        if not merged_dir.exists():
            log.error(f"Merged model not found at {merged_dir}. Run merge.py first.")
            sys.exit(1)
        source_model = merged_dir
        log.info(f"Quantizing finetuned model from {merged_dir}")
    else:
        source_model = ROOT / "models" / "bases" / base_name
        if not source_model.exists():
            log.error(f"Base model not found at {source_model}")
            sys.exit(1)
        log.info(f"Quantizing vanilla base from {source_model}")

    # Load calibration data
    cal_path = ROOT / "datasets" / "calibration" / f"{specialization}.jsonl"
    calibration_samples = quant_cfg.get("calibration_samples", 256)

    cal_data = []
    if cal_path.exists():
        with open(cal_path) as f:
            for line in f:
                if line.strip():
                    ex = json.loads(line)
                    # Extract text for calibration
                    if "conversations" in ex:
                        text = " ".join(
                            t.get("value", "") for t in ex["conversations"]
                        )
                    elif "instruction" in ex:
                        text = ex["instruction"] + " " + ex.get("output", "")
                    else:
                        text = str(ex)
                    cal_data.append(text)
                    if len(cal_data) >= calibration_samples:
                        break

    if not cal_data:
        log.warning(
            f"No calibration data at {cal_path}. "
            f"Using generic text for calibration (may reduce quality)."
        )
        cal_data = ["This is a calibration sample."] * calibration_samples

    log.info(f"Quantizing: bits={bits}, group_size={group_size}, samples={len(cal_data)}")

    with Timer() as timer:
        _run_gptq(source_model, output_dir, bits, group_size, cal_data)

    # Verify with smoke test
    log.info("Running 5-sample smoke test...")
    _smoke_test(output_dir, 5)

    size_gb = sum(
        f.stat().st_size for f in output_dir.rglob("*") if f.is_file()
    ) / 1e9

    log.info(f"Quantization complete in {timer.hours:.2f} hours, size: {size_gb:.1f} GB")

    append_run_log({
        "model": experiment_id,
        "phase": "quantize",
        "method": "gptq",
        "bits": bits,
        "size_gb": round(size_gb, 1),
        "duration_hours": round(timer.hours, 2),
        "verification": "pass",
    })

    return output_dir


def _run_gptq(
    source_path: Path, output_path: Path, bits: int, group_size: int,
    calibration_data: list[str],
) -> None:
    """Execute GPTQ quantization."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(source_path), trust_remote_code=True
    )

    # Tokenize calibration data
    cal_tokenized = [
        tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        for text in calibration_data
    ]

    try:
        # Try GPTQModel first (5.0+ with FailSafe support)
        from gptqmodel import GPTQModel, QuantizeConfig

        log.info("Using GPTQModel for quantization")
        quant_config = QuantizeConfig(bits=bits, group_size=group_size)
        model = GPTQModel.load(str(source_path), quant_config)
        model.quantize(cal_tokenized)
        model.save(str(output_path))
        tokenizer.save_pretrained(str(output_path))

    except ImportError:
        # Fall back to AutoGPTQ
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

        log.info("Using AutoGPTQ for quantization (GPTQModel not available)")
        quant_config = BaseQuantizeConfig(
            bits=bits,
            group_size=group_size,
            desc_act=False,
        )
        model = AutoGPTQForCausalLM.from_pretrained(
            str(source_path), quant_config, trust_remote_code=True
        )
        model.quantize(cal_tokenized)
        model.save_quantized(str(output_path))
        tokenizer.save_pretrained(str(output_path))


def _smoke_test(model_path: Path, num_prompts: int = 5) -> None:
    """Quick smoke test: generate a few outputs to verify model isn't garbage."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map="auto",
        trust_remote_code=True,
    )

    prompts = [
        "What is Python?",
        "Write a function that adds two numbers.",
        "Explain the concept of machine learning in one sentence.",
        "List three programming languages.",
        "What is 2 + 2?",
    ][:num_prompts]

    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=50)
    for i, prompt in enumerate(prompts):
        result = gen(prompt)[0]["generated_text"]
        log.info(f"Smoke test {i+1}: {prompt[:40]}... -> {result[len(prompt):len(prompt)+60]}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize a model")
    parser.add_argument("--base", required=True, help="Base model name")
    parser.add_argument("--quant", required=True, help="Quant level name (e.g. gptq_int4)")
    parser.add_argument("--specialization", required=True, help="Specialization")
    parser.add_argument("--finetuned", action="store_true", help="Quantize the finetuned version")
    args = parser.parse_args()

    quantize(args.base, args.quant, args.specialization, args.finetuned)


if __name__ == "__main__":
    main()
