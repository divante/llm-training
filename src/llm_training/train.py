"""Fine-tune models using QLoRA or bf16 LoRA.

Auto-selects training method based on architecture and train_method in config.

Usage:
    python scripts/train.py --base qwen3.5-9b --specialization chat
    python scripts/train.py --base qwen2.5-coder-32b --specialization code
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm_training.common import (
    ROOT,
    Timer,
    append_run_log,
    load_experiments_config,
    load_yaml,
    log,
    merge_training_config,
    resolve_base_template,
    training_cache_key,
)


def load_training_data(specialization: str) -> Path:
    """Resolve training data path for a specialization."""
    # First check if there's a processed dataset for this specialization
    processed = ROOT / "datasets" / "processed" / specialization / "train.jsonl"
    if processed.exists():
        return processed

    # Check models.yaml for the model that matches this specialization
    # (specialization IDs match model IDs in models.yaml)
    log.error(
        f"No training data found at {processed}. "
        f"Run curate.py first, or provide data for specialization '{specialization}'."
    )
    sys.exit(1)


def train(
    base_name: str,
    specialization: str,
    output_dir: Path | None = None,
) -> Path:
    """Run training. Returns path to the saved adapter.

    Uses shared training cache: training depends on (base, specialization),
    not on quant level. Multiple quant variants reuse the same adapter.
    """
    cache_key = training_cache_key(base_name, specialization)
    adapter_dir = output_dir or (ROOT / "models" / "lora" / cache_key)

    # Check shared cache
    if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
        log.info(
            f"Adapter already exists at {adapter_dir} (shared cache hit). "
            f"Skipping training."
        )
        return adapter_dir

    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Load configs
    exp_cfg = load_experiments_config()
    base_cfg = exp_cfg["bases"][base_name]
    architecture = base_cfg["architecture"]
    train_method = base_cfg["train_method"]

    # Resolve and merge training config
    base_template_path = resolve_base_template(architecture, train_method)
    base_template = load_yaml(base_template_path)

    spec_config_path = ROOT / "config" / "train" / f"{specialization}.yaml"
    spec_config = load_yaml(spec_config_path) if spec_config_path.exists() else {}
    train_cfg = merge_training_config(base_template, spec_config, base_name)

    # Resolve base model path
    base_model_path = ROOT / "models" / "bases" / base_name
    if not base_model_path.exists():
        log.error(f"Base model not found at {base_model_path}. Run download.py first.")
        sys.exit(1)

    # Resolve training data
    train_data = load_training_data(specialization)

    import torch
    log.info(f"Training config:")
    log.info(f"  Base: {base_name} ({base_cfg['hf_id']})")
    log.info(f"  Architecture: {architecture}")
    log.info(f"  Method: {train_method}")
    log.info(f"  Specialization: {specialization}")
    log.info(f"  Data: {train_data}")
    log.info(f"  Output: {adapter_dir}")
    log.info(f"  Torch version: {torch.__version__}")
    log.info(f"  Torch HIP version: {torch.version.hip}")

    training_params = train_cfg.get("training", {})
    lora_params = train_cfg.get("lora", {})

    log.info(f"  Epochs: {training_params.get('num_epochs', 3)}")
    log.info(f"  Batch size: {training_params.get('batch_size', 4)}")
    log.info(f"  Gradient accum: {training_params.get('gradient_accumulation', 4)}")
    log.info(f"  Max seq length: {training_params.get('max_seq_length', 4096)}")
    log.info(f"  LoRA r: {lora_params.get('r', 64)}, alpha: {lora_params.get('alpha', 128)}")

    with Timer() as timer:
        _run_training(
            base_model_path=base_model_path,
            train_data_path=train_data,
            output_dir=adapter_dir,
            train_cfg=train_cfg,
            architecture=architecture,
            train_method=train_method,
            base_name=base_name,
            specialization=specialization,
        )

    log.info(f"Training completed in {timer.hours:.1f} hours")

    append_run_log({
        "model": f"{base_name}_{specialization}",
        "phase": "train",
        "train_method": train_method,
        "architecture": architecture,
        "epochs": training_params.get("num_epochs", 3),
        "duration_hours": round(timer.hours, 2),
    })

    return adapter_dir


def _detect_device() -> tuple[str, bool]:
    """Return (device_type, bf16_supported). device_type: 'cuda', 'rocm', 'cpu'."""
    import torch
    if torch.cuda.is_available():
        # PyTorch ROCm presents as CUDA
        is_rocm = torch.version.hip is not None
        device_type = "rocm" if is_rocm else "cuda"
        try:
            bf16_ok = torch.cuda.is_bf16_supported()
        except Exception:
            bf16_ok = is_rocm  # ROCm RDNA3+ supports bf16
        return device_type, bf16_ok
    return "cpu", False


def _run_training(
    base_model_path: Path,
    train_data_path: Path,
    output_dir: Path,
    train_cfg: dict,
    architecture: str,
    train_method: str,
    base_name: str,
    specialization: str,
) -> None:
    """Execute the actual training loop."""
    import torch
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )

    device_type, bf16_supported = _detect_device()
    log.info(f"Device: {device_type}, bf16: {bf16_supported}")

    training_params = train_cfg.get("training", {})
    lora_params = train_cfg.get("lora", {})
    quant_params = train_cfg.get("quantization", {})

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path), trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_seq_length = training_params.get("max_seq_length", 4096)

    # Determine dtype
    if bf16_supported:
        compute_dtype = torch.bfloat16
    elif device_type != "cpu":
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    # Load model — use 4-bit quantization on CUDA only (BNB doesn't support ROCm reliably)
    # ROCm + accelerate device_map="auto" causes SIGSEGV; use explicit single-device instead
    if device_type == "cuda":
        device_map = "auto"
    elif device_type == "rocm":
        device_map = {"": 0}  # force single GPU, avoids accelerate SIGSEGV on ROCm
    else:
        device_map = "cpu"

    load_kwargs: dict = {
        "pretrained_model_name_or_path": str(base_model_path),
        "trust_remote_code": True,
        "device_map": device_map,
        "dtype": compute_dtype,
    }

    use_bnb_4bit = (
        train_method == "qlora"
        and quant_params.get("load_in_4bit", False)
        and device_type == "cuda"
    )

    if use_bnb_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quant_params.get("bnb_4bit_quant_type", "nf4"),
        )
        load_kwargs["quantization_config"] = bnb_config
        del load_kwargs["dtype"]
        log.info("Loading with 4-bit BNB quantization (QLoRA)")
    else:
        log.info(f"Loading in {compute_dtype} (full precision LoRA)")

    log.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

    # Configure LoRA
    target_modules = lora_params.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    lora_config = LoraConfig(
        r=lora_params.get("r", 64),
        lora_alpha=lora_params.get("alpha", 128),
        target_modules=target_modules,
        modules_to_save=lora_params.get("modules_to_save", None),
        lora_dropout=lora_params.get("dropout", 0.05),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and tokenize training data
    log.info(f"Loading training data from {train_data_path}...")
    train_examples = []
    with open(train_data_path) as f:
        for line in f:
            if line.strip():
                train_examples.append(json.loads(line))

    log.info(f"Loaded {len(train_examples)} training examples")

    def tokenize_example(example: dict) -> dict:
        if "conversations" in example:
            text_parts = []
            for turn in example["conversations"]:
                role = "user" if turn["from"] == "human" else "assistant"
                text_parts.append(f"<|{role}|>\n{turn['value']}")
            text = "\n".join(text_parts)
        elif "instruction" in example:
            text = f"<|user|>\n{example['instruction']}"
            if example.get("input"):
                text += f"\n{example['input']}"
            text += f"\n<|assistant|>\n{example.get('output', '')}"
        else:
            text = example.get("text", "")
        return tokenizer(text, truncation=True, max_length=max_seq_length, padding=False)

    from datasets import Dataset

    dataset = Dataset.from_list(train_examples)
    tokenized = dataset.map(
        tokenize_example,
        remove_columns=dataset.column_names,
        num_proc=4,
        desc="Tokenizing",
    )

    effective_batch = (
        training_params.get("batch_size", 4)
        * training_params.get("gradient_accumulation", 4)
    )
    log.info(f"Effective batch size: {effective_batch}")

    log_dir = ROOT / "logs" / "training" / f"{base_name}_{specialization}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # paged_adamw_8bit requires BNB paged memory (CUDA only); ROCm uses fused adamw
    configured_optimizer = train_cfg.get("optimizer", "paged_adamw_8bit")
    if configured_optimizer == "paged_adamw_8bit" and (not use_bnb_4bit or device_type == "rocm"):
        optimizer = "adamw_torch_fused" if device_type != "cpu" else "adamw_torch"
    else:
        optimizer = configured_optimizer

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_params.get("num_epochs", 3),
        per_device_train_batch_size=training_params.get("batch_size", 4),
        gradient_accumulation_steps=training_params.get("gradient_accumulation", 4),
        gradient_checkpointing=training_params.get("gradient_checkpointing", True),
        learning_rate=training_params.get("learning_rate", 2e-4),
        lr_scheduler_type=training_params.get("lr_scheduler", "cosine"),
        warmup_ratio=training_params.get("warmup_ratio", 0.03),
        optim=optimizer,
        logging_dir=str(log_dir),
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        bf16=(compute_dtype == torch.bfloat16 and device_type != "cpu"),
        fp16=(compute_dtype == torch.float16 and device_type != "cpu"),
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    log.info("Starting training...")
    trainer.train()

    log.info(f"Saving adapter to {output_dir}...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    log.info("Training complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--base", required=True, help="Base model name from experiments.yaml")
    parser.add_argument("--specialization", required=True, help="Specialization (code, creative, research, chat)")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    args = parser.parse_args()

    train(args.base, args.specialization, args.output_dir)


if __name__ == "__main__":
    main()
