# Project: llm-training

Local LLM fine-tuning pipeline for Strix Halo (128GB unified RAM).

## What This Is

Automated pipeline to fine-tune, quantize, and serve specialized LLM models:
- **Game Dev** — Unreal/Godot coding (Qwen3.6-27B, QLoRA via unsloth)
- **Shader** — GLSL/HLSL/shader pipeline (Devstral-Small-2-24B)
- **FIM** — autocomplete/fill-in-middle (Qwen2.5-Coder-1.5B)
- **Companion** — VSCode side-by-side (Qwen3.5-9B, held for Qwen3.6-9B)
- **Creative** (GLM-4-9B)
- **Chat** (GLM-4-9B)

## Hardware

- **Training + Inference:** Strix Halo iGPU (128GB unified RAM, RDNA 3.5)
- **Quantization:** TBD — GPTQ if vLLM supports the model, GGUF otherwise
- **Serving:** vLLM or llama-server depending on model support

## Project Structure

```
~/git/llm-training/
├── pyproject.toml                  <- uv project (deps, entry points)
├── src/llm_training/               <- Python package
│   ├── run_experiments.py          <- Main entry point
│   ├── common.py                   <- Shared utilities
│   ├── download.py, curate.py, train.py, merge.py, quantize.py, eval.py
│   └── generate_report.py
├── scripts/
│   ├── serve.sh                    <- vLLM launch wrapper
│   └── generate/                   <- Dataset generation scripts
├── config/                         <- All YAML configs
├── models/, datasets/, logs/       <- Runtime data (gitignored)
└── plan.md                         <- Full spec
```

## Dataset Layout

Raw generation outputs: `datasets/raw/<name>/`
Curate-ready (flattened): `datasets/raw/<name>/training/`

`curate.py` reads from `training/` subdirectories. Raw response files (with nested
structures, metadata wrappers) stay in the parent dir for regeneration/debugging.

## Setup

```bash
cd ~/git/llm-training
uv sync                  # install deps (torch auto-resolves from ROCm index)
```

## Quick Start

```bash
# Curate datasets:
uv run llm-curate --model game-dev
uv run llm-curate --model shader
uv run llm-curate --model fim

# Download base models:
uv run llm-download --base qwen3.6-27b

# Train:
uv run llm-train --base qwen3.6-27b --specialization game-dev

# Full matrix:
uv run llm-run --resume
```

## Key Configs

- `config/models.yaml` — Model registry (bases, datasets, training methods)
- `config/experiments.yaml` — Experiment matrix (base x quant x specialization)
- `config/train/` — Training templates + per-specialization overrides
- `config/serve/instances.yaml` — vLLM instance configs + session presets

## Training Methods

| Model | Size | Method | Tool |
|---|---|---|---|
| Qwen3.6-27B | 27B | QLoRA | unsloth |
| Devstral-Small-2 | 24B | QLoRA | peft |
| Qwen2.5-Coder-1.5B | 1.5B | bf16 LoRA | peft |
| Qwen3.5-9B | 9B | bf16 LoRA | peft |
| GLM-4-9B | 9B | bf16 LoRA | peft |

## Shared Training Cache

Training depends on `(base_model, specialization)`, NOT quant level. Multiple quant variants reuse one adapter.

## State & Resumability

- `logs/experiment_state.jsonl` — append-only, last entry per ID wins
- `logs/runs.jsonl` — audit trail of all pipeline steps
- Kill and restart at any point — completed steps are skipped

## Package Manager

`uv` — lock file: `uv.lock`, Python 3.12+
