# Project: llm-training

Local LLM fine-tuning pipeline for Strix Halo (128GB unified RAM).

## What This Is

Automated pipeline to fine-tune, quantize, and serve specialized LLM models:
- **Code** (Qwen2.5-Coder-32B/14B)
- **Creative** (GLM-4 / Qwen3.5-9B)
- **Research** (Qwen3.5-35B-A3B MoE, disabled by default)
- **Chat** (GLM-4 / Qwen3.5-9B)

## Hardware

- **Training + Inference:** Strix Halo iGPU (128GB unified RAM, RDNA 3.5)
- **Quantization:** GPTQ INT4/INT8 only (vLLM on ROCm constraint)
- **Serving:** vLLM >= 0.11.0, one instance per model

## Project Structure

```
~/git/llm-training/
├── pyproject.toml                  ← uv project (deps, entry points)
├── src/llm_training/               ← Python package
│   ├── run_experiments.py          ← Main entry point
│   ├── common.py                   ← Shared utilities
│   ├── download.py, curate.py, train.py, merge.py, quantize.py, eval.py
│   └── generate_report.py
├── scripts/serve.sh                ← vLLM launch wrapper
├── config/                         ← All YAML configs
├── models/, datasets/, logs/       ← Runtime data (gitignored)
└── plan.md                         ← Full spec
```

## Setup

```bash
cd ~/git/llm-training
uv sync                  # install deps (torch auto-resolves from ROCm index)
```

## Quick Start

```bash
# Validate pipeline end-to-end with smallest experiment:
uv run llm-run --filter "qwen3.5-9b_gptq-int4_chat*" --resume

# Full matrix:
uv run llm-run --resume

# Individual steps:
uv run llm-download --base qwen3.5-9b
uv run llm-curate --model chat
uv run llm-train --base qwen3.5-9b --specialization chat
uv run llm-merge --base qwen3.5-9b --specialization chat
uv run llm-quantize --base qwen3.5-9b --quant gptq_int4 --specialization chat --finetuned
uv run llm-eval --experiment-id qwen3.5-9b_gptq-int4_chat_finetuned
uv run llm-report

# Serve models:
./scripts/serve.sh --session coding
```

## Key Configs

- `config/models.yaml` — Model registry (bases, datasets, training methods)
- `config/experiments.yaml` — Experiment matrix (base x quant x specialization)
- `config/train/` — Training templates + per-specialization overrides
- `config/serve/instances.yaml` — vLLM instance configs + session presets

## Training Methods

| Architecture | Size | Method | Template |
|---|---|---|---|
| Dense | <= 14B | bf16 LoRA | `base_dense_lora.yaml` |
| Dense | 27B+ | QLoRA | `base_qlora.yaml` |
| MoE | any | bf16 LoRA | `base_moe.yaml` |

## Shared Training Cache

Training depends on `(base_model, specialization)`, NOT quant level. Multiple quant variants reuse one adapter. Cuts training runs roughly in half.

## State & Resumability

- `logs/experiment_state.jsonl` — append-only, last entry per ID wins
- `logs/runs.jsonl` — audit trail of all pipeline steps
- Kill and restart at any point — completed steps are skipped

## Package Manager

`uv` — lock file: `uv.lock`, Python 3.12+
