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

## Key Files

- `plan.md` — Full spec (the source of truth)
- `config/models.yaml` — Model registry (bases, datasets, training methods)
- `config/experiments.yaml` — Experiment matrix (base x quant x specialization)
- `config/train/` — Training config templates (base_qlora, base_moe, base_dense_lora) + per-specialization overrides
- `config/serve/instances.yaml` — vLLM instance configs + session presets
- `scripts/run_experiments.py` — Main entry point (fire-and-forget, resumable)
- `scripts/serve.sh` — vLLM launch wrapper

## Quick Start

```bash
# 1. Install deps (Phase 0)
# 2. Validate pipeline end-to-end with smallest experiment:
python3 scripts/run_experiments.py --filter "qwen3.5-9b_gptq-int4_chat*" --resume

# 3. Full matrix:
python3 scripts/run_experiments.py --resume

# 4. Serve models:
./scripts/serve.sh --session coding
```

## Training Methods

| Architecture | Size | Method | Template |
|---|---|---|---|
| Dense | <= 14B | bf16 LoRA | `base_dense_lora.yaml` |
| Dense | 27B+ | QLoRA | `base_qlora.yaml` |
| MoE | any | bf16 LoRA | `base_moe.yaml` |

## Shared Training Cache

Training depends on `(base_model, specialization)`, NOT quant level. Multiple quant variants of the same base+spec reuse one adapter. This cuts training runs roughly in half.

## State & Resumability

- `logs/experiment_state.jsonl` — append-only, last entry per ID wins
- `logs/runs.jsonl` — audit trail of all pipeline steps
- Kill and restart at any point — completed steps are skipped

## Dependencies

torch (ROCm), transformers, accelerate, peft, bitsandbytes, datasets, huggingface_hub, auto-gptq, gptqmodel, axolotl, datasketch, nltk, lm-eval, pyyaml, tqdm, rich
