# Local LLM Training Plan

> **Hardware:** Strix Halo (128GB unified RAM, RDNA 3.5 iGPU) — training AND inference
> **Quantization:** GPTQ (AutoGPTQ / GPTQModel 5.0+, post-merge)
> **Inference:** vLLM (≥0.11.0) on Strix Halo iGPU — one instance per model
> **Training (dense):** QLoRA or bf16 LoRA on Strix Halo (128GB) → merge → GPTQ quantize
> **Training (MoE):** bf16 LoRA on Strix Halo (128GB) → adapter-based serving OR careful merge+GPTQ
> **Last updated:** 2026-04-05

---

## Agent Instructions

**This document is the complete spec.** It contains everything needed to build and run the
training pipeline. Follow these steps:

### How to execute this plan

1. **Read the entire document first.** Understand the Strategy, Models, Training Configs,
   Experiment Matrix, and the Runner before writing any code.

2. **Phase 0 — Environment Setup.** Create the directory tree, install dependencies,
   validate hardware. See the "Directory Structure" and "Phase 0" sections.

3. **Implement the scripts.** The "Claude Code Execution Guide" and "Experiment Runner"
   sections contain pseudocode/specs for each script. Implement them as Python modules
   in `src/llm_training/`. Key modules:
   - `run_experiments.py` — main entry point (fire-and-forget, resumable)
   - `download.py`, `curate.py`, `train.py`, `merge.py`, `quantize.py`, `eval.py`
   - `generate_report.py` — comparison tables
   - `scripts/serve.sh` — vLLM launch wrapper (stays as shell script)

4. **Write `config/models.yaml`** from the "Model Registry" section.
   Write `config/experiments.yaml` from the "Experiment Matrix" section.
   Write `config/train/base_qlora.yaml` and `config/train/base_moe.yaml` from the
   "Training Config Templates" section.

5. **Run Phase 1 (download) + Phase 2 (curate)** for the HuggingFace datasets.
   Skip datasets marked `custom: true` — those require manual curation and are not
   available yet. The pipeline must handle missing custom datasets gracefully
   (warn and continue with available data).

6. **Start the experiment runner:**
   ```bash
   uv run llm-run --resume --filter "*chat*"
   ```
   This validates the pipeline end-to-end on the smallest experiment set.
   Once chat experiments pass, remove the filter for the full run.

### Key constraints

- **vLLM only.** GPTQ INT4 and INT8 are the only active quant levels.
  GGUF/llama.cpp code paths should exist in the scripts (branching on
  `quant_level.backend`) but are not active. See the deferred entries
  in `config/experiments.yaml`.
- **Training on Strix Halo iGPU (128GB unified).** Dense models use QLoRA
  (or bf16 LoRA for smaller models — memory is not the constraint, compute is).
  MoE models use bf16 LoRA (NOT QLoRA — BNB breaks MoE expert routing).
- **Shared training cache.** Training depends on `(base, specialization)`,
  not quant level. Reuse adapters across quant variants of the same base+spec.
- **Resumable.** Every step writes state to `logs/experiment_state.jsonl`.
  On restart, skip completed steps. Never let one experiment failure kill the run.
- **Custom datasets may be missing.** If a dataset has `custom: true` and its
  local path is empty, log a warning and proceed with available datasets.
  The model can be retrained later when custom data is ready.

---

## Strategy

**Multiple specialized models, hot-swapped on Strix Halo.**

Rationale: iGPU throughput is the bottleneck, not memory. Smaller specialized GPTQ-4bit models run faster and perform better in-domain than one large generalist. 128GB gives massive headroom for context, KV cache, and running multiple vLLM instances.

MoE models are particularly well-suited to Strix Halo — they have large total param counts (need lots of memory) but only activate a fraction per token (lower compute per token). Unified RAM + memory bandwidth = ideal MoE inference.

**Dense vs MoE decision:** Use MoE when the active parameter count gives you better quality-per-FLOP than an equivalently-sized dense model. For example, Qwen3.5-35B-A3B (3B active) outperforms many 7-9B dense models while being faster to infer. Use dense when you need coding-specialized models (Qwen2.5-Coder) that don't have MoE equivalents yet.

**Training:** All training runs on Strix Halo iGPU (128GB unified RAM). This is compute-limited (iGPU is slower per step than discrete), but memory-rich. QLoRA is still useful for larger dense models (27B+) to reduce compute per step. For smaller dense models (≤14B), bf16 LoRA is feasible and avoids quantization artifacts during training.

**Serving:** Each model runs in its own vLLM instance. Load/unload models as needed for different sessions.

**Session rotation examples:**
- Coding session → Code + Research instances
- Game dev session → Code + Creative instances
- Planning session → Creative + Research instances
- Always-on → Chat instance

---

## Models

Each model entry defines: base, target size after quant, use cases, and datasets.
Add/remove entries freely. The `enabled` flag controls what's actively in the pipeline.

### Model: Code

| Field | Value |
|-------|-------|
| **ID** | `code` |
| **Enabled** | yes |
| **Architecture** | dense |
| **Base** | `Qwen/Qwen2.5-Coder-32B-Instruct` |
| **Fallback base** | `Qwen/Qwen2.5-Coder-14B-Instruct` (if 32B throughput is too slow on iGPU) |
| **Qwen3.5 upgrade path** | `Qwen/Qwen3-Coder-*` when available. Qwen3.5-27B dense is an option but lacks Coder-specific pretraining. Monitor HF for Qwen3.5-Coder releases. |
| **Training method** | QLoRA (32B — reduce compute) or bf16 LoRA (14B — fits in 128GB) |
| **GPTQ size** | ~18GB (32B) / ~8GB (14B) |
| **Use cases** | General coding (Python, Bash, TypeScript), game engine code (UE C++, Unity C#, GDScript), shaders, build systems, CI/CD |

**Datasets:**

| Dataset | Source | Purpose | Est. Size |
|---------|--------|---------|-----------|
| glaive-code-assistant-v3 | HuggingFace | Multi-language instruction/completion pairs | ~130K |
| CodeFeedback-Filtered-Instruction | HuggingFace | Multi-turn code refinement, debugging | ~60K |
| CommitPackFT | HuggingFace (bigcode) | Git-aware coding: diffs, commit messages | ~50K |
| the-stack-v2-dedup (filtered) | HuggingFace (bigcode) | Continued pretraining — Python/Bash/TS only | Large (sample ~100K) |
| _Custom: Game Engine Docs_ | Manual curation | UE C++ API, Unity C# scripting, GDScript reference → instruction format | ~20-30K |
| _Custom: Shader/Pipeline_ | Manual curation | HLSL/GLSL snippets + explanations → instruction format | ~5-10K |
| _Custom: Game Dev GitHub_ | Synthetic generation | Game dev code samples → Q&A pairs via strong model | ~10-20K |

**Notes:**
- Game dev datasets don't exist pre-made at quality. Budget time for curation.
- For the-stack-v2, filter aggressively: only files with docstrings/comments, skip minified/generated code.
- Total target: ~50K-100K high-quality examples after dedup and filtering.

---

### Model: Creative/Planning

| Field | Value |
|-------|-------|
| **ID** | `creative` |
| **Enabled** | yes |
| **Architecture** | dense (or MoE — see alt base) |
| **Base** | `THUDM/glm-4-9b-chat` (GLM-4.7-Flash) |
| **Alt base (MoE)** | `Qwen/Qwen3.5-35B-A3B` — 35B total, 3B active. Rivals GLM-4 quality, native thinking mode. ~10-12GB GPTQ-4bit. |
| **Alt base (dense)** | `Qwen/Qwen3.5-9B` — rivals Qwen2.5-72B benchmarks. Native 262K context. |
| **Training method** | QLoRA (dense) or bf16 LoRA (MoE) |
| **GPTQ size** | ~5GB (GLM-4) / ~10-12GB (Qwen3.5-35B-A3B MoE) / ~5GB (Qwen3.5-9B) |
| **Use cases** | Creative writing, game lore/narrative/store copy, project planning, PRDs, roadmaps, structured documents |

**Datasets:**

| Dataset | Source | Purpose | Est. Size |
|---------|--------|---------|-----------|
| lmsys-chat-1m (filtered) | HuggingFace (lmsys) | Creative/writing conversations only | ~30K after filter |
| cosmopedia-v2 | HuggingFace (HuggingFaceTB) | High-quality structured prose, knowledge articles | ~50K sample |
| OpenOrca (filtered) | HuggingFace | Analytical/planning tasks subset | ~30K after filter |
| UltraFeedback | HuggingFace | Preference-tuned quality reasoning | ~60K |
| _Custom: Game Lore_ | Synthetic generation | Wiki lore (Witcher, ME, TES, etc.) → "write lore for X with Y constraints" | ~10-15K |
| _Custom: Project Plans_ | Synthetic generation | Real PRDs/roadmaps → instruction pairs | ~5-10K |
| _Custom: Store/Marketing_ | Manual curation | Game store pages, descriptions, patch notes → instruction format | ~5K |

**Notes:**
- GLM-4.7-Flash is already strong here. Fine-tuning is for steering style and domain.
- Total target: ~30K-50K examples.

---

### Model: Research

| Field | Value |
|-------|-------|
| **ID** | `research` |
| **Enabled** | no _(start without this — vanilla Qwen3.5 is already strong at research; enable later if needed)_ |
| **Architecture** | MoE preferred |
| **Base** | `Qwen/Qwen3.5-35B-A3B` (35B total, 3B active — strong reasoning with thinking mode) |
| **Alt base (larger)** | `Qwen/Qwen3.5-122B-A10B` — 122B total, 10B active. ~68GB GPTQ-4bit. Fits Strix Halo but tight for pairs. Best reasoning quality. |
| **Alt base (dense)** | `Qwen/Qwen2.5-14B-Instruct` (original pick, still solid) |
| **Training method** | bf16 LoRA (MoE) or QLoRA (dense) |
| **GPTQ size** | ~10-12GB (35B-A3B) / ~68GB (122B-A10B) / ~8GB (Qwen2.5-14B) |
| **Use cases** | Exploration, analysis, summarization, documentation, deep-dive reasoning |

**Datasets:**

| Dataset | Source | Purpose | Est. Size |
|---------|--------|---------|-----------|
| FLAN-v2 (reasoning subset) | HuggingFace | Chain-of-thought, multi-step reasoning | ~50K sample |
| SlimOrca-Dedup | HuggingFace | Clean reasoning-focused instruction pairs | ~500K (sample ~50K) |
| OpenOrca (full) | HuggingFace | Broad reasoning coverage | ~1M (sample ~50K) |
| UltraFeedback | HuggingFace | Quality calibration | ~60K |

**Notes:**
- Only train this if vanilla Qwen2.5-14B instruct doesn't meet your bar.
- If you skip this model, the Creative model can absorb some research tasks with broader dataset mix.
- Total target: ~50K examples.

---

### Model: Chat

| Field | Value |
|-------|-------|
| **ID** | `chat` |
| **Enabled** | yes |
| **Architecture** | dense or MoE |
| **Base** | `THUDM/glm-4-9b-chat` (GLM-4.7-Flash) |
| **Alt base (MoE)** | `Qwen/Qwen3.5-35B-A3B` — more capable, fits 24GB GPTQ-4bit with room to spare |
| **Alt base (dense)** | `Qwen/Qwen3.5-9B` — direct upgrade over GLM-4, native thinking mode |
| **Training method** | QLoRA (dense) or bf16 LoRA (MoE) |
| **GPTQ size** | ~5GB (GLM-4) / ~10-12GB (35B-A3B) / ~5GB (Qwen3.5-9B) |
| **Hardware** | Strix Halo iGPU — own vLLM instance, can run alongside other models |
| **Use cases** | General assistant, personal chat, quick lookups, casual conversation |

**Datasets:**

| Dataset | Source | Purpose | Est. Size |
|---------|--------|---------|-----------|
| lmsys-chat-1m | HuggingFace (lmsys) | Broad conversational coverage | ~100K sample |
| ShareGPT-Vicuna-unfiltered | HuggingFace | Natural multi-turn conversation style | ~90K |
| Capybara | HuggingFace (LDJnr) | Multi-turn, natural assistant behavior | ~16K |
| _Custom: Personal style_ | Manual curation | Your own chat preferences, tone, personality quirks | ~1-5K |

**Notes:**
- Runs on Strix Halo alongside other models (own vLLM instance, small memory footprint).
- Personal style dataset is small but high-impact — even 500 examples of preferred tone/responses makes a difference.
- Total target: ~30K-50K examples.

---

## Training Config Templates

Two templates: one for dense models (QLoRA), one for MoE models (bf16 LoRA).
The `architecture` field in `config/models.yaml` determines which template to use.

### Template A: Dense Models (QLoRA)

For: Qwen2.5-Coder, GLM-4, Qwen3.5 dense (9B, 27B)

```yaml
# QLoRA settings (Strix Halo iGPU, 128GB unified)
# Use QLoRA for larger dense models (27B+) to reduce compute per step.
# For smaller dense models (≤14B), consider bf16 LoRA instead (Template C below).
method: qlora
lora:
  r: 64
  alpha: 128
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  dropout: 0.05

quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: bfloat16  # verify ROCm bf16 support; fallback to fp16
  bnb_4bit_quant_type: nf4

training:
  num_epochs: 3
  batch_size: 4             # 128GB unified = generous headroom; increase for smaller models
  gradient_accumulation: 4   # effective batch = batch_size × accum = 16
  gradient_checkpointing: true  # still recommended to save memory for KV cache
  learning_rate: 2e-4
  lr_scheduler: cosine
  warmup_ratio: 0.03
  max_seq_length: 8192      # 128GB allows longer contexts comfortably

optimizer: paged_adamw_8bit

# Post-training
merge: true  # merge LoRA weights into base
quantize_output:
  method: gptq
  bits: 4
  group_size: 128
  calibration_samples: 256  # from same domain as training data
```

### Template B: MoE Models (bf16 LoRA)

For: Qwen3.5-35B-A3B, Qwen3.5-122B-A10B, any MoE architecture.

**Why not QLoRA for MoE:** BitsAndBytes 4-bit quantization breaks expert routing in MoE models. Expert gating weights get corrupted at 4-bit precision, leading to training instability and degraded expert selection. Use bf16 LoRA instead — it fits on 24GB for MoE models where active params are small (3-10B).

```yaml
# bf16 LoRA settings for MoE (Strix Halo iGPU, 128GB unified)
method: lora
lora:
  r: 32                    # lower rank than dense — MoE has more params to update
  alpha: 64
  # Target shared attention + expert FFN layers
  # Shared layers (always active):
  #   q_proj, k_proj, v_proj, o_proj
  # Expert layers (per-expert, only active experts update):
  #   gate_proj, up_proj, down_proj (inside each expert)
  # Router (gating network):
  #   gate (the expert selector — train this to steer specialization)
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "gate"]
  modules_to_save: ["gate"]  # always save router weights — critical for MoE
  dropout: 0.05

quantization:
  load_in_4bit: false       # DO NOT use 4-bit for MoE
  # Model loaded in bf16 (or fp16 fallback)

training:
  num_epochs: 2             # MoE converges faster due to expert specialization
  batch_size: 2             # 128GB unified allows more headroom than discrete 24GB
  gradient_accumulation: 8   # effective batch = 16
  gradient_checkpointing: true  # recommended — MoE models have large param count
  learning_rate: 1e-4       # lower LR — MoE is more sensitive
  lr_scheduler: cosine
  warmup_ratio: 0.05        # longer warmup for stability
  max_seq_length: 4096

optimizer: paged_adamw_8bit

# Post-training — TWO OPTIONS:
# Option 1 (recommended): Keep LoRA adapters separate, serve with adapter
post_training:
  merge: false
  serve_mode: adapter       # vLLM loads base GPTQ + LoRA adapter at runtime
  base_quantized: true      # use pre-quantized base (e.g. Qwen3.5-35B-A3B-GPTQ-Int4 from HF)

# Option 2 (experimental): Merge + requantize with GPTQModel FailSafe
# merge: true
# quantize_output:
#   method: gptq
#   tool: gptqmodel         # NOT auto-gptq — need GPTQModel 5.0+ for FailSafe
#   bits: 4
#   group_size: 128
#   fail_safe: true          # weight-only fallback for non-activated experts
#   calibration_samples: 512  # more samples needed for MoE (expert coverage)
```

### Template C: Small Dense Models (bf16 LoRA)

For: Qwen3.5-9B, GLM-4, Qwen2.5-14B — models ≤14B where 128GB unified RAM allows full bf16 loading.

**Why bf16 LoRA for small dense:** With 128GB unified RAM, there's no reason to quantize during training for models that fit easily. bf16 LoRA avoids quantization artifacts during training while still being parameter-efficient. QLoRA is still preferred for 27B+ dense models to reduce compute per training step.

```yaml
# bf16 LoRA settings for small dense models (Strix Halo, 128GB unified)
method: lora
lora:
  r: 64
  alpha: 128
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  dropout: 0.05

quantization:
  load_in_4bit: false       # full bf16, no quantization during training

training:
  num_epochs: 3
  batch_size: 8             # plenty of room for small models
  gradient_accumulation: 2   # effective batch = 16
  gradient_checkpointing: false  # not needed for ≤14B on 128GB
  learning_rate: 2e-4
  lr_scheduler: cosine
  warmup_ratio: 0.03
  max_seq_length: 8192

optimizer: paged_adamw_8bit

# Post-training
merge: true
quantize_output:
  method: gptq
  bits: 4
  group_size: 128
  calibration_samples: 256
```

---

### MoE Training: Expert Targeting Strategies

Choose based on your goal:

| Strategy | What to train | When to use | Memory usage (35B-A3B) |
|----------|--------------|-------------|------------------------|
| **Router + shared only** | gate, q/k/v/o_proj | Steering the model's general behavior, style | ~15GB (trivial on 128GB) |
| **Router + all experts** | gate, all expert FFN layers | Domain specialization, new knowledge | ~18-20GB |
| **Frozen router, experts only** | expert FFN only | Fine-grained expert knowledge, preserve routing | ~17GB |

For most fine-tuning: **Router + shared** is the sweet spot. Only train experts if you need the model to learn genuinely new domain knowledge (e.g., your custom game engine API). All strategies fit comfortably within 128GB unified RAM.

### MoE Quantization: Post-Fine-Tuning

**The problem:** Standard GPTQ needs activation data to calibrate. In MoE, some experts rarely activate during calibration, producing garbage quantization for those experts.

**Solutions (pick one):**

1. **Adapter-based serving (recommended):**
   - Download pre-quantized base from HF (e.g., `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`)
   - Train LoRA adapter, keep it separate (don't merge)
   - vLLM loads base GPTQ + LoRA adapter at runtime
   - Pro: no requantization risk. Con: slightly higher memory (adapter overhead).

2. **GPTQModel 5.0+ with FailSafe:**
   - Merge LoRA into full-precision model
   - Quantize with `fail_safe=true` — non-activated experts get weight-only quantization
   - Higher error on those experts but they're rarely used anyway
   - Pro: single model file. Con: quality risk on edge cases.

3. **LLM Compressor 0.8+:**
   - Red Hat's tool, supports Qwen3/3.5 MoE natively
   - Better expert-aware calibration than vanilla AutoGPTQ
   - Pro: best quality. Con: more setup, less community adoption.

**Per-model overrides:**

| Model | method | batch_size | max_seq_length | Notes |
|-------|--------|-----------|----------------|-------|
| code (32B dense) | qlora | 4 | 8192 | QLoRA to reduce compute; 128GB handles it |
| code (14B dense) | qlora | 8 | 8192 | Comfortable, can push batch size |
| creative (GLM-4 dense) | qlora | 8 | 4096 | Standard |
| creative (35B-A3B MoE) | bf16 lora | 2 | 4096 | 128GB unified = no memory pressure |
| research (35B-A3B MoE) | bf16 lora | 2 | 4096 | Same as creative MoE |
| research (Qwen2.5-14B dense) | bf16 lora | 8 | 8192 | bf16 feasible at 14B on 128GB |
| chat (GLM-4 dense) | qlora | 8 | 2048 | Short conversations, fast training |
| chat (35B-A3B MoE) | bf16 lora | 2 | 2048 | More headroom with shorter seqlen |
| chat (Qwen3.5-9B dense) | bf16 lora | 8 | 4096 | bf16 feasible at 9B on 128GB |

---

## vLLM Serving Configs

Each model runs in its own vLLM instance on a dedicated port. Start/stop instances
as needed for each session. Requires vLLM ≥0.11.0 for Qwen3.5 support.

```yaml
# config/serve/instances.yaml
# Each model is a standalone vLLM instance. Start the ones you need.

instances:
  code:
    path: ./models/gptq/code/                    # dense, merged GPTQ
    port: 8100
    max_model_len: 8192
    gpu_memory_utilization: 0.45

  creative:
    path: ./models/gptq/creative/                # dense GLM-4 or Qwen3.5-9B
    port: 8101
    max_model_len: 4096
    gpu_memory_utilization: 0.25

  research:
    path: ./models/gptq/research-base/           # pre-quantized Qwen3.5-35B-A3B-GPTQ
    port: 8102
    lora_adapter: ./models/lora/research/        # LoRA adapter (if fine-tuned)
    max_model_len: 8192
    gpu_memory_utilization: 0.35
    enable_expert_parallel: true

  chat:
    path: ./models/gptq/chat/
    port: 8103
    max_model_len: 4096
    gpu_memory_utilization: 0.20

  research-large:                                 # 122B-A10B solo — fills most of 128GB
    path: ./models/gptq/research-large-base/     # Qwen3.5-122B-A10B-GPTQ
    port: 8102
    lora_adapter: ./models/lora/research-large/  # optional
    max_model_len: 8192
    gpu_memory_utilization: 0.80
    enable_expert_parallel: true

# Session presets (which instances to start together)
sessions:
  coding: [code, research]
  gamedev: [code, creative]
  planning: [creative, research]
  research-heavy: [research-large]               # solo, uses most of 128GB
  chat-only: [chat]
```

**vLLM launch flags for MoE models:**
```bash
vllm serve <model_path> \
  --quantization gptq \
  --dtype float16 \
  --enable-lora \                    # if serving with LoRA adapter
  --lora-modules adapter=<adapter_path> \
  --enable-expert-parallel \         # required for MoE
  --max-model-len 8192
```

---

## Dataset Curation Checklist

For each dataset before training:

- [ ] Download and inspect samples (min 100 random examples)
- [ ] Filter for target languages/domains
- [ ] Dedup (exact + near-duplicate via MinHash)
- [ ] Remove low-quality: too short (<50 tokens), too long (>max_seq_length), nonsensical
- [ ] Convert to consistent format (ShareGPT multi-turn or Alpaca instruction)
- [ ] Validate tokenization — check no examples exceed max_seq_length after tokenization
- [ ] Create calibration split for GPTQ quantization (256 for dense, 512 for MoE — more samples needed for expert coverage)
- [ ] Create eval split (500 examples) held out from training

**For synthetic/custom datasets:**
- [ ] Use strong model (Claude, GPT-4) for generation
- [ ] Human-review at least 50 samples per dataset
- [ ] Iterate on generation prompt if quality is off
- [ ] Version the generation prompts alongside the data

---

## Eval Strategy

Run evals before AND after fine-tuning to measure improvement.

| Specialization | Eval Benchmarks | Custom Evals |
|----------------|----------------|--------------|
| code | HumanEval, MBPP, MultiPL-E (Python/TS/Bash) | Game engine code completion (custom set from curated data) |
| creative | AlpacaEval (writing subset), MT-Bench | Lore generation quality (blind A/B vs base), planning structure |
| research | MMLU (subset), ARC-Challenge | Summarization quality on held-out docs |
| chat | MT-Bench, AlpacaEval | Conversational naturalness (manual review) |

---

## Experiment Runner (Fire-and-Forget Automation)

The experiment runner replaces the manual phase-by-phase execution. It processes a matrix of
experiments, tracks state for resumability, and produces a comparison table at the end.

### Experiment Matrix

Each experiment is a unique combination of:

```
(base_model, quant_level, fine_tuned, specialization)
```

Define the full matrix in `config/experiments.yaml`:

```yaml
# config/experiments.yaml
# The runner iterates every combination and runs the pipeline for each.
# Comment out rows you don't want to run.

specializations:
  code:
    datasets_config: "config/train/code.yaml"
    eval_suite: ["humaneval", "mbpp", "multipl_e"]
    custom_eval: "datasets/eval/code.jsonl"
  creative:
    datasets_config: "config/train/creative.yaml"
    eval_suite: ["mt_bench_writing", "alpaca_eval_writing"]
    custom_eval: "datasets/eval/creative.jsonl"
  research:
    datasets_config: "config/train/research.yaml"
    eval_suite: ["mmlu_subset", "arc_challenge"]
    custom_eval: "datasets/eval/research.jsonl"
  chat:
    datasets_config: "config/train/chat.yaml"
    eval_suite: ["mt_bench", "alpaca_eval"]
    custom_eval: "datasets/eval/chat.jsonl"

# Base models to test
bases:
  qwen3.5-9b:
    hf_id: "Qwen/Qwen3.5-9B"
    architecture: dense
    train_method: qlora
  qwen3.5-27b:
    hf_id: "Qwen/Qwen3.5-27B"
    architecture: dense
    train_method: qlora
  qwen3.5-35b-a3b:
    hf_id: "Qwen/Qwen3.5-35B-A3B"
    architecture: moe
    train_method: bf16_lora
    pre_quantized_base: "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
  qwen2.5-coder-32b:
    hf_id: "Qwen/Qwen2.5-Coder-32B-Instruct"
    architecture: dense
    train_method: qlora
  qwen2.5-coder-14b:
    hf_id: "Qwen/Qwen2.5-Coder-14B-Instruct"
    architecture: dense
    train_method: qlora
  glm-4-9b:
    hf_id: "THUDM/glm-4-9b-chat"
    architecture: dense
    train_method: qlora

# ═══════════════════════════════════════════════════════════════════
# QUANTIZATION CONSTRAINTS (Strix Halo iGPU + ROCm)
#
# vLLM on AMD ROCm supports:
#   - GPTQ: INT4 and INT8 ONLY (no 3/5/6-bit)
#   - AWQ:  INT4 ONLY (hard-enforced in vLLM source)
#   - GGUF: all levels (Q2-Q8), BUT ~3x slower than native GPTQ/AWQ
#   - FP8:  NOT on RDNA 3/3.5 (needs MI300+ or CUDA 8.9+)
#   - BNB:  NOT on ROCm at all
#
# Current focus: vLLM + GPTQ only.
# GGUF/llama.cpp support is designed-in but deferred — see quant_levels
# for the extension point.
# ═══════════════════════════════════════════════════════════════════

quant_levels:
  # --- ACTIVE: vLLM native (GPTQ) ---
  gptq_int4: { method: gptq, bits: 4, group_size: 128, backend: vllm }
  gptq_int8: { method: gptq, bits: 8, group_size: 32,  backend: vllm }

  # --- DEFERRED: GGUF via llama.cpp (uncomment to enable) ---
  # When enabling: also build llama.cpp in Phase 0 (see commented step 7)
  # and add models/gguf/ directory.
  # gguf_q3:   { method: gguf, gguf_type: "Q3_K_M", backend: llamacpp }
  # gguf_q4:   { method: gguf, gguf_type: "Q4_K_M", backend: llamacpp }
  # gguf_q5:   { method: gguf, gguf_type: "Q5_K_M", backend: llamacpp }
  # gguf_q6:   { method: gguf, gguf_type: "Q6_K",   backend: llamacpp }
  # gguf_q8:   { method: gguf, gguf_type: "Q8_0",   backend: llamacpp }

# Which experiments to actually run.
# Each entry generates 2 runs: vanilla (no fine-tune) + fine-tuned.
# Format: base × quant × specialization
experiments:

  # ─── 9B vs 27B comparison ───
  - base: qwen3.5-9b
    quants: [gptq_int4, gptq_int8]
    specializations: [code, creative, research, chat]

  - base: qwen3.5-27b
    quants: [gptq_int4]
    specializations: [code, creative, research, chat]

  # ─── MoE comparison ───
  - base: qwen3.5-35b-a3b
    quants: [gptq_int4]       # MoE: adapter-based, base is pre-quantized GPTQ-INT4
    specializations: [creative, research, chat]

  # ─── Coding-specific (Coder models for code specialization) ───
  - base: qwen2.5-coder-32b
    quants: [gptq_int4]
    specializations: [code]

  - base: qwen2.5-coder-14b
    quants: [gptq_int4, gptq_int8]
    specializations: [code]

  # ─── GLM-4 baseline (current pick) ───
  - base: glm-4-9b
    quants: [gptq_int4]
    specializations: [creative, chat]
```

### Experiment State Machine

Each experiment has a deterministic ID and tracks its progress through steps.

**Experiment ID:** `{base}_{quant}_{specialization}_{vanilla|finetuned}`
Example: `qwen3.5-9b_gptq-int8_code_finetuned`

**Steps per experiment:**

```
download → [train → merge] → quantize → benchmark → done
              ↑ skipped for vanilla experiments
```

**State file:** `logs/experiment_state.jsonl`

Each line is one experiment's current state:
```jsonl
{"id": "qwen3.5-9b_q8_code_vanilla", "status": "done", "completed_steps": ["download", "quantize", "benchmark"], "results": {"humaneval": 0.72, "mbpp": 0.68}, "duration_hours": 1.2}
{"id": "qwen3.5-9b_q8_code_finetuned", "status": "done", "completed_steps": ["download", "train", "merge", "quantize", "benchmark"], "results": {"humaneval": 0.81, "mbpp": 0.76}, "duration_hours": 4.5}
{"id": "qwen3.5-27b_q4_code_finetuned", "status": "running", "completed_steps": ["download", "train"], "current_step": "merge", "started_at": "2026-04-06T03:00:00Z"}
{"id": "qwen3.5-27b_q4_creative_vanilla", "status": "pending", "completed_steps": []}
{"id": "qwen3.5-35b-a3b_q4_research_finetuned", "status": "failed", "completed_steps": ["download", "train"], "failed_step": "quantize", "error": "GPTQModel FailSafe: 12 experts had zero activations", "retry_count": 1}
```

### Runner Script

```bash
uv run llm-run [--resume] [--filter "qwen3.5-9b*"] [--dry-run]
```

```python
"""
run_experiments.py

Main automation loop. Processes experiment matrix, tracks state, handles failures.

Usage:
  --resume          Skip completed experiments, retry failed ones (default behavior)
  --filter PATTERN  Only run experiments matching glob pattern on ID
  --dry-run         Print what would run without executing
  --max-retries N   Max retries per failed experiment (default: 2)
  --notify          Send notification on completion/failure (see notify config)

Algorithm:
1. Load config/experiments.yaml
2. Expand matrix into individual experiment entries:
   For each (base × quant × specialization):
     Create TWO experiments: vanilla + finetuned
3. Load logs/experiment_state.jsonl (if --resume / exists)
4. For each experiment NOT in "done" status:

   a. DOWNLOAD step:
      - Check if base model already in models/bases/<base>/
      - If not, download from HF
      - For MoE: also download pre-quantized base
      - Mark step complete in state file

   b. TRAIN step (finetuned experiments only):
      - Check if adapter already in models/lora/<experiment_id>/
      - Load base model with correct method (qlora or bf16_lora)
      - Train on specialization dataset
      - Save adapter to models/lora/<experiment_id>/
      - Mark step complete
      - SHARED CACHE: If another experiment with same (base, specialization)
        already trained, symlink the adapter instead of retraining.
        Training result is quant-independent — only base+data matters.

   c. MERGE step (finetuned dense experiments only):
      - Merge LoRA into base
      - Save to models/merged/<experiment_id>/
      - Mark step complete
      - MoE experiments skip this (adapter-based serving)

   d. QUANTIZE step:
      Branch on quant_level.method (currently only "gptq" is active):

      If method == "gptq":
        - Dense vanilla: AutoGPTQ quantize base model (INT4 or INT8)
        - Dense finetuned: AutoGPTQ quantize merged model
        - MoE vanilla: download pre-quantized base from HF (skip quantize)
        - MoE finetuned: pre-quantized base + adapter (skip quantize)
        - Save to models/gptq/<experiment_id>/

      If method == "gguf":  # DEFERRED — code path exists, not active
        - Convert model to GGUF using llama.cpp's convert script
        - Quantize to target gguf_type (Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0)
        - Save to models/gguf/<experiment_id>/
        - Command: python3 llama.cpp/convert_hf_to_gguf.py <model_path>
                   llama.cpp/llama-quantize <fp16.gguf> <output.gguf> <gguf_type>

      - Run 5-sample smoke test (via appropriate backend)
      - Mark step complete

   e. BENCHMARK step:
      Branch on quant_level.backend (currently only "vllm" is active):

      If backend == "vllm":
        - Start vLLM server with GPTQ model (+ adapter for MoE)
        - Run eval suite via OpenAI-compatible API
        - Measure: tok/s, VRAM, TTFT, quality scores

      If backend == "llamacpp":  # DEFERRED — code path exists, not active
        - Start llama-server with GGUF model
        - Run eval suite via llama.cpp's OpenAI-compatible API
        - Measure: tok/s, RAM usage, TTFT, quality scores
        - Use: llama-server -m <model.gguf> --host 0.0.0.0 --port 8080 -fa

      Both backends expose OpenAI-compatible APIs, so eval scripts are
      backend-agnostic — only the server startup differs. This means
      enabling GGUF later requires zero changes to eval code.

      - Run custom eval on held-out data
      - Save results to logs/eval/<experiment_id>.jsonl
      - Update experiment state with results
      - Mark step complete → status = "done"

5. On ANY step failure:
   - Log error to experiment state
   - Increment retry count
   - If retries < max_retries: mark as "pending" to retry on next pass
   - If retries >= max_retries: mark as "failed", continue to next experiment
   - NEVER let one failure kill the entire run

6. After all experiments complete (or fail):
   - Generate comparison report (see below)
   - Save to logs/comparison_report.md
   - If --notify: send notification

IMPORTANT — Shared training cache logic:
  Training depends on (base_model, specialization) NOT on quant_level.
  A q4 and q8 experiment with the same base+specialization share one training run.
  The runner MUST detect this and reuse the adapter:
    cache_key = f"{base}_{specialization}"
    If models/lora/{cache_key}/ exists → symlink, skip train+merge
    If not → train, save to models/lora/{cache_key}/, then symlink
  This cuts total training runs roughly in half.
"""
```

### Failure Recovery

The runner is designed to be killed and restarted at any point.

**What makes this safe:**
- State file is append-only JSONL — last entry per ID wins
- Each step writes to a unique output path (no partial overwrites)
- Steps are idempotent — rerunning a completed step is a no-op (output exists → skip)
- Training checkpoints saved every N steps — on restart, training resumes from last checkpoint

**Recovery scenarios:**

| Failure | What happens on restart |
|---------|----------------------|
| OOM during training | Retry with lower batch_size (auto-halve, min=1). Log adjustment. |
| Download interrupted | HF hub resumes partial downloads automatically |
| GPTQ calibration crash | Retry. If fails twice: try with more calibration samples (512→1024) |
| Benchmark timeout | Re-run benchmark only (prior steps still marked done) |
| Disk full | Log error, stop. Print which models/merged/ dirs can be cleaned. |
| Power loss / kill -9 | Restart picks up from last completed step per experiment |

**Auto-cleanup between experiments:**
After each experiment completes benchmark, the runner can optionally delete intermediate
artifacts to reclaim disk. Controlled by config:

```yaml
# config/experiments.yaml (add to top level)
cleanup:
  delete_merged_after_quantize: true    # models/merged/<id>/ — large, no longer needed
  delete_lora_after_merge: false        # keep adapters for MoE serving
  keep_last_n_checkpoints: 2           # training checkpoints
```

### Comparison Report

Auto-generated after all experiments complete. Saved to `logs/comparison_report.md`.

```python
"""
generate_report.py

Reads logs/experiment_state.jsonl, produces comparison tables.

Output format (Markdown):

## Comparison Report — Generated 2026-04-07T08:00:00Z

### By Specialization: Code

| Experiment | Base | Quant | Fine-tuned | HumanEval | MBPP | MultiPL-E | tok/s | VRAM (GB) |
|------------|------|-------|------------|-----------|------|-----------|-------|-----------|
| qwen3.5-9b_q8_code_vanilla | Qwen3.5-9B | Q8 | no | 0.65 | 0.61 | 0.58 | 42 | 9.2 |
| qwen3.5-9b_q8_code_finetuned | Qwen3.5-9B | Q8 | yes | 0.81 | 0.76 | 0.73 | 41 | 9.2 |
| qwen3.5-9b_q6_code_vanilla | Qwen3.5-9B | Q6 | no | 0.64 | 0.60 | 0.57 | 48 | 7.1 |
| qwen3.5-9b_q6_code_finetuned | Qwen3.5-9B | Q6 | yes | 0.79 | 0.74 | 0.71 | 47 | 7.1 |
| qwen3.5-27b_q4_code_vanilla | Qwen3.5-27B | Q4 | no | 0.71 | 0.67 | 0.64 | 18 | 15.3 |
| qwen3.5-27b_q4_code_finetuned | Qwen3.5-27B | Q4 | yes | 0.84 | 0.79 | 0.77 | 17 | 15.3 |
| qwen2.5-coder-32b_q4_code_vanilla | Qwen2.5-Coder-32B | Q4 | no | 0.76 | 0.73 | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

### Key Findings: 9B Q8 vs 27B Q4
| Metric | 9B-Q8-FT | 27B-Q4-FT | Winner | Delta |
|--------|----------|-----------|--------|-------|
| HumanEval | 0.81 | 0.84 | 27B | +3.7% |
| tok/s | 41 | 17 | 9B | +141% |
| VRAM | 9.2 | 15.3 | 9B | -40% |
| Quality/VRAM ratio | ... | ... | ... | ... |

### Pareto Frontier
Models on the quality-vs-speed Pareto curve (nothing is both faster AND better):
1. qwen3.5-9b_q6_code_finetuned — best speed (47 tok/s, HumanEval 0.79)
2. qwen3.5-27b_q4_code_finetuned — best quality (17 tok/s, HumanEval 0.84)
3. qwen3.5-35b-a3b_q4_code_finetuned — best quality/compute (MoE, tok/s TBD)

### Pair Compatibility
Models that fit together in 128GB for hot-swap serving:
| Pair | Combined VRAM | Speed (slower model) | Recommended Profile |
|------|--------------|---------------------|-------------------|
| 9B-Q8 code + 9B-Q8 creative | 18.4 GB | 41 tok/s | gamedev, coding |
| 27B-Q4 code + 9B-Q6 research | 22.4 GB | 17 tok/s | coding (quality) |
| 35B-A3B creative + 35B-A3B research | 24 GB | TBD tok/s | planning (MoE) |
"""
```

### Estimated Runtime

Rough estimates per experiment step on Strix Halo iGPU (compute-limited, expect slower per-step than discrete GPU):

| Step | 9B model | 27B model | 35B-A3B MoE | Notes |
|------|----------|-----------|-------------|-------|
| Download | 15-30 min | 30-60 min | 30-60 min | Depends on bandwidth |
| Train (50K examples, 3 epochs) | 3-5 hours | 8-12 hours | 6-10 hours | QLoRA/bf16 LoRA |
| Merge | 5-10 min | 15-30 min | N/A (adapter) | RAM-bound |
| Quantize (GPTQ) | 30-60 min | 1-2 hours | N/A (pre-quant) | Calibration-bound |
| Benchmark | 30-60 min | 1-2 hours | 30-60 min | Depends on eval suite |
| **Total per experiment** | **~5-7 hours** | **~11-16 hours** | **~7-12 hours** | |

With shared training cache (quant levels share one train run):
- 9B × 2 quants × 4 specs × 2 (vanilla+FT) = 16 experiments, but only 4 unique training runs
- 27B × 2 quants × 4 specs × 2 = 16 experiments, but only 4 unique training runs

**Total estimated wall time:** ~120-180 hours (~5-7 days) for the full matrix.
Plan to let it run over a week. The runner handles restarts.

---

## Execution Order

### Step 0: End-to-End Pipeline Validation (DO THIS FIRST)

Before launching the full experiment matrix, validate the entire pipeline works on Strix Halo with the smallest possible run:

```bash
uv run llm-run --filter "qwen3.5-9b_gptq-int4_chat*" --resume
```

This runs exactly 2 experiments (vanilla + finetuned) for the smallest model × cheapest quant × smallest dataset. Expected wall time: ~6-8 hours.

**What this validates:**
- [ ] ROCm + bf16 LoRA training works on Strix Halo iGPU
- [ ] GPTQ quantization (AutoGPTQ) produces working models
- [ ] vLLM serves the quantized model correctly
- [ ] Eval harness runs and produces scores
- [ ] State file tracks progress correctly (kill and restart mid-run to test resumability)
- [ ] Report generation works

**If this fails:** Fix the issue before scaling up. Common failure points:
- bf16 not supported → fall back to fp16 and update all configs
- BitsAndBytes not available on ROCm → QLoRA path broken, use bf16 LoRA for everything
- vLLM GPTQ loading fails → check vLLM version, ROCm compatibility

### Step 1+: Full Matrix

After validation passes, prioritize smaller models first:

1. **All Qwen3.5-9B experiments** — fastest training, broadest comparison (all 4 specializations × 2 quants)
2. **GLM-4 experiments** — creative + chat comparison against Qwen3.5-9B
3. **Qwen2.5-Coder-14B** — code specialization baseline
4. **Qwen3.5-27B + Qwen3.5-35B-A3B** — larger models, slower training
5. **Qwen2.5-Coder-32B** — largest dense model, run last

```bash
# After validation, run everything:
uv run llm-run --resume
```

---

## Directory Structure

The agent expects this layout. Create it before running any phase.

```
~/git/llm-training/
├── pyproject.toml             ← uv project (deps, entry points, ROCm torch index)
├── plan.md                    ← THIS FILE
├── src/llm_training/          ← Python package
│   ├── run_experiments.py     ← MAIN ENTRY POINT: fire-and-forget experiment runner
│   ├── common.py              ← shared utilities, config loading, state management
│   ├── download.py            ← dataset download + initial filtering
│   ├── curate.py              ← dedup, quality filter, format conversion
│   ├── train.py               ← QLoRA / bf16 LoRA training (auto-selects based on architecture)
│   ├── merge.py               ← LoRA merge into base (dense only)
│   ├── quantize.py            ← GPTQ quantization (AutoGPTQ or GPTQModel w/ FailSafe)
│   ├── eval.py                ← benchmark runner (lm-eval-harness + custom evals)
│   └── generate_report.py     ← comparison report generator (called by runner at end)
├── scripts/
│   └── serve.sh               ← vLLM launch with session presets
├── config/
│   ├── models.yaml            ← model registry (parsed by scripts)
│   ├── experiments.yaml       ← experiment matrix
│   ├── train/
│   │   ├── base_qlora.yaml    ← QLoRA config (27B+ dense)
│   │   ├── base_moe.yaml      ← bf16 LoRA config (MoE)
│   │   ├── base_dense_lora.yaml ← bf16 LoRA config (<=14B dense)
│   │   ├── code.yaml          ← per-specialization overrides
│   │   ├── creative.yaml
│   │   ├── research.yaml
│   │   └── chat.yaml
│   └── serve/
│       └── instances.yaml     ← vLLM instance configs + session presets
├── datasets/
│   ├── raw/                   ← downloaded datasets (HF cache or exports)
│   ├── processed/             ← cleaned, deduped, formatted
│   │   ├── code/
│   │   ├── creative/
│   │   ├── research/
│   │   └── chat/
│   ├── calibration/           ← GPTQ calibration splits (256 samples each)
│   └── eval/                  ← held-out eval splits (500 samples each)
├── models/
│   ├── bases/                 ← downloaded base models (shared across experiments)
│   ├── lora/                  ← LoRA adapters, keyed by {base}_{specialization} (shared cache)
│   ├── merged/                ← merged full-precision models (temp, auto-cleaned)
│   ├── gptq/                  ← GPTQ quantized models (vLLM backend), keyed by experiment ID
│   └── gguf/                  ← (deferred) GGUF quantized models for llama.cpp backend
└── logs/
    ├── experiment_state.jsonl ← experiment runner state (resumable, append-only)
    ├── comparison_report.md   ← auto-generated comparison tables
    ├── training/              ← training logs, loss curves
    ├── eval/                  ← eval results per experiment ID
    └── runs.jsonl             ← append-only run log (legacy, still populated)
```

---

## Model Registry (`config/models.yaml`)

Central source of truth. Scripts read this to know what to download, train, quantize, and serve.

```yaml
# config/models.yaml
# Add/remove models here. Scripts iterate over enabled entries.

models:
  code:
    enabled: true
    architecture: dense       # dense | moe
    base: "Qwen/Qwen2.5-Coder-32B-Instruct"
    fallback_base: "Qwen/Qwen2.5-Coder-14B-Instruct"
    use_fallback: false  # flip to true to switch to 14B
    # upgrade_base: "Qwen/Qwen3-Coder-*"  # uncomment when available on HF
    gptq_bits: 4
    gptq_group_size: 128
    train_method: qlora       # qlora (dense) | bf16_lora (moe)
    train_config: "config/train/code.yaml"
    datasets:
      - name: "glaive-code-assistant-v3"
        source: "huggingface"
        hf_id: "glaiveai/glaive-code-assistant-v3"
        sample_size: null  # null = use all
        format: "sharegpt"
        filters:
          languages: ["python", "bash", "typescript", "cpp", "csharp", "gdscript"]
      - name: "CodeFeedback-Filtered-Instruction"
        source: "huggingface"
        hf_id: "m-a-p/CodeFeedback-Filtered-Instruction"
        sample_size: null
        format: "sharegpt"
      - name: "CommitPackFT"
        source: "huggingface"
        hf_id: "bigcode/commitpackft"
        sample_size: 50000
        format: "alpaca"
      - name: "the-stack-v2-dedup"
        source: "huggingface"
        hf_id: "bigcode/the-stack-v2-dedup"
        sample_size: 100000
        format: "completion"
        filters:
          languages: ["python", "bash", "typescript"]
          min_docstring_ratio: 0.1
          skip_generated: true
      - name: "game-engine-docs"
        source: "local"
        path: "datasets/raw/game-engine-docs/"
        format: "alpaca"
        custom: true
      - name: "shader-pipeline"
        source: "local"
        path: "datasets/raw/shader-pipeline/"
        format: "alpaca"
        custom: true
      - name: "game-dev-github"
        source: "local"
        path: "datasets/raw/game-dev-github/"
        format: "sharegpt"
        custom: true

  creative:
    enabled: true
    architecture: dense       # switch to "moe" if using 35B-A3B
    base: "THUDM/glm-4-9b-chat"
    # alt_base_moe: "Qwen/Qwen3.5-35B-A3B"
    # alt_base_dense: "Qwen/Qwen3.5-9B"
    gptq_bits: 4
    gptq_group_size: 128
    train_method: qlora       # switch to bf16_lora if architecture=moe
    train_config: "config/train/creative.yaml"
    datasets:
      - name: "lmsys-chat-1m-creative"
        source: "huggingface"
        hf_id: "lmsys/lmsys-chat-1m"
        sample_size: 30000
        format: "sharegpt"
        filters:
          categories: ["creative", "writing", "roleplay", "worldbuilding"]
      - name: "cosmopedia-v2"
        source: "huggingface"
        hf_id: "HuggingFaceTB/cosmopedia-v2"
        sample_size: 50000
        format: "alpaca"
      - name: "OpenOrca-planning"
        source: "huggingface"
        hf_id: "Open-Orca/OpenOrca"
        sample_size: 30000
        format: "sharegpt"
        filters:
          categories: ["planning", "analytical", "structured"]
      - name: "UltraFeedback"
        source: "huggingface"
        hf_id: "openbmb/UltraFeedback"
        sample_size: null
        format: "preference"
      - name: "game-lore"
        source: "local"
        path: "datasets/raw/game-lore/"
        format: "sharegpt"
        custom: true
      - name: "project-plans"
        source: "local"
        path: "datasets/raw/project-plans/"
        format: "alpaca"
        custom: true
      - name: "store-marketing"
        source: "local"
        path: "datasets/raw/store-marketing/"
        format: "alpaca"
        custom: true

  research:
    enabled: false  # enable when vanilla Qwen3.5 isn't enough
    architecture: moe
    base: "Qwen/Qwen3.5-35B-A3B"
    # alt_base_larger: "Qwen/Qwen3.5-122B-A10B"  # ~68GB GPTQ — solo on Strix Halo, no pairs
    # alt_base_dense: "Qwen/Qwen2.5-14B-Instruct"
    gptq_bits: 4
    gptq_group_size: 128
    train_method: bf16_lora
    moe_serve_mode: adapter   # adapter | merged_failsafe
    moe_base_quantized: "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"  # pre-quantized base from HF
    train_config: "config/train/research.yaml"
    datasets:
      - name: "FLAN-v2-reasoning"
        source: "huggingface"
        hf_id: "Muennighoff/flan"
        sample_size: 50000
        format: "alpaca"
        filters:
          categories: ["reasoning", "chain_of_thought"]
      - name: "SlimOrca-Dedup"
        source: "huggingface"
        hf_id: "Open-Orca/SlimOrca-Dedup"
        sample_size: 50000
        format: "sharegpt"
      - name: "OpenOrca-full"
        source: "huggingface"
        hf_id: "Open-Orca/OpenOrca"
        sample_size: 50000
        format: "sharegpt"
      - name: "UltraFeedback"
        source: "huggingface"
        hf_id: "openbmb/UltraFeedback"
        sample_size: null
        format: "preference"

  chat:
    enabled: true
    architecture: dense       # switch to "moe" if using 35B-A3B
    base: "THUDM/glm-4-9b-chat"
    # alt_base_moe: "Qwen/Qwen3.5-35B-A3B"    # fits 24GB GPTQ-4bit (~10-12GB)
    # alt_base_dense: "Qwen/Qwen3.5-9B"        # direct GLM-4 upgrade
    gptq_bits: 4
    gptq_group_size: 128
    train_method: qlora       # switch to bf16_lora if architecture=moe
    train_config: "config/train/chat.yaml"
    serve_target: "strix-halo"  # own vLLM instance
    datasets:
      - name: "lmsys-chat-1m"
        source: "huggingface"
        hf_id: "lmsys/lmsys-chat-1m"
        sample_size: 100000
        format: "sharegpt"
      - name: "ShareGPT-Vicuna-unfiltered"
        source: "huggingface"
        hf_id: "anon8231489123/ShareGPT_Vicuna_unfiltered"
        sample_size: null
        format: "sharegpt"
      - name: "Capybara"
        source: "huggingface"
        hf_id: "LDJnr/Capybara"
        sample_size: null
        format: "sharegpt"
      - name: "personal-style"
        source: "local"
        path: "datasets/raw/personal-style/"
        format: "sharegpt"
        custom: true
```

---

## Claude Code Execution Guide

Instructions for an agent (Claude Code or similar) to execute this plan phase by phase.
Each phase is independent and idempotent — safe to re-run.

### Phase 0: Environment Setup

**Goal:** Create directory structure, install dependencies, validate hardware.

```bash
# 1. Install dependencies via uv
cd ~/git/llm-training
uv sync

# 2. Validate ROCm + GPU
uv run python -c "
import torch
print(f'ROCm available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} — {torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB')
print(f'bf16 support: {torch.cuda.is_bf16_supported()}')
"

# 3. Validate HuggingFace auth (needed for gated datasets)
uv run huggingface-cli whoami
# If not logged in: uv run huggingface-cli login --token <token>

# 4. DEFERRED: Build llama.cpp for GGUF quantization + inference
# Uncomment when enabling GGUF quant levels in config/experiments.yaml
# git clone https://github.com/ggml-org/llama.cpp.git
# cd llama.cpp && cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS="gfx1151" && cmake --build build -j
# gfx1151 = Strix Halo iGPU. Verify with: rocminfo | grep gfx
```

**Verify before proceeding:**
- [ ] `torch.cuda.is_available()` returns True
- [ ] Strix Halo iGPU visible (128GB unified)
- [ ] bf16 support confirmed (or note to use fp16 fallback)
- [ ] HuggingFace authenticated
- [ ] Directory tree exists
- [ ] (DEFERRED) llama.cpp built with HIP support — only needed if enabling GGUF experiments

---

### Phase 1: Download Base Models + Datasets

**Goal:** Pull all base models and raw datasets for enabled models.

```bash
# Read config/models.yaml, iterate enabled models
# For each enabled model:

# 1a. Download base model
python3 -c "
from huggingface_hub import snapshot_download
import yaml

with open('config/models.yaml') as f:
    cfg = yaml.safe_load(f)

for model_id, model_cfg in cfg['models'].items():
    if not model_cfg.get('enabled', False):
        continue
    base = model_cfg['base']
    if model_cfg.get('use_fallback'):
        base = model_cfg.get('fallback_base', base)
    print(f'Downloading base: {base}')
    snapshot_download(base, local_dir=f'models/bases/{model_id}')
"

# 1b. Download HuggingFace datasets
python3 -c "
from datasets import load_dataset
import yaml, json

with open('config/models.yaml') as f:
    cfg = yaml.safe_load(f)

for model_id, model_cfg in cfg['models'].items():
    if not model_cfg.get('enabled', False):
        continue
    for ds in model_cfg['datasets']:
        if ds['source'] != 'huggingface':
            continue
        print(f'[{model_id}] Downloading {ds[\"name\"]} from {ds[\"hf_id\"]}...')
        dataset = load_dataset(ds['hf_id'], split='train')
        if ds.get('sample_size'):
            dataset = dataset.shuffle(seed=42).select(range(min(ds['sample_size'], len(dataset))))
        out_path = f'datasets/raw/{ds[\"name\"]}'
        dataset.save_to_disk(out_path)
        print(f'  → saved {len(dataset)} examples to {out_path}')
"
```

**Verify:**
- [ ] Each `models/bases/<model_id>/` has model files (config.json, safetensors, tokenizer)
- [ ] Each `datasets/raw/<dataset_name>/` has data
- [ ] Log any download failures — some gated datasets may need HF approval

---

### Phase 2: Dataset Curation

**Goal:** Clean, filter, dedup, format, and split each dataset.

For each enabled model, process its datasets into `datasets/processed/<model_id>/`:

```bash
uv run llm-curate --model <model_id>
```

The curate script should (implement or use as pseudocode):

```python
"""
curate.py --model <model_id>

For each dataset listed under the model in config/models.yaml:
1. Load from datasets/raw/<name>/
2. Apply filters (language, category, etc. from config)
3. Quality filter:
   - Remove examples < 50 tokens
   - Remove examples > max_seq_length tokens (from train config)
   - Remove obviously broken/nonsensical (heuristic: >50% non-alphanumeric)
4. Dedup:
   - Exact dedup on normalized text
   - Near-dedup via MinHash (datasketch, threshold=0.85)
5. Format conversion:
   - Target: ShareGPT multi-turn OR Alpaca instruction (per dataset config)
   - Output schema for ShareGPT:
     {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
   - Output schema for Alpaca:
     {"instruction": "...", "input": "...", "output": "..."}
6. Merge all datasets for this model into one JSONL
7. Shuffle with fixed seed
8. Split off:
   - Last 256 examples → datasets/calibration/<model_id>.jsonl
   - Last 500 examples (before calibration) → datasets/eval/<model_id>.jsonl
   - Rest → datasets/processed/<model_id>/train.jsonl
9. Print stats: total examples, per-dataset contribution, token length distribution
"""
```

**Verify per model:**
- [ ] `datasets/processed/<model_id>/train.jsonl` exists and has expected row count
- [ ] `datasets/calibration/<model_id>.jsonl` has 256 rows
- [ ] `datasets/eval/<model_id>.jsonl` has 500 rows
- [ ] Spot-check 10 random examples from train.jsonl — look for quality issues
- [ ] No near-duplicates in a random 100-sample check

---

### Phase 3: Baseline Eval

**Goal:** Measure base model performance BEFORE fine-tuning.

```bash
uv run llm-eval --model <model_id> --stage baseline
```

```python
"""
eval.py --model <model_id> --stage <baseline|finetuned>

1. Load model:
   - If stage=baseline: load from models/bases/<model_id>/
   - If stage=finetuned: load from models/gptq/<model_id>/
2. Run benchmarks from Eval Strategy table:
   - code: HumanEval, MBPP (via lm-eval-harness)
   - creative: MT-Bench (via fastchat)
   - research: MMLU subset, ARC-Challenge (via lm-eval-harness)
   - chat: MT-Bench
3. Run custom eval on datasets/eval/<model_id>.jsonl:
   - Generate responses for each prompt
   - Save to logs/eval/<model_id>_<stage>.jsonl
4. Append summary to logs/runs.jsonl:
   {"timestamp": "...", "model": "<model_id>", "stage": "<stage>",
    "benchmarks": {"humaneval": 0.xx, ...}, "custom_eval_samples": 500}
"""
```

**Verify:**
- [ ] `logs/eval/<model_id>_baseline.jsonl` exists
- [ ] Benchmark scores logged to `logs/runs.jsonl`
- [ ] Scores match roughly expected performance for the base model

---

### Phase 4: Training (QLoRA or bf16 LoRA)

**Goal:** Fine-tune each enabled model. Script auto-selects QLoRA (dense) or bf16 LoRA (MoE) based on `train_method` in config.

```bash
uv run llm-train --model <model_id>
```

```python
"""
train.py --model <model_id>

1. Load config:
   - Read model entry from config/models.yaml
   - Determine train_method: "qlora" or "bf16_lora"
   - Load base template: config/train/base_qlora.yaml or config/train/base_moe.yaml
   - Merge per-model overrides from config/train/<model_id>.yaml
2. Load base model:
   - If qlora: load with 4-bit NF4 quantization (BitsAndBytes) — for larger dense models (27B+)
   - If bf16_lora: load in bf16 full precision (NO 4-bit quantization) — for MoE and small dense (≤14B)
3. Apply LoRA adapters to target modules (from config)
   - For MoE: ensure router/gate weights are in target_modules AND modules_to_save
4. Load training data from datasets/processed/<model_id>/train.jsonl
5. Train using HuggingFace Trainer or Axolotl:
   - For MoE: recommend Axolotl (has ScatterMoE LoRA + quantize_moe_experts support)
   - For dense: either Axolotl or plain Trainer
   - All params from merged config
   - Log to logs/training/<model_id>/
   - Save checkpoints to models/lora/<model_id>/
6. On completion, append to logs/runs.jsonl:
   {"timestamp": "...", "model": "<model_id>", "phase": "train",
    "train_method": "qlora|bf16_lora", "architecture": "dense|moe",
    "epochs": N, "final_loss": X.XX, "duration_hours": X.X}
"""
```

**Training order (from Execution Order section):**
1. `chat` — validate the pipeline end-to-end
2. `code` (with `use_fallback: true` initially for 14B)
3. `creative`
4. `research` (only if enabled)

**Verify per model:**
- [ ] `models/lora/<model_id>/` has adapter_config.json + adapter_model.safetensors
- [ ] Training loss decreased over epochs (check logs/training/<model_id>/)
- [ ] No OOM errors in training log
- [ ] Final loss is reasonable (< 1.5 for instruction tuning, varies by model)

---

### Phase 5: Merge + Quantize (or Adapter Prep)

**Goal:** Prepare model for inference. Path depends on architecture.

**Dense models:** Merge LoRA → GPTQ quantize (same as before).
**MoE models:** Either (a) keep adapter separate + use pre-quantized base, or (b) merge + requantize with FailSafe.

```bash
uv run llm-quantize --model <model_id>
```

```python
"""
post_train.py --model <model_id>

1. Read model config from config/models.yaml
2. Branch on architecture + moe_serve_mode:

--- PATH A: Dense (or MoE with merged_failsafe) ---
3a. Merge LoRA into base:
    model = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    Save to models/merged/<model_id>/
4a. Quantize:
    - Dense: AutoGPTQ, standard calibration (256 samples)
    - MoE (merged_failsafe): GPTQModel 5.0+ with fail_safe=True,
      use 512 calibration samples for better expert coverage
    Save to models/gptq/<model_id>/
5a. Verify: load quantized model, run 5 sample prompts
6a. Log to runs.jsonl

--- PATH B: MoE with adapter serving ---
3b. Download pre-quantized base if not already present:
    e.g., Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 → models/gptq/<model_id>-base/
4b. Copy LoRA adapter to models/lora/<model_id>/ (already there from training)
5b. Create adapter config at models/adapters/<model_id>/config.json:
    {
      "base_model": "models/gptq/<model_id>-base/",
      "adapter": "models/lora/<model_id>/",
      "merge_at_load": false
    }
6b. Verify: load base + adapter in vLLM, run 5 sample prompts
7b. Log to runs.jsonl
"""
```

**Verify per model:**

Dense:
- [ ] `models/gptq/<model_id>/` has quantized model files
- [ ] Model size matches expected GPTQ size from model table
- [ ] 5-sample smoke test passes (coherent outputs, not garbage)
- [ ] Can delete `models/merged/<model_id>/` to reclaim disk space

MoE (adapter):
- [ ] `models/gptq/<model_id>-base/` has pre-quantized base
- [ ] `models/lora/<model_id>/` has adapter weights
- [ ] `models/adapters/<model_id>/config.json` points to correct paths
- [ ] vLLM loads base + adapter successfully
- [ ] 5-sample smoke test passes

MoE (merged_failsafe):
- [ ] `models/gptq/<model_id>/` has quantized model files
- [ ] GPTQModel FailSafe log shows which experts got weight-only quant
- [ ] 5-sample smoke test passes — pay extra attention to edge-case prompts
- [ ] Compare 10 outputs vs adapter-serving path for quality sanity check

---

### Phase 6: Post-Training Eval

**Goal:** Measure improvement over baseline.

```bash
uv run llm-eval --model <model_id> --stage finetuned
```

**Verify:**
- [ ] `logs/eval/<model_id>_finetuned.jsonl` exists
- [ ] Compare scores to baseline in `logs/runs.jsonl`
- [ ] Improvement on target domains (or at least no regression)
- [ ] If regression detected: flag for review, do NOT deploy

---

### Phase 7: Deploy to vLLM / ut3g

**Goal:** Make models available for inference.

```bash
# Start a session (launches vLLM instances for selected models):
bash scripts/serve.sh --session coding  # or gamedev, planning, research-heavy

# Start individual model instances:
bash scripts/serve.sh --model code
bash scripts/serve.sh --model chat

# Stop all instances:
bash scripts/serve.sh --stop-all
```

```bash
"""
serve.sh --session <session_name> | --model <model_name> | --stop-all

1. Read config/serve/instances.yaml
2. If --session: resolve session preset to list of model names
   If --model: use single model name
3. For each model to start:
   - Check if port is already in use (skip if instance already running)
   - Start vLLM instance:
     vllm serve <model_path> \
       --model-name <model_name> \
       --port <port> \
       --max-model-len <from config> \
       --gpu-memory-utilization <from config> \
       --quantization gptq \
       --dtype float16 \
       [--enable-lora --lora-modules adapter=<adapter_path>]  # if adapter-based
       [--enable-expert-parallel]                               # if MoE
4. Verify: curl health endpoint for each instance
"""
```

**Verify:**
- [ ] vLLM health endpoint responds on each model's port
- [ ] Can generate completions from each loaded model
- [ ] Inference speed acceptable (>15 tok/s for 9B, >8 tok/s for 27B+ on iGPU)

---

### Run Log Schema (`logs/runs.jsonl`)

Every script appends to this file. Provides full audit trail.

```jsonl
{"timestamp": "2026-04-05T10:00:00Z", "model": "chat", "phase": "download", "base": "THUDM/glm-4-9b-chat", "datasets_downloaded": 4}
{"timestamp": "2026-04-05T10:30:00Z", "model": "chat", "phase": "curate", "train_examples": 45000, "calibration": 256, "eval": 500}
{"timestamp": "2026-04-05T11:00:00Z", "model": "chat", "phase": "eval", "stage": "baseline", "mt_bench": 7.2, "alpaca_eval": 0.68}
{"timestamp": "2026-04-05T14:00:00Z", "model": "chat", "phase": "train", "epochs": 3, "final_loss": 1.12, "duration_hours": 2.5}
{"timestamp": "2026-04-05T14:30:00Z", "model": "chat", "phase": "merge", "output": "models/merged/chat/"}
{"timestamp": "2026-04-05T15:00:00Z", "model": "chat", "phase": "quantize", "bits": 4, "size_gb": 5.1, "verification": "pass"}
{"timestamp": "2026-04-05T15:30:00Z", "model": "chat", "phase": "eval", "stage": "finetuned", "mt_bench": 7.8, "alpaca_eval": 0.74}
{"timestamp": "2026-04-05T15:45:00Z", "model": "chat", "phase": "deploy", "target": "strix-halo-vllm", "port": 8103, "status": "live"}
```

---

### Adding a New Model

1. Add entry to `config/models.yaml` under `models:` (copy an existing one as template)
2. Create `config/train/<new_id>.yaml` with any overrides
3. Place custom datasets in `datasets/raw/<dataset_name>/`
4. Run phases 1-7 for the new model ID
5. Add a vLLM serving profile that includes it, or add it to an existing profile

### Removing a Model

1. Set `enabled: false` in `config/models.yaml` (keeps config for later)
2. Or delete the entry entirely
3. Optionally clean up: `rm -rf models/{lora,merged,gptq}/<model_id>/ datasets/processed/<model_id>/`

### Swapping a Dataset

1. Update the dataset entry in `config/models.yaml`
2. Re-run Phase 1 (download) + Phase 2 (curate) for that model
3. Re-run Phase 4 (train) onward — the pipeline is idempotent

---

## Qwen3.5 Model Reference

Quick reference for all Qwen3.5 variants relevant to this plan.

| Model | Architecture | Total Params | Active Params | GPTQ-4bit Size | Fits Strix Halo (pair)? | Trainable on 7900XTX? |
|-------|-------------|-------------|---------------|----------------|------------------------|----------------------|
| Qwen3.5-9B | Dense | 9B | 9B | ~5GB | Yes (easily) | QLoRA: yes |
| Qwen3.5-27B | Dense | 27B | 27B | ~15GB | Yes | QLoRA: tight, batch=1 |
| Qwen3.5-35B-A3B | MoE | 35B | 3B | ~10-12GB | Yes | bf16 LoRA: yes (~18GB) |
| Qwen3.5-122B-A10B | MoE | 122B | 10B | ~68GB | Solo only | bf16 LoRA: no (too large) |
| Qwen3.5-397B-A17B | MoE | 397B | 17B | ~220GB | No | No |

**Key advantage of MoE on Strix Halo:** The iGPU is memory-bandwidth-bound, not compute-bound. MoE models only activate a fraction of params per token, so effective compute per token is much lower than the total param count suggests. A 35B-A3B MoE runs at similar speed to a 3-5B dense model while delivering much better quality.

**Qwen3.5 vs GLM-4.7-Flash:** Qwen3.5-9B rivals Qwen2.5-72B on benchmarks, has native thinking mode (can be toggled), and 262K native context. It's a strict upgrade over GLM-4.7-Flash for most tasks. Consider switching creative + chat bases to Qwen3.5-9B or Qwen3.5-35B-A3B.

**No Qwen3.5-Coder yet:** As of April 2026, Qwen3-Coder exists but not Qwen3.5-Coder. The code model should stay on Qwen2.5-Coder-32B until a Qwen3.5-Coder drops. Monitor: https://huggingface.co/Qwen

---

## Open Questions

### Resolved
- [x] ~~vLLM quantization support on Strix Halo~~ → GPTQ INT4/INT8 only. No Q5/Q6. GGUF supported but ~3x slower. Use llama.cpp for GGUF. FP8 and BNB not available on ROCm/RDNA3.

### Still Open
- [ ] ROCm bf16 support on Strix Halo iGPU — test before committing to bf16 training (required for MoE path and small dense models)
- [ ] vLLM LoRA adapter serving + GPTQ base — verify `--enable-lora` works with MoE GPTQ models on ROCm
- [ ] End-to-end pipeline validation — run smallest model (Qwen3.5-9B, chat spec) through full pipeline before launching the full matrix
- [ ] (DEFERRED) llama.cpp Strix Halo gfx target — confirm gfx1151 is correct; ROCm 7.x has reported 3x regression vs 6.4.4 for some targets
- [ ] GLM-4.7-Flash fine-tuning compatibility with standard tooling (Axolotl, etc.) — THUDM models sometimes need custom chat templates
- [ ] Qwen3.5 thinking mode: does it interfere with fine-tuning? May need to disable during training or curate datasets with/without thinking tokens
- [ ] GPTQModel 5.0+ FailSafe: test quality on Qwen3.5-35B-A3B — compare merged+failsafe vs adapter-serving on 50 prompts
- [ ] GPTQ calibration: same-domain vs. general (C4) — benchmark both
- [ ] Game dev dataset size: is 20-30K enough or do we need more coverage?
- [ ] Qwen3.5-122B-A10B inference speed on Strix Halo — worth it as solo research model? (~68GB GPTQ, leaves ~50GB for KV cache)
- [ ] Axolotl ScatterMoE LoRA on ROCm — Triton kernels may need ROCm-specific builds. If Axolotl MoE support doesn't work on ROCm, fall back to HuggingFace Trainer with standard LoRA (loses expert-level targeting but still functional)
- [ ] (DEFERRED) llama.cpp vs vLLM throughput on Strix Halo for same model — measure when enabling GGUF experiments
