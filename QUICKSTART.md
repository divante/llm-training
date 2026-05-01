## Quick Start

All scripts live in `~/git/llm-training/scripts/generate/`. Run from the repo root.

```bash
cd ~/git/llm-training
```

### Prerequisites

```bash
# 1. Install deps
uv sync

# 2. Local LLM server (primary — leave running in tmux)
#    Available at http://localhost:11437/v1
#    Default model: qwen3.5:35b-a3b

# 3. Secondary server (optional, for parallel workloads)
export LLM_BASE_URL=http://192.168.50.233:11434/v1
export LLM_MODEL=qwen2.5-coder:14b
export LLM_API_KEY=unused
# NOTE: Do NOT use qwen3.5:* on ollama — thinking mode burns all tokens, returns empty content.
#       Use qwen2.5-coder:14b or qwen2.5-coder:7b instead.
```

---

## Dataset Directory Layout

Raw generation outputs live in `datasets/raw/<name>/` alongside source chunks and
intermediate files. **Curate-ready data** (flattened, one record per line, in alpaca
or sharegpt format) lives in `datasets/raw/<name>/training/`.

`curate.py` reads from the `training/` subdirectories (configured in `config/models.yaml`).
This prevents it from accidentally ingesting source chunks or raw response wrappers.

After generating raw data, run the flatten step to produce curate-ready files:

```bash
# Already done for current datasets — only needed after new generation runs
# See "Flatten Raw Data" section below
```

---

## Coding Agent Datasets

### Step 1: Game Engine Docs (~8-12h generation)

```bash
# Generate QA pairs from indexed engine docs
uv run scripts/generate/generate_engine_docs.py

# Single source:
uv run scripts/generate/generate_engine_docs.py --source unreal
uv run scripts/generate/generate_engine_docs.py --source godot-api
uv run scripts/generate/generate_engine_docs.py --source godot-docs

# Fix truncated/unparseable entries:
uv run scripts/generate/generate_engine_docs.py --source unreal --redo-failures
```

**Status:** Done. 11,734 QA pairs extracted (Godot API: 1,621 + Godot docs: 1,490 + Unreal: 8,760).
Note: ~38% of Unreal responses were truncated (max_tokens cutoff on long C++ outputs).
Run `--redo-failures` to regenerate those.

### Step 2: Game Dev GitHub (~1-2 days, 3 phases)

```bash
# Phase 1: Automated screening (no LLM needed, just git clone + heuristics)
uv run scripts/generate/screen_repo.py --engine godot --from-file datasets/raw/game-dev-github/candidates_godot.txt
uv run scripts/generate/screen_repo.py --engine unreal --from-file datasets/raw/game-dev-github/candidates_unreal.txt

# Phase 2: Model-based quality review (auto-reads Phase 1 PASS+REVIEW repos)
# Use the secondary server to keep primary free for other generation
LLM_BASE_URL=http://192.168.50.233:11434/v1 LLM_MODEL=qwen2.5-coder:14b LLM_API_KEY=unused \
  uv run scripts/generate/review_repo.py --engine unreal
LLM_BASE_URL=http://192.168.50.233:11434/v1 LLM_MODEL=qwen2.5-coder:14b LLM_API_KEY=unused \
  uv run scripts/generate/review_repo.py --engine godot

# Check results: datasets/raw/game-dev-github/quality_reviews/*.json
# approved_repos.yaml is populated automatically from repos scoring >=3.5

# Phase 3: Extract from approved repos -> multi-turn conversations
LLM_BASE_URL=http://192.168.50.117:11438/v1 LLM_API_KEY=unused \
  uv run scripts/generate/extract_github.py --target 15000  --worker-id 0 --num-workers 2
```

**Status:** 1,256 conversations extracted (956 Unreal, 300 Godot, 90 repos).
Target was 10-20K. Extraction needs more runs to reach target.

### Step 3: Shader Pipeline (~3-4h generation)

```bash
# 3a. Generate source chunks (run all three — they're independent)
uv run scripts/generate/export_glyph_shaders.py          # Unreal + Godot from Glyph DB -> ~6K chunks
uv run scripts/generate/import_hf_shaders.py --limit 10000  # HuggingFace shader_dataset -> 10K chunks
uv run scripts/generate/scrape_bookofshaders.py           # Book of Shaders repo -> ~376 chunks

# 3b. Generate Q&A pairs from chunks
uv run scripts/generate/generate_shaders.py --target 7500

# Parallel (two workers):
uv run scripts/generate/generate_shaders.py --target 7500 --worker-id 0 --num-workers 2
LLM_BASE_URL=http://192.168.50.117:11438/v1 uv run scripts/generate/generate_shaders.py --target 7500 --worker-id 1 --num-workers 2
```

**Status:** Done. 2,013 source chunks processed across 4 parallel workers (anthropic: 518,
gemini: 500, local_2: 500, local_3: 500). Deduped to 2,012 records -> 10,036 QA pairs.

### Flatten Raw Data

After generation, flatten raw responses into curate-ready format. Raw responses have nested
structures (multiple QA pairs per record, metadata wrappers) that `curate.py` can't read
directly.

```bash
# One-time flatten — produces files in datasets/raw/<name>/training/
# These are what curate.py reads (paths configured in config/models.yaml)

# Already done for current data. Re-run after new generation runs:
python3 scripts/flatten_for_curation.py  # TODO: make this a proper script if needed

# Manual verification:
wc -l datasets/raw/shader-pipeline/training/shader_qa.jsonl       # 10,036 alpaca records
wc -l datasets/raw/game-dev-github/training/github_conversations.jsonl  # 1,256 sharegpt records
wc -l datasets/raw/game-engine-docs/training/engine_docs_qa.jsonl # 11,734 alpaca records
```

---
 
## Creative Agent Datasets

### Step 4: Project Plans (~10-14h generation)

```bash
# Fully synthetic — no scraping needed
uv run scripts/generate/generate_plans.py --target 7500

# Or on secondary server:
LLM_BASE_URL=http://192.168.50.233:11434/v1 LLM_MODEL=qwen2.5-coder:14b LLM_API_KEY=unused \
  nohup uv run scripts/generate/generate_plans.py --target 7500 &
```

### Step 5: Store Marketing (~3-4h generation)

```bash
uv run scripts/generate/generate_marketing.py --target 5000
```

**Status:** In progress, paused at 670/1,666 (rate limited by Steam API). Will complete
in background.

### Step 6: Game Lore (~16-24h generation)

Requires wiki chunks. Use Glyph to crawl wikis first, or scrape manually.

```bash
# 1. Review/edit genre config
$EDITOR datasets/raw/game-lore/genres.yaml

# 2. Place wiki chunks at datasets/raw/game-lore/*_chunks.jsonl

# 3. Generate creative writing pairs
uv run scripts/generate/generate_lore.py --target 12500
```

---

## Chat Agent Datasets

### Step 7: Personal Style (manual, ~30 min of your time)

Requires conversation exports. Skip if not ready — the pipeline handles missing data.

```bash
# Export from Claude.ai: Settings -> Privacy -> Export Data
# Place at: datasets/raw/personal-style/claude_export/conversations.json
#
# Export from Gemini: https://takeout.google.com -> select "Gemini Apps"
# Place at: datasets/raw/personal-style/gemini_export/MyActivity.json

# Parse exports into ShareGPT format
uv run scripts/parse_conversation_exports.py parse \
  --source claude \
  --input datasets/raw/personal-style/claude_export/conversations.json \
  --output datasets/raw/personal-style/claude.jsonl

uv run scripts/parse_conversation_exports.py parse \
  --source gemini \
  --input datasets/raw/personal-style/gemini_export/ \
  --output datasets/raw/personal-style/gemini.jsonl

# Check stats
uv run scripts/parse_conversation_exports.py stats --input datasets/raw/personal-style/claude.jsonl
uv run scripts/parse_conversation_exports.py stats --input datasets/raw/personal-style/gemini.jsonl

# Merge + dedup
uv run scripts/parse_conversation_exports.py merge \
  --inputs datasets/raw/personal-style/claude.jsonl datasets/raw/personal-style/gemini.jsonl \
  --output datasets/raw/personal-style/all_conversations.jsonl
```

---

## Curate + Train

Once any dataset is ready, feed it into the training pipeline:

```bash
# Curate (quality filter, dedup, format, split)
uv run llm-curate --model code     # game-engine-docs, shader-pipeline, game-dev-github
uv run llm-curate --model creative # game-lore, project-plans, store-marketing
uv run llm-curate --model chat     # personal-style

# Train affected specialization
uv run llm-run --filter "*_code*" --resume
uv run llm-run --filter "*_creative*" --resume
uv run llm-run --filter "*_chat*" --resume
```

## Monitoring

```bash
# Generation progress
tail -f logs/dataset_generation.jsonl | python3 -m json.tool

# GPU memory (should stay under ~80GB)
watch -n 5 rocm-smi

# Resume after interruption — all scripts auto-resume from last checkpoint
uv run scripts/generate/generate_engine_docs.py  # picks up where it left off
```

---

## Current Progress (2026-04-22)

| Dataset | Status | Records | Curate-ready |
|---------|--------|---------|--------------|
| game-engine-docs | Generated (38% Unreal truncated) | 11,734 QA pairs | Yes |
| shader-pipeline | Done | 10,036 QA pairs | Yes |
| game-dev-github | Partial (1,256/15,000 target) | 1,256 conversations | Yes |
| store-marketing | Paused (rate limited) | 670/1,666 | No |
| project-plans | Not started | 0 | No |
| game-lore | Not started | 0 | No |
| personal-style | Exports collected, not processed | 0 | No |

**Next steps:**
1. Run `uv run llm-curate --model code` — all three code datasets have curate-ready data
2. Download base models (`uv run llm-download`)
3. Start first training run on code specialization
4. Continue github extraction to reach 15K target in parallel
5. Regenerate truncated Unreal engine docs (`--redo-failures`)
