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

# Phase 3: Extract from approved repos → multi-turn conversations
LLM_BASE_URL=http://192.168.50.117:11438/v1 LLM_API_KEY=unused \
  uv run scripts/generate/extract_github.py --target 15000  --worker-id 0 --num-workers 2
```

### Step 3: Shader Pipeline (~3-4h generation)

```bash
# 3a. Generate source chunks (run all three — they're independent)
uv run scripts/generate/export_glyph_shaders.py          # Unreal + Godot from Glyph DB → ~6K chunks
uv run scripts/generate/import_hf_shaders.py --limit 10000  # HuggingFace shader_dataset → 10K chunks
uv run scripts/generate/scrape_bookofshaders.py           # Book of Shaders repo → ~376 chunks

# 3b. Generate Q&A pairs from chunks
uv run scripts/generate/generate_shaders.py --target 7500

# Parallel (two workers):
uv run scripts/generate/generate_shaders.py --target 7500 --worker-id 0 --num-workers 2
LLM_BASE_URL=http://192.168.50.117:11438/v1 uv run scripts/generate/generate_shaders.py --target 7500 --worker-id 1 --num-workers 2
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
# Export from Claude.ai: Settings → Privacy → Export Data
# Place at: datasets/raw/personal-style/claude_export/conversations.json
#
# Export from Gemini: https://takeout.google.com → select "Gemini Apps"
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
