## Quick Start

All scripts live in `~/git/llm-training/scripts/generate/`. Run from the repo root.

```bash
cd ~/git/llm-training
```

### Prerequisites

```bash
# 1. Install deps
uv sync

# 2. Start the generator model (leave running in a dedicated terminal/tmux)
llama-server \
  --model qwen3.5:122b-a10b-Q4_K_M.gguf \
  --ctx-size 16384 \
  --n-gpu-layers 999 \
  --host 0.0.0.0 --port 8080

# 3. Verify it's responding
curl -s http://localhost:8080/v1/models | uv run -m json.tool
```

### Step 1: Personal Style (manual, ~30 min of your time)

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

### Step 2: Game Engine Docs (~8-12h generation) - IM HERE

```bash
# Try Glyph export first (if sources are indexed):
uv run scripts/generate/generate_engine_docs.py

# If Glyph sources aren't indexed, use --skip-glyph and provide chunks manually:
# Place scraped doc chunks at datasets/raw/game-engine-docs/*_chunks.jsonl
uv run scripts/generate/generate_engine_docs.py --skip-glyph
```

### Step 3: Project Plans + Store Marketing (parallel, ~1 day each)

These are fully synthetic — no scraping needed. Run them in parallel.

LLM_BASE_URL=https://openrouter.ai/api/v1 LLM_MODEL=openrouter/free LLM_API_KEY=REDACTED_KEY uv run scripts/generate/generate_plans.py --target 7500

```bash
# Terminal A — project plans (~10-14h)
uv run scripts/generate/generate_plans.py --target 7500

# Terminal B — store marketing (~3-4h)
# Note: populate Steam AppIDs in the script first, or provide scraped data
uv run scripts/generate/generate_marketing.py --target 5000
```

### Step 4: Game Lore (~16-24h generation)

Requires wiki chunks. Use Glyph to crawl wikis first, or scrape manually.

```bash
# 1. Review/edit genre config
$EDITOR datasets/raw/game-lore/genres.yaml

# 2. Ingest wikis via Glyph (configure sources in glyph.yaml first)
# cd ~/git/glyph && uv run glyph ingest

# 3. Export wiki chunks to the game-lore directory
# uv run glyph export -s mass-effect-wiki -V 2025 > \
#   ~/git/llm-training/datasets/raw/game-lore/mass-effect-wiki_chunks.jsonl

# 4. Generate creative writing pairs from chunks
uv run scripts/generate/generate_lore.py --target 12500
```

### Step 5: Game Dev GitHub (~1-2 weeks, quality gate is the bottleneck)

Three phases. Phase 3 requires human review between steps.

```bash
# Phase 1: Automated screening
uv run scripts/generate/screen_repo.py \
  --engine godot \
  https://github.com/godotengine/godot-demo-projects \
  https://github.com/dialogic-godot/dialogic

uv run scripts/generate/screen_repo.py \
  --engine unreal \
  https://github.com/EpicGames/Lyra

# Phase 2: Model-based quality review (auto-reads Phase 1 PASS results)
uv run scripts/generate/review_repo.py --engine godot
uv run scripts/generate/review_repo.py --engine unreal

# ── STOP: Human review ──
# Check: datasets/raw/game-dev-github/quality_reviews/*.json
# Populate: datasets/raw/game-dev-github/approved_repos.yaml

# Phase 3: Extract from approved repos → multi-turn conversations
uv run scripts/generate/extract_github.py --target 15000
```

### Step 6: Shader Pipeline (~3-4h generation)

```bash
# Export shader docs via Glyph (or place chunks manually)
# Place at: datasets/raw/shader-pipeline/*_chunks.jsonl

uv run scripts/generate/generate_shaders.py --target 7500
```

### Step 7: Curate + Train

Once any dataset is ready, feed it into the training pipeline:

```bash
# Curate (quality filter, dedup, format, split)
uv run llm-curate --model code    # for game-engine-docs, shader-pipeline, game-dev-github
uv run llm-curate --model creative # for game-lore, project-plans, store-marketing
uv run llm-curate --model chat    # for personal-style

# Train affected specialization
uv run llm-run --filter "*_code*" --resume
uv run llm-run --filter "*_creative*" --resume
uv run llm-run --filter "*_chat*" --resume
```

### Monitoring

```bash
# Generation progress
tail -f logs/dataset_generation.jsonl | uv run -m json.tool

# GPU memory (should stay under ~80GB)
watch -n 5 rocm-smi

# Resume after interruption — all scripts auto-resume from last checkpoint
uv run scripts/generate/generate_plans.py --target 7500  # picks up where it left off
```

---