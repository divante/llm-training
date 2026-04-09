#!/usr/bin/env bash
# Cron wrapper for generate_marketing.py
# Sources credentials, sets working dir, stops cleanly on rate limit.
set -euo pipefail

. /home/shepard/.bash_env

export LLM_BASE_URL="${OPENROUTER_BASE_URL}"
export LLM_MODEL="openrouter/free"
export LLM_API_KEY="${OPENROUTER_API_KEY}"

cd /home/shepard/git/llm-training
exec /home/shepard/.local/bin/uv run scripts/generate/generate_marketing.py --target 5000 --skip-scrape
