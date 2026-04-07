#!/usr/bin/env python3
"""Generate game-lore dataset.

Pipeline: Load genres.yaml → scrape wikis (via Glyph or BeautifulSoup) →
generate creative writing tasks from lore references.
Output format: ShareGPT multi-turn.
Target: ~10-15K conversations, balanced across enabled genres.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from common import (
    RAW_DIR,
    append_jsonl,
    count_lines,
    generate,
    get_client,
    log,
    log_progress,
    read_jsonl,
)

DATASET = "game-lore"
OUTPUT_DIR = RAW_DIR / DATASET
GENRES_CONFIG = OUTPUT_DIR / "genres.yaml"

SYSTEM_PROMPT = "You are a world-building expert and creative writer specializing in game narrative design."

GENERATION_PROMPT = """\
Given this lore reference from {universe}:
---
{wiki_excerpt}
---

Generate a training pair where the instruction asks to CREATE original lore
(not reproduce existing lore) inspired by similar themes/structures.

Types of tasks to generate (pick one):
- "Write a codex entry for a [type] in a [genre] setting that [constraint]"
- "Create a faction profile including history, beliefs, and internal conflicts"
- "Write an in-game item description for [item type] that hints at [lore element]"
- "Design a magic/tech system with 3 tiers, including limitations and costs"
- "Write a short environmental narrative (found note, terminal log, inscription)"
- "Create a character background that connects to [broader setting element]"
- "Write dialogue between two characters from opposing factions about [topic]"
- "Design a quest narrative with branching moral choices"

The OUTPUT should be original creative writing, NOT a summary of the wiki content.
The wiki content is structural reference only — for tone, depth, and format.

Output as JSON (ShareGPT multi-turn):
{{"conversations": [
  {{"from": "human", "value": "Write a codex entry for an extinct alien species..."}},
  {{"from": "gpt", "value": "[full creative writing output]"}},
  {{"from": "human", "value": "Now add a second entry from a different in-universe perspective that contradicts the first"}},
  {{"from": "gpt", "value": "[contradicting perspective entry]"}}
]}}
"""


def load_genres() -> dict:
    """Load and validate genres.yaml."""
    if not GENRES_CONFIG.exists():
        log.error("genres.yaml not found at %s", GENRES_CONFIG)
        raise FileNotFoundError(f"Missing {GENRES_CONFIG}")

    config = yaml.safe_load(GENRES_CONFIG.read_text())
    genres = config.get("genres", {})

    enabled = {k: v for k, v in genres.items() if v.get("enabled", False)}
    log.info("Enabled genres: %s", list(enabled.keys()))

    # Normalize weights
    total_weight = sum(g.get("weight", 1.0) for g in enabled.values())
    for g in enabled.values():
        g["_normalized_weight"] = g.get("weight", 1.0) / total_weight

    return enabled


def main():
    parser = argparse.ArgumentParser(description="Generate game-lore dataset")
    parser.add_argument("--target", type=int, default=12500, help="Total examples to generate")
    parser.add_argument("--chunks-dir", type=Path, default=None,
                        help="Directory with pre-scraped wiki chunks (JSONL per source)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    genres = load_genres()

    raw_output = OUTPUT_DIR / "raw_responses.jsonl"
    done = count_lines(raw_output)

    if done >= args.target:
        log.info("Already have %d/%d examples", done, args.target)
        return

    # Calculate per-genre targets
    for name, genre in genres.items():
        genre["_target"] = int(args.target * genre["_normalized_weight"])
        log.info("  %s: %d examples (%.0f%%)", name, genre["_target"], genre["_normalized_weight"] * 100)

    # Check for available wiki chunks
    chunks_dir = args.chunks_dir or OUTPUT_DIR
    available_chunks = {}
    for genre_name, genre in genres.items():
        for source in genre.get("sources", []):
            slug = source["name"].lower().replace(" ", "-").replace("(", "").replace(")", "")
            chunk_file = chunks_dir / f"{slug}_chunks.jsonl"
            if chunk_file.exists():
                available_chunks[slug] = {
                    "path": chunk_file,
                    "genre": genre_name,
                    "universe": source["name"],
                }
                log.info("Found chunks: %s (%d records)", slug, count_lines(chunk_file))

    if not available_chunks:
        log.warning("No wiki chunks available. Run Glyph ingest or scrape wikis first.")
        log.warning("Expected chunk files at: %s/<source-slug>_chunks.jsonl", chunks_dir)
        return

    # Generate from available chunks
    client = get_client()
    log_progress(DATASET, "generate", "running", progress=f"{done}/{args.target}")

    for slug, info in available_chunks.items():
        chunks = read_jsonl(info["path"])
        genre_name = info["genre"]
        universe = info["universe"]
        genre_target = genres[genre_name]["_target"]

        for i, chunk in enumerate(chunks):
            if done >= args.target:
                break
            excerpt = chunk.get("text", chunk.get("content", json.dumps(chunk)))[:3000]
            prompt = GENERATION_PROMPT.format(universe=universe, wiki_excerpt=excerpt)
            response = generate(client, prompt, system=SYSTEM_PROMPT, max_tokens=2048)
            append_jsonl(raw_output, {
                "index": done,
                "genre": genre_name,
                "universe": universe,
                "source_slug": slug,
                "response": response,
            })
            done += 1
            if done % 100 == 0:
                log_progress(DATASET, "generate", "running", progress=f"{done}/{args.target}")

    log_progress(DATASET, "generate", "done", records=done)
    log.info("game-lore generation complete")


if __name__ == "__main__":
    main()
