#!/usr/bin/env python3
"""Generate store-marketing dataset.

Scrapes real store descriptions (Steam AppDetails API) then generates
synthetic instruction pairs for store copy, patch notes, press kits.
Output format: Alpaca (instruction/input/output).
Target: ~5K examples.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests
import yaml

from common import (
    RAW_DIR,
    append_jsonl,
    count_lines,
    generate,
    get_client,
    log,
    log_progress,
    write_jsonl,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

DATASET = "store-marketing"
OUTPUT_DIR = RAW_DIR / DATASET

SYSTEM_PROMPT = "You are a game marketing copywriter with expertise in store listings, patch notes, and press materials."

# Steam AppDetails API (public, no auth needed)
STEAM_API = "https://store.steampowered.com/api/appdetails"

GENERATION_PROMPT = """\
Here is a real game store description for reference:
---
Title: {title}
Genre: {genre}
Description: {description}
---

Generate 3 training pairs as a JSON array. Each pair should teach a model to write
compelling game marketing copy. Types to generate:

1. A store description task: "Write a Steam store description for a [genre] game about [premise]"
2. A patch notes task: "Write patch notes for version X.Y that adds [features] and fixes [bugs]"
3. One of: press kit one-pager / Early Access announcement / rewrite-to-improve

Output format:
[
  {{"instruction": "...", "input": "", "output": "..."}},
  {{"instruction": "...", "input": "", "output": "..."}},
  {{"instruction": "...", "input": "", "output": "..."}}
]

Requirements:
- Outputs should be professional quality, not generic
- Vary tone: some serious, some lighthearted
- Include concrete details (features, numbers, dates)
"""


def load_steam_app_ids() -> list[int]:
    """Load Steam AppIDs from config/steam_apps.yaml."""
    config_path = CONFIG_DIR / "steam_apps.yaml"
    if not config_path.exists():
        log.error("steam_apps.yaml not found at %s", config_path)
        raise FileNotFoundError(f"Missing {config_path}")

    config = yaml.safe_load(config_path.read_text())
    app_ids = []
    for genre, entries in config.items():
        for entry in entries:
            app_ids.append(entry["id"])
    log.info("Loaded %d Steam AppIDs across %d genres", len(app_ids), len(config))
    return app_ids


def scrape_steam_descriptions(app_ids: list[int], output_path: Path) -> list[dict]:
    """Fetch game descriptions from Steam's public API."""
    if output_path.exists():
        existing = []
        with open(output_path) as f:
            for line in f:
                existing.append(json.loads(line))
        if len(existing) >= len(app_ids):
            log.info("Steam descriptions already scraped (%d), skipping", len(existing))
            return existing

    results = []
    for app_id in app_ids:
        try:
            resp = requests.get(STEAM_API, params={"appids": app_id}, timeout=10)
            data = resp.json()
            app_data = data.get(str(app_id), {}).get("data", {})
            if app_data:
                record = {
                    "app_id": app_id,
                    "title": app_data.get("name", ""),
                    "genre": ", ".join(g.get("description", "") for g in app_data.get("genres", [])),
                    "description": app_data.get("detailed_description", "")[:2000],
                    "short_description": app_data.get("short_description", ""),
                }
                results.append(record)
                append_jsonl(output_path, record)
        except Exception as e:
            log.warning("Failed to fetch app %d: %s", app_id, e)
        time.sleep(1.5)  # Rate limit

    log_progress(DATASET, "scrape", "done", records=len(results))
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate store-marketing dataset")
    parser.add_argument("--target", type=int, default=5000, help="Target example count")
    parser.add_argument("--skip-scrape", action="store_true", help="Skip Steam scraping, use existing data")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_output = OUTPUT_DIR / "raw_responses.jsonl"
    scraped_path = OUTPUT_DIR / "steam_descriptions.jsonl"

    # Phase 1: Scrape (or load existing)
    if not args.skip_scrape:
        app_ids = load_steam_app_ids()
        if app_ids:
            scrape_steam_descriptions(app_ids, scraped_path)

    # Phase 2: Generate from scraped descriptions
    done = count_lines(raw_output)
    descriptions = []
    if scraped_path.exists():
        with open(scraped_path) as f:
            descriptions = [json.loads(line) for line in f if line.strip()]

    if not descriptions:
        log.warning("No scraped descriptions available. Populate Steam AppIDs or provide data manually.")
        return

    client = get_client()
    target_calls = args.target // 3

    for i in range(done, target_calls):
        desc = descriptions[i % len(descriptions)]
        prompt = GENERATION_PROMPT.format(
            title=desc["title"],
            genre=desc["genre"],
            description=desc["description"][:1500],
        )
        response = generate(client, prompt, system=SYSTEM_PROMPT)
        append_jsonl(raw_output, {
            "index": i,
            "source_app_id": desc.get("app_id"),
            "response": response,
        })
        if (i + 1) % 100 == 0:
            log_progress(DATASET, "generate", "running", progress=f"{i + 1}/{target_calls}")

    log_progress(DATASET, "generate", "done", records=count_lines(raw_output))
    log.info("store-marketing generation complete")


if __name__ == "__main__":
    main()
