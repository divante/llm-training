#!/usr/bin/env python3
"""Scrape Shadertoy public API for GLSL shader code.

Uses the Shadertoy REST API to fetch public shaders and saves them
as chunks for the shader-pipeline dataset.

Output: datasets/raw/shader-pipeline/shadertoy_chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

from common import RAW_DIR, append_jsonl, log, log_progress, read_jsonl

DATASET = "shader-pipeline"
OUTPUT_DIR = RAW_DIR / DATASET

API_BASE = "https://www.shadertoy.com/api/v1"
API_KEY = "BtHtWD"  # Public demo key

# Sorting options: "name", "love", "popular", "newest", "hot"
DEFAULT_SORT = "popular"
REQUEST_DELAY = 0.3  # seconds between API calls


def fetch_shader_ids(sort_by: str = DEFAULT_SORT) -> list[str]:
    """Fetch all public shader IDs from the API."""
    url = f"{API_BASE}/shaders?sort={sort_by}&key={API_KEY}"
    log.info("Fetching shader ID list (sort=%s)...", sort_by)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    shader_ids = data.get("Results", [])
    log.info("Found %d public shaders", len(shader_ids))
    return shader_ids


def fetch_shader(shader_id: str) -> dict | None:
    """Fetch a single shader's full data."""
    url = f"{API_BASE}/shaders/{shader_id}?key={API_KEY}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("Shader")
    except Exception as e:
        log.warning("Failed to fetch shader %s: %s", shader_id, e)
        return None


def extract_chunk(shader: dict) -> dict | None:
    """Extract a chunk record from a shader API response.

    Filters out multipass shaders and shaders outside the 10-500 line range.
    """
    info = shader.get("info", {})
    renderpasses = shader.get("renderpass", [])

    # Skip multipass shaders (complex, hard to learn from in isolation)
    image_passes = [rp for rp in renderpasses if rp.get("type") == "image"]
    if len(image_passes) != 1:
        return None

    code = image_passes[0].get("code", "")
    lines = code.strip().splitlines()

    # Filter by line count
    if len(lines) < 10 or len(lines) > 500:
        return None

    # Build text chunk: title + description + code
    title = info.get("name", "Untitled")
    description = info.get("description", "")
    tags = info.get("tags", [])

    parts = [f"// Shadertoy: {title}"]
    if description:
        parts.append(f"// {description[:500]}")
    if tags:
        parts.append(f"// Tags: {', '.join(tags)}")
    parts.append("// Language: GLSL (Fragment Shader)")
    parts.append("")
    parts.append(code)

    return {
        "text": "\n".join(parts),
        "shader_id": info.get("id", ""),
        "title": title,
        "tags": tags,
        "likes": info.get("likes", 0),
        "views": info.get("viewed", 0),
        "line_count": len(lines),
    }


def main():
    parser = argparse.ArgumentParser(description="Scrape Shadertoy shaders")
    parser.add_argument("--limit", type=int, default=10000, help="Max shaders to scrape")
    parser.add_argument("--sort", default=DEFAULT_SORT, help="Sort order: popular, love, newest, hot")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY, help="Delay between requests (s)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "shadertoy_chunks.jsonl"

    # Load already-processed IDs for resume
    existing = read_jsonl(output_path)
    processed_ids = {r["shader_id"] for r in existing}
    done = len(existing)
    log.info("Resuming from %d existing chunks", done)

    if done >= args.limit:
        log.info("Already have %d shaders (limit=%d)", done, args.limit)
        return

    # Fetch all shader IDs
    all_ids = fetch_shader_ids(sort_by=args.sort)
    remaining = [sid for sid in all_ids if sid not in processed_ids]
    log.info("%d shaders remaining after filtering processed", len(remaining))

    log_progress(DATASET, "scrape-shadertoy", "running", progress=f"{done}/{args.limit}")

    skipped = 0
    for i, shader_id in enumerate(remaining):
        if done >= args.limit:
            break

        shader = fetch_shader(shader_id)
        time.sleep(args.delay)

        if shader is None:
            skipped += 1
            continue

        chunk = extract_chunk(shader)
        if chunk is None:
            skipped += 1
            continue

        append_jsonl(output_path, chunk)
        done += 1

        if done % 100 == 0:
            log.info("Progress: %d/%d scraped (%d skipped)", done, args.limit, skipped)
            log_progress(DATASET, "scrape-shadertoy", "running", progress=f"{done}/{args.limit}")

    log_progress(DATASET, "scrape-shadertoy", "done", records=done)
    log.info("Shadertoy scrape complete: %d chunks (%d skipped)", done, skipped)


if __name__ == "__main__":
    main()
