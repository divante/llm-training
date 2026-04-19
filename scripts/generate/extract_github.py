#!/usr/bin/env python3
"""Phase 3: Extract code from approved repos → multi-turn conversations.

Reads approved_repos.yaml, clones approved repos, extracts code units,
generates multi-turn ShareGPT conversations via local LLM.
Target: ~10-20K multi-turn conversations.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
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

DATASET = "game-dev-github"
OUTPUT_DIR = RAW_DIR / DATASET
APPROVED_REPOS = OUTPUT_DIR / "approved_repos.yaml"

SYSTEM_PROMPT = "You are a senior game developer creating training conversations that teach clean, SOLID code architecture."

CONVERSATION_PROMPT = """\
Based on this real game development code:

File: {file_path}
Engine: {engine}
---
{code}
---

Generate a realistic multi-turn conversation where a developer asks for help
implementing something similar. The conversation should teach SOLID principles
and {engine} best practices.

Structure:
Turn 1 (human): "I need to implement [what this code does]. How should I structure it?"
Turn 2 (gpt): Architecture/approach discussion (emphasizing SOLID/DRY)
Turn 3 (human): "Can you write the implementation?"
Turn 4 (gpt): Complete implementation code (inspired by the reference, not copied verbatim)
Turn 5 (human): Follow-up question (testing, edge cases, optimization, or extension)
Turn 6 (gpt): Detailed follow-up answer

Output as JSON:
{{"conversations": [
  {{"from": "human", "value": "..."}},
  {{"from": "gpt", "value": "..."}},
  ...
]}}

Requirements:
- The implementation should follow SOLID principles, avoid code duplication,
  and use {engine} best practices
- Don't copy the source code verbatim — teach the PATTERNS, not the specific implementation
- Code must be complete and compilable
"""


def load_approved_repos() -> list[dict]:
    """Load approved repos from YAML config."""
    if not APPROVED_REPOS.exists():
        log.error("approved_repos.yaml not found at %s", APPROVED_REPOS)
        return []

    config = yaml.safe_load(APPROVED_REPOS.read_text())
    repos = config.get("repos", [])
    return [r for r in repos if r.get("include_mode") != "exclude"]


def extract_code_units(repo_path: Path, repo_config: dict) -> list[dict]:
    """Extract individual code units (classes/functions) from a repo."""
    ext_map = {
        "unreal": {".cpp", ".h", ".hpp"},
        "godot": {".gd", ".cs"},
    }
    target_exts = ext_map.get(repo_config.get("engine", ""), {".cpp", ".h", ".gd", ".cs"})

    include_dirs = repo_config.get("include_dirs")
    exclude_dirs = set(repo_config.get("exclude_dirs", []))
    skip_dirs = {".git", "build", "bin", "third_party", "node_modules", "Intermediate", "Saved"}

    units = []
    for f in sorted(repo_path.rglob("*")):
        if not f.is_file() or f.suffix not in target_exts:
            continue
        if any(p in f.parts for p in skip_dirs):
            continue

        rel = str(f.relative_to(repo_path))

        # Respect include/exclude dirs
        if include_dirs and not any(rel.startswith(d) for d in include_dirs):
            continue
        if any(rel.startswith(d) for d in exclude_dirs):
            continue

        try:
            content = f.read_text(errors="replace")
        except Exception:
            continue

        lines = content.splitlines()
        if len(lines) > 500 or len(lines) < 10:
            continue

        units.append({
            "file_path": rel,
            "content": content,
            "engine": repo_config.get("engine", "unknown"),
            "repo_url": repo_config.get("url", ""),
            "quality_score": repo_config.get("quality_score", 0),
        })

    return units


import hashlib

def main():
    parser = argparse.ArgumentParser(description="Extract code from approved repos (Phase 3)")
    parser.add_argument("--target", type=int, default=15000, help="Target conversation count")
    parser.add_argument("--worker-id", type=int, default=0, help="ID of this worker (0-indexed)")
    parser.add_argument("--num-workers", type=int, default=1, help="Total number of parallel workers")
    args = parser.parse_args()

    repos = load_approved_repos()
    if not repos:
        log.warning("No approved repos. Populate approved_repos.yaml first.")
        return

    raw_output = OUTPUT_DIR / "raw_responses.jsonl"

    # Truly resumable: load already processed files
    log.info("Loading processed records for deduplication...")
    processed_records = read_jsonl(raw_output)
    processed_units = { (r["repo"], r["file"]) for r in processed_records }
    done = len(processed_records)

    if done >= args.target:
        log.info("Already have %d conversations", done)
        return

    client = get_client()
    log_progress(DATASET, "extract", "running", progress=f"{done}/{args.target}")

    for i, repo_config in enumerate(repos):
        # Deterministic repository assignment (Parallelism)
        # Worker 0 takes 0, 2, 4... Worker 1 takes 1, 3, 5...
        if (i % args.num_workers) != args.worker_id:
            continue

        url = repo_config["url"]
        slug = url.rstrip("/").split("/")[-1]

        # Skip clone if all known files already processed
        include_dirs = repo_config.get("include_dirs", [])
        if include_dirs and all((url, f) in processed_units for f in include_dirs):
            log.info("Skipping %s — all %d files already processed", slug, len(include_dirs))
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            clone_path = Path(tmpdir) / "repo"
            log.info("Cloning %s ...", url)
            result = subprocess.run(
                ["git", "clone", "--depth=1", url, str(clone_path)],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                log.warning("Failed to clone %s: %s", url, result.stderr[:200])
                continue

            units = extract_code_units(clone_path, repo_config)
            log.info("Extracted %d code units from %s", len(units), slug)

            for unit in units:
                if done >= args.target:
                    break

                # Skip if already processed (Resumability)
                if (url, unit["file_path"]) in processed_units:
                    continue

                prompt = CONVERSATION_PROMPT.format(
                    file_path=unit["file_path"],
                    engine=unit["engine"],
                    code=unit["content"][:6000],
                )
                try:
                    response = generate(client, prompt, system=SYSTEM_PROMPT, max_tokens=4096)
                except Exception as e:
                    log.warning("Skipping %s/%s after LLM error: %s", slug, unit["file_path"], e)
                    continue
                append_jsonl(raw_output, {
                    "repo": url,
                    "file": unit["file_path"],
                    "engine": unit["engine"],
                    "quality_score": unit["quality_score"],
                    "response": response,
                })
                done += 1
                if done % 50 == 0:
                    log_progress(DATASET, "extract", "running", progress=f"{done}/{args.target}")

        if done >= args.target:
            break

    log_progress(DATASET, "extract", "done", records=done)
    log.info("game-dev-github extraction complete: %d conversations", done)


if __name__ == "__main__":
    main()
