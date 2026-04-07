#!/usr/bin/env python3
"""Generate game-engine-docs dataset.

Pipeline: Read filtered manifest (from filter_chunks.py) → Q&A generation via local LLM.
Uses tier-aware Q&A counts: Tier 1 = 5 per doc, Tier 2 = 2 per doc.
Output format: Alpaca (instruction/input/output).

Prerequisite: Run filter_chunks.py first to produce *_filtered.jsonl manifests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import (
    RAW_DIR,
    append_jsonl,
    count_lines,
    generate,
    get_client,
    log,
    log_progress,
    read_jsonl,
    repair_json,
    write_jsonl,
)

DATASET = "game-engine-docs"
OUTPUT_DIR = RAW_DIR / DATASET

# Engine display names for prompt
ENGINE_NAMES = {
    "unreal": "Unreal Engine 5 (C++)",
    "godot-api": "Godot 4.x",
    "godot-docs": "Godot 4.x (GDScript)",
}

ENGINE_VERSIONS = {
    "unreal": "5.7",
    "godot-api": "4.7",
    "godot-docs": "4.7",
}

SYSTEM_PROMPT = "You are generating training data for a coding assistant specialized in game development."

QA_PROMPT_TEMPLATE = """\
You are generating training data for a coding assistant specialized in {engine} game development.

Given the following documentation excerpt:
---
{doc_content}
---

Generate {n} diverse instruction/output pairs. Each pair should be a realistic question
a game developer would ask, paired with a correct, detailed answer.

Vary the question types:
- "How do I..." (task-oriented)
- "What's the difference between..." (conceptual)
- "Debug this..." (troubleshooting, include broken code)
- "Convert this from..." (e.g., Blueprint logic to C++, GDScript to C#)
- "Optimize..." (performance-focused)
- "Explain..." (understanding)

Output format (JSON array):
[
  {{
    "instruction": "How do I implement a health component in UE5 C++?",
    "input": "",
    "output": "Here's how to create a reusable health component..."
  }}
]

Requirements:
- Answers must be technically accurate to {engine_version}
- Include complete, compilable code snippets (not fragments)
- Use modern patterns ({engine}-specific best practices)
- Vary difficulty: beginner (30%), intermediate (50%), advanced (20%)
"""


def generate_qa_from_manifest(manifest_path: Path, source_name: str) -> None:
    """Generate Q&A pairs from a filtered manifest JSONL."""
    if not manifest_path.exists():
        log.warning("No manifest for %s at %s — run filter_chunks.py first", source_name, manifest_path)
        return

    docs = read_jsonl(manifest_path)
    if not docs:
        log.warning("Empty manifest for %s, skipping", source_name)
        return

    engine = ENGINE_NAMES.get(source_name, source_name)
    engine_version = ENGINE_VERSIONS.get(source_name, "latest")
    raw_output = OUTPUT_DIR / f"{source_name}_qa.jsonl"
    done = count_lines(raw_output)

    if done >= len(docs):
        log.info("Already generated %d/%d for %s, skipping", done, len(docs), source_name)
        return

    client = get_client()
    log.info("Generating Q&A for %s: %d docs (%d already done)", source_name, len(docs), done)
    log_progress(DATASET, f"generate_qa/{source_name}", "running", progress=f"{done}/{len(docs)}")

    for i, doc in enumerate(docs[done:], start=done):
        n = doc.get("qa_count", 5)
        if n <= 0:
            continue

        prompt = QA_PROMPT_TEMPLATE.format(
            engine=engine,
            doc_content=doc["text"][:4000],
            n=n,
            engine_version=engine_version,
        )
        response = generate(client, prompt, system=SYSTEM_PROMPT, max_tokens=8192)
        qa_pairs = repair_json(response)
        append_jsonl(raw_output, {
            "source": source_name,
            "path": doc["path"],
            "engine": engine,
            "tier": doc.get("tier", 1),
            "qa_count": n,
            "response": response,
            "parsed": qa_pairs is not None,
            "pair_count": len(qa_pairs) if qa_pairs else 0,
        })
        if (i + 1) % 50 == 0:
            log_progress(DATASET, f"generate_qa/{source_name}", "running",
                         progress=f"{i + 1}/{len(docs)}")

    log_progress(DATASET, f"generate_qa/{source_name}", "done", records=count_lines(raw_output))


def _is_truncated(response: str | None) -> bool:
    """Check if an LLM response was cut off before completing the JSON array."""
    if not response:
        return True
    text = response.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return not text.strip().endswith("]")


def redo_failures(manifest_path: Path, source_name: str) -> None:
    """Re-generate entries that are truncated or unparseable."""
    if not manifest_path.exists():
        log.warning("No manifest for %s at %s", source_name, manifest_path)
        return

    docs = read_jsonl(manifest_path)
    raw_output = OUTPUT_DIR / f"{source_name}_qa.jsonl"
    entries = read_jsonl(raw_output)
    if not entries:
        log.warning("No existing output for %s, nothing to redo", source_name)
        return

    # Find indices that need redo
    redo_indices = []
    for i, entry in enumerate(entries):
        resp = entry.get("response")
        if _is_truncated(resp) or repair_json(resp) is None:
            redo_indices.append(i)

    if not redo_indices:
        log.info("No failures to redo for %s", source_name)
        return

    engine = ENGINE_NAMES.get(source_name, source_name)
    engine_version = ENGINE_VERSIONS.get(source_name, "latest")
    client = get_client()
    log.info("Redoing %d failed entries for %s", len(redo_indices), source_name)
    log_progress(DATASET, f"redo/{source_name}", "running", progress=f"0/{len(redo_indices)}")

    for count, idx in enumerate(redo_indices):
        if idx >= len(docs):
            log.warning("Index %d out of range for manifest (%d docs), skipping", idx, len(docs))
            continue

        doc = docs[idx]
        n = doc.get("qa_count", 5)
        if n <= 0:
            continue

        prompt = QA_PROMPT_TEMPLATE.format(
            engine=engine,
            doc_content=doc["text"][:4000],
            n=n,
            engine_version=engine_version,
        )
        response = generate(client, prompt, system=SYSTEM_PROMPT, max_tokens=8192)
        qa_pairs = repair_json(response)
        entries[idx] = {
            "source": source_name,
            "path": doc["path"],
            "engine": engine,
            "tier": doc.get("tier", 1),
            "qa_count": n,
            "response": response,
            "parsed": qa_pairs is not None,
            "pair_count": len(qa_pairs) if qa_pairs else 0,
        }
        if (count + 1) % 25 == 0:
            log.info("Redo progress: %d/%d", count + 1, len(redo_indices))
            log_progress(DATASET, f"redo/{source_name}", "running",
                         progress=f"{count + 1}/{len(redo_indices)}")

    # Write back the full file with replacements
    write_jsonl(raw_output, entries)
    log_progress(DATASET, f"redo/{source_name}", "done", records=len(entries))
    log.info("Redo complete for %s — replaced %d entries", source_name, len(redo_indices))


def main():
    parser = argparse.ArgumentParser(description="Generate game-engine-docs dataset")
    parser.add_argument("--source", type=str, default=None,
                        help="Only process a specific source (unreal, godot-api, godot-docs)")
    parser.add_argument("--redo-failures", action="store_true",
                        help="Re-generate truncated/unparseable entries instead of continuing")
    args = parser.parse_args()

    sources = list(ENGINE_NAMES.keys())
    if args.source:
        if args.source not in ENGINE_NAMES:
            log.error("Unknown source: %s (valid: %s)", args.source, ", ".join(sources))
            sys.exit(1)
        sources = [args.source]

    for source_name in sources:
        manifest_path = OUTPUT_DIR / f"{source_name}_filtered.jsonl"
        if args.redo_failures:
            redo_failures(manifest_path, source_name)
        else:
            generate_qa_from_manifest(manifest_path, source_name)

    log.info("game-engine-docs generation complete")


if __name__ == "__main__":
    main()
