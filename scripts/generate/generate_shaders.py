#!/usr/bin/env python3
"""Generate shader-pipeline dataset.

Pipeline: Glyph export / HuggingFace / Book of Shaders → Q&A generation.
Covers GLSL/HLSL fundamentals, UE5 materials, Godot shaders, pipeline concepts.
Output format: Alpaca (instruction/input/output).
Target: ~5-10K examples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    RAW_DIR,
    append_jsonl,
    generate,
    get_client,
    log,
    log_progress,
    read_jsonl,
)

DATASET = "shader-pipeline"
OUTPUT_DIR = RAW_DIR / DATASET

SYSTEM_PROMPT = "You are a graphics programming expert specializing in real-time rendering, shader development, and GPU pipeline optimization."

SHADER_QA_PROMPT = """\
Given this shader/graphics reference material:
---
{content}
---

Generate {n} diverse instruction/output pairs for training a coding assistant
on shader and render pipeline topics.

Types to include:
- "Write a shader that does X" → full GLSL/HLSL/Godot shader code
- "Explain how this shader works" → line-by-line breakdown
- "Port this to {target_language}" → converted code
- "Optimize this shader for {platform}" → optimized version
- "Create a UE5 material that..." → material node setup + Custom HLSL
- "Debug this shader..." → identify and fix the issue
- "What's the performance impact of..." → analysis with alternatives

Output as JSON array:
[
  {{"instruction": "...", "input": "", "output": "..."}}
]

Requirements:
- Complete, working shader code (not fragments)
- Specify target API/language (GLSL, HLSL, Godot Shading Language)
- Include comments explaining non-obvious techniques
- Cover 2D and 3D use cases
"""


def load_all_chunks() -> list[dict]:
    """Load all chunk files and tag each with source file + index."""
    chunk_files = sorted(OUTPUT_DIR.glob("*_chunks.jsonl"))
    if not chunk_files:
        return []

    all_chunks = []
    for chunk_file in chunk_files:
        chunks = read_jsonl(chunk_file)
        for i, chunk in enumerate(chunks):
            chunk["_source_file"] = chunk_file.name
            chunk["_chunk_index"] = i
            all_chunks.append(chunk)

    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Generate shader-pipeline dataset")
    parser.add_argument("--target", type=int, default=7500, help="Target example count")
    parser.add_argument("--n-per-chunk", type=int, default=5, help="Q&A pairs per source chunk")
    parser.add_argument("--worker-id", type=int, default=0, help="ID of this worker (0-indexed)")
    parser.add_argument("--num-workers", type=int, default=1, help="Total number of parallel workers")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_output = OUTPUT_DIR / "raw_responses.jsonl"

    # Load all chunks across all source files
    all_chunks = load_all_chunks()
    if not all_chunks:
        log.warning("No shader chunks available. Run export/scrape scripts first.")
        log.warning("Expected: %s/*_chunks.jsonl", OUTPUT_DIR)
        return

    log.info("Loaded %d total chunks from %d files",
             len(all_chunks),
             len(set(c["_source_file"] for c in all_chunks)))

    # Resumability: track processed (source_file, chunk_index) pairs
    processed_records = read_jsonl(raw_output)
    processed_keys = {(r["source"], r["chunk_index"]) for r in processed_records}
    done = len(processed_records)

    if done >= args.target:
        log.info("Already have %d examples (target=%d)", done, args.target)
        return

    client = get_client()
    log_progress(DATASET, "generate", "running", progress=f"{done}/{args.target}")

    for i, chunk in enumerate(all_chunks):
        if done >= args.target:
            break

        # Deterministic worker assignment
        if (i % args.num_workers) != args.worker_id:
            continue

        source_file = chunk["_source_file"]
        chunk_index = chunk["_chunk_index"]

        # Skip already processed
        if (source_file, chunk_index) in processed_keys:
            continue

        content = chunk.get("text", chunk.get("content", chunk.get("code", "")))[:4000]
        if not content or len(content) < 50:
            continue

        prompt = SHADER_QA_PROMPT.format(
            content=content,
            n=args.n_per_chunk,
            target_language="Godot Shading Language",
            platform="mobile",
        )

        try:
            response = generate(client, prompt, system=SYSTEM_PROMPT, max_tokens=4096)
        except Exception as e:
            log.warning("Skipping %s[%d] after LLM error: %s", source_file, chunk_index, e)
            continue

        append_jsonl(raw_output, {
            "source": source_file,
            "chunk_index": chunk_index,
            "response": response,
        })
        done += 1

        if done % 50 == 0:
            log_progress(DATASET, "generate", "running", progress=f"{done}/{args.target}")

    log_progress(DATASET, "generate", "done", records=done)
    log.info("shader-pipeline generation complete: %d examples", done)


if __name__ == "__main__":
    main()
