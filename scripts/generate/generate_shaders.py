#!/usr/bin/env python3
"""Generate shader-pipeline dataset.

Pipeline: Glyph export / Shadertoy scrape → Q&A generation.
Covers GLSL/HLSL fundamentals, UE5 materials, Godot shaders, pipeline concepts.
Output format: Alpaca (instruction/input/output).
Target: ~5-10K examples.
"""

from __future__ import annotations

import argparse
import json
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


def main():
    parser = argparse.ArgumentParser(description="Generate shader-pipeline dataset")
    parser.add_argument("--target", type=int, default=7500, help="Target example count")
    parser.add_argument("--n-per-chunk", type=int, default=5, help="Q&A pairs per source chunk")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_output = OUTPUT_DIR / "raw_responses.jsonl"
    done = count_lines(raw_output)

    if done >= args.target:
        log.info("Already have %d examples", done)
        return

    # Look for available chunks (Glyph exports or scraped data)
    chunk_files = list(OUTPUT_DIR.glob("*_chunks.jsonl"))
    if not chunk_files:
        log.warning("No shader chunks available. Run Glyph export or scrape Shadertoy/docs first.")
        log.warning("Expected: %s/*_chunks.jsonl", OUTPUT_DIR)
        return

    client = get_client()
    log_progress(DATASET, "generate", "running", progress=f"{done}/{args.target}")

    for chunk_file in chunk_files:
        chunks = read_jsonl(chunk_file)
        for i, chunk in enumerate(chunks):
            if done >= args.target:
                break
            content = chunk.get("text", chunk.get("content", chunk.get("code", "")))[:4000]
            prompt = SHADER_QA_PROMPT.format(
                content=content,
                n=args.n_per_chunk,
                target_language="Godot Shading Language",
                platform="mobile",
            )
            response = generate(client, prompt, system=SYSTEM_PROMPT)
            append_jsonl(raw_output, {
                "index": done,
                "source": chunk_file.name,
                "chunk_index": i,
                "response": response,
            })
            done += 1
            if done % 100 == 0:
                log_progress(DATASET, "generate", "running", progress=f"{done}/{args.target}")

    log_progress(DATASET, "generate", "done", records=done)
    log.info("shader-pipeline generation complete")


if __name__ == "__main__":
    main()
