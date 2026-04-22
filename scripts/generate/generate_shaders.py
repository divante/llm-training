#!/usr/bin/env python3
"""Generate shader-pipeline dataset.

Pipeline: Glyph export / HuggingFace / Book of Shaders → Q&A generation.
Covers GLSL/HLSL fundamentals, UE5 materials, Godot shaders, pipeline concepts.
Output format: Alpaca (instruction/input/output).
Target: ~5-10K examples.
"""

from __future__ import annotations

import argparse
import json
import random

from common import (
    RAW_DIR,
    add_worker_args,
    append_jsonl,
    generate,
    get_client,
    get_output_path,
    load_all_processed_keys,
    log,
    log_progress,
    read_jsonl,
    repair_json,
)

DATASET = "shader-pipeline"
OUTPUT_DIR = RAW_DIR / DATASET

SYSTEM_PROMPT = "You are a graphics programming expert specializing in real-time rendering, shader development, and GPU pipeline optimization."

TARGET_LANGUAGES = [
    "Godot Shading Language",
    "HLSL (DirectX 12)",
    "GLSL (OpenGL 4.6 / Vulkan)",
    "Metal Shading Language",
    "WGSL (WebGPU)",
]

PLATFORMS = [
    "mobile (ARM Mali/Adreno)",
    "integrated GPU (AMD APU)",
    "Nintendo Switch",
    "Steam Deck",
    "WebGL 2.0",
]

SOURCE_EMPHASIS: dict[str, str] = {
    "hf_shader_chunks.jsonl": """\
Focus areas for this GLSL/HLSL shader code:
- Explain the specific algorithm or technique used and why it works
- Optimize it for {platform} with concrete changes (not generic advice)
- Port it to {target_language}, adapting API-specific features
- Extend it with a specific visual variation (e.g. add fog, animate a parameter, combine with another effect)
- Debug a plausible issue (introduce a realistic bug, then fix it)""",

    "godot_shader_chunks.jsonl": """\
Focus areas for this Godot engine shader/rendering content:
- Implement the described technique as a complete Godot shader with shader_type declaration
- Explain Godot-specific concepts (hints, render modes, built-in uniforms) referenced in the material
- Convert the approach to a different Godot shader type (spatial ↔ canvas_item ↔ particles)
- Show how to control shader parameters from GDScript with concrete code
- Optimize for {platform} with Godot-specific considerations""",

    "unreal_shader_chunks.jsonl": """\
Focus areas for this Unreal Engine rendering/material content:
- Create a UE5 material graph that implements the described technique, with Custom HLSL nodes where needed
- Explain the C++ rendering pipeline class/method and when to use vs alternatives
- Write a complete Custom HLSL node for a material that uses the described functionality
- Show how to set up material parameters and drive them from Blueprints/C++
- Port the technique to {target_language} outside of Unreal""",

    "bookofshaders_chunks.jsonl": """\
Focus areas for this Book of Shaders / GLSL fundamentals content:
- Implement the mathematical concept as a complete fragment shader with concrete values
- Explain the math (SDF, noise, transforms) with step-by-step breakdown of specific operations
- Create a creative variation that combines this technique with another (e.g. noise + color mapping)
- Port to {target_language}, noting API differences in built-in functions
- Show how to animate/parameterize the effect for interactive use""",
}

DEFAULT_EMPHASIS = """\
Focus areas:
- Write a complete shader implementing a specific technique from the source material
- Explain the algorithm with references to specific functions and parameters
- Port to {target_language} with API-specific adaptations
- Optimize for {platform} with concrete, measurable changes
- Extend with a specific visual enhancement"""

SHADER_QA_PROMPT = """\
Given this shader/graphics reference material:
---
{content}
---

Generate {n} instruction/output pairs for training a coding assistant.

{emphasis}

Output as JSON array:
[
  {{"instruction": "...", "input": "", "output": "..."}}
]

CRITICAL RULES:
- Every instruction MUST reference specific techniques, functions, or concepts from the source material above
- DO NOT use generic placeholders like "Write a shader that does X" or "Explain how this shader works"
- Each instruction must be self-contained and distinct from the others
- Include complete, working shader code (not fragments)
- Specify the target API/language in each instruction
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
    add_worker_args(parser)
    parser.add_argument("--n-per-chunk", type=int, default=5, help="Q&A pairs per source chunk")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_output = get_output_path(OUTPUT_DIR, args.worker_id)

    all_chunks = load_all_chunks()
    if not all_chunks:
        log.warning("No shader chunks available. Run export/scrape scripts first.")
        log.warning("Expected: %s/*_chunks.jsonl", OUTPUT_DIR)
        return

    log.info("Loaded %d total chunks from %d files",
             len(all_chunks),
             len(set(c["_source_file"] for c in all_chunks)))

    processed_keys = load_all_processed_keys(OUTPUT_DIR, ("source", "chunk_index"))
    done = len(read_jsonl(raw_output))

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

        target_lang = random.choice(TARGET_LANGUAGES)
        platform = random.choice(PLATFORMS)
        emphasis_template = SOURCE_EMPHASIS.get(source_file, DEFAULT_EMPHASIS)
        emphasis = emphasis_template.format(target_language=target_lang, platform=platform)

        prompt = SHADER_QA_PROMPT.format(
            content=content,
            n=args.n_per_chunk,
            emphasis=emphasis,
        )

        try:
            response = generate(client, prompt, system=SYSTEM_PROMPT)
        except Exception as e:
            log.warning("Skipping %s[%d] after LLM error: %s", source_file, chunk_index, e)
            continue

        parsed = repair_json(response)
        if not parsed or not isinstance(parsed, list) or len(parsed) == 0:
            log.warning("Skipping %s[%d]: response failed JSON repair", source_file, chunk_index)
            continue

        append_jsonl(raw_output, {
            "source": source_file,
            "chunk_index": chunk_index,
            "response": json.dumps(parsed, ensure_ascii=False),
        })
        done += 1

        if done % 50 == 0:
            log_progress(DATASET, "generate", "running", progress=f"{done}/{args.target}")

    log_progress(DATASET, "generate", "done", records=done)
    log.info("shader-pipeline generation complete: %d examples", done)


if __name__ == "__main__":
    main()
