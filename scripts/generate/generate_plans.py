#!/usr/bin/env python3
"""Generate project-plans dataset.

Fully synthetic — no scraping required. Generates planning documents
across game dev, SaaS, infrastructure, and open-source project archetypes.
Output format: Alpaca (instruction/input/output).
Target: ~5-10K examples.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

from common import (
    RAW_DIR,
    append_jsonl,
    count_lines,
    generate,
    get_client,
    log,
    log_progress,
    repair_json,
)

DATASET = "project-plans"
OUTPUT_DIR = RAW_DIR / DATASET

SYSTEM_PROMPT = "You are an experienced technical project manager and software architect."

# Project archetypes × task types = generation matrix
PROJECT_ARCHETYPES = [
    "indie game (small team, 6-12 month cycle)",
    "game jam (48-72 hour rapid prototyping)",
    "SaaS product (continuous delivery, feature flags)",
    "infrastructure migration (phased rollout, rollback plans)",
    "open source library (semver, RFC process, community input)",
    "mobile game (live service, seasonal content updates)",
    "VR/AR experience (hardware constraints, performance targets)",
    "game engine plugin (cross-platform, backward compatibility)",
]

TASK_TYPES = [
    "Break down this feature into subtasks with estimates",
    "Write a PRD for a new feature",
    "Create a milestone plan for the next 3 months",
    "Write a technical ADR for choosing between two competing approaches",
    "Create a risk assessment for a critical project component",
    "Write a sprint retrospective summary and action items",
    "Plan the architecture for a core system considering real-world constraints",
    "Write an incident postmortem and action items for a production outage",
]

# Concrete details the model can draw from — one picked at random per generation
# to add variety without relying on the model to invent everything.
SCENARIO_DETAILS = {
    "indie game (small team, 6-12 month cycle)": [
        "2D roguelike with procedural generation, 3 developers, Unity engine",
        "narrative RPG with branching dialogue, 4 developers, Godot engine",
        "multiplayer arena shooter, 2 developers, Unreal Engine",
        "cozy farming sim with crafting system, 3 developers, custom C++ engine",
    ],
    "game jam (48-72 hour rapid prototyping)": [
        "theme: 'connected worlds', solo developer, Godot",
        "theme: 'one button', 2-person team, Unity",
        "theme: 'decay', 3-person team, PICO-8",
        "theme: 'ancient technology', solo developer, Raylib + C",
    ],
    "SaaS product (continuous delivery, feature flags)": [
        "B2B analytics dashboard, React + Python, 8-person team, AWS",
        "developer tool CLI with cloud sync, Go + TypeScript, 5-person team, GCP",
        "e-commerce platform with real-time inventory, Next.js + Rust, 12-person team",
        "project management tool with AI features, Rails + React, 6-person team",
    ],
    "infrastructure migration (phased rollout, rollback plans)": [
        "migrating from self-hosted Postgres to Aurora, 200+ microservices",
        "Kubernetes migration from EKS to on-prem k3s, 50 services",
        "monolith to microservices decomposition, Java to Go, 18-month timeline",
        "datacenter migration from US-East to multi-region, 99.99% SLA requirement",
    ],
    "open source library (semver, RFC process, community input)": [
        "Rust async runtime with 5K GitHub stars, 3 core maintainers",
        "Python data validation library, 200 contributors, used by 10K projects",
        "JavaScript state management library, major version bump planned",
        "Go HTTP framework with middleware ecosystem, breaking API changes proposed",
    ],
    "mobile game (live service, seasonal content updates)": [
        "match-3 puzzle game with guild system, Unity, 15-person team",
        "idle RPG with gacha mechanics, React Native, 8-person team",
        "location-based AR game, Kotlin + Swift, 20-person team",
        "card battler with ranked PvP, Godot, 10-person team",
    ],
    "VR/AR experience (hardware constraints, performance targets)": [
        "VR fitness app targeting Quest 3, Unity, must hold 90fps",
        "AR museum guide for iOS, ARKit + SwiftUI, 4-person team",
        "VR multiplayer escape room, Unreal Engine, 6-person team",
        "mixed reality productivity tool, Quest Pro, 8-person team",
    ],
    "game engine plugin (cross-platform, backward compatibility)": [
        "procedural terrain generator for Unity, must support URP and HDRP",
        "dialogue system plugin for Godot 4.x, GDScript + C# bindings",
        "networking plugin for Unreal Engine, must support Steam and EOS",
        "AI behavior tree editor for Unity, visual scripting interface",
    ],
}

PROMPT_TEMPLATE = """\
Generate a realistic project planning task and response.

Project type: {archetype}
Project details: {details}
Task: {task_type}

Requirements for the output:
- Use concrete, realistic details drawn from the project context above
- Include actual estimates, priorities, dependencies where relevant
- Format the output field with markdown (headers, tables, checklists)
- Reflect real tradeoffs and constraints for this specific project
- The output field must be a single markdown string, not nested JSON

Output as JSON:
{{
  "instruction": "<the specific planning task/question>",
  "input": "",
  "output": "<the complete planning response in markdown>"
}}
"""


def main():
    parser = argparse.ArgumentParser(description="Generate project-plans dataset")
    parser.add_argument("--target", type=int, default=7500, help="Number of examples to generate")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model name (default: LLM_MODEL env var)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_output = OUTPUT_DIR / "raw_responses.jsonl"
    done = count_lines(raw_output)

    if done >= args.target:
        log.info("Already have %d/%d examples, skipping", done, args.target)
        return

    client = get_client()
    model = args.model
    log_progress(DATASET, "generate", "running", progress=f"{done}/{args.target}")

    # Build generation matrix and shuffle for diversity
    combos = list(itertools.product(PROJECT_ARCHETYPES, TASK_TYPES))
    random.seed(42)
    random.shuffle(combos)

    # Cycle through combos to reach target
    combo_cycle = itertools.cycle(combos)
    for i in range(done, args.target):
        archetype, task_type = next(combo_cycle)
        details = random.choice(SCENARIO_DETAILS[archetype])
        prompt = PROMPT_TEMPLATE.format(task_type=task_type, archetype=archetype, details=details)
        response = generate(client, prompt, system=SYSTEM_PROMPT, model=model, max_tokens=8192)
        parsed = repair_json(response) if response else None
        append_jsonl(raw_output, {
            "index": i,
            "archetype": archetype,
            "task_type": task_type,
            "response": response,
            "parsed": parsed is not None,
        })
        if (i + 1) % 100 == 0:
            log_progress(DATASET, "generate", "running", progress=f"{i + 1}/{args.target}")

    log_progress(DATASET, "generate", "done", records=count_lines(raw_output))
    log.info("project-plans generation complete")


if __name__ == "__main__":
    main()
