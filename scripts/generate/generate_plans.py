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
    "Write a PRD for {feature}",
    "Create a milestone plan for the next 3 months",
    "Write a technical ADR for choosing between {option_a} and {option_b}",
    "Create a risk assessment for {project_element}",
    "Write a sprint retrospective summary and action items",
    "Plan the architecture for {system} considering {constraints}",
    "Write an incident postmortem and action items for {incident}",
]

PROMPT_TEMPLATE = """\
Generate a realistic project planning task and response.

Scenario: {task_type} for a {archetype}

Requirements for the output:
- Use concrete, realistic details (not generic filler)
- Include actual estimates, priorities, dependencies
- Format appropriately (markdown headers, tables, checklists)
- Reflect real tradeoffs and constraints

Output as JSON:
{{
  "instruction": "<the planning task/question>",
  "input": "",
  "output": "<the complete planning response>"
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
        prompt = PROMPT_TEMPLATE.format(task_type=task_type, archetype=archetype)
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
