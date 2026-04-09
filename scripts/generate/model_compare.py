#!/usr/bin/env python3
"""Quick model comparison for planning document generation quality."""

import json
import time
from openai import OpenAI

OLLAMA_URL = "http://192.168.50.233:11434/v1"
MODELS = [
    "qwen2.5-coder:14b",
    "qwen2.5:14b",
    "gemma3:12b",
    "qwen3:14b",
    "qwen3.5:9b",
]

SYSTEM = "You are an experienced technical project manager and software architect."

PROMPT = """\
Generate a realistic project planning task and response.

Scenario: Write a technical ADR for choosing between ECS and a custom entity system for a indie game (small team, 6-12 month cycle)

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

client = OpenAI(base_url=OLLAMA_URL, api_key="unused")

for model in MODELS:
    print(f"\n{'='*60}")
    print(f"MODEL: {model}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": PROMPT},
            ],
            max_tokens=8192,
            temperature=0.7,
        )
        elapsed = time.time() - t0
        content = resp.choices[0].message.content
        tokens = resp.usage.completion_tokens if resp.usage else 0
        tps = tokens / elapsed if elapsed > 0 else 0

        print(f"Time: {elapsed:.1f}s | Tokens: {tokens} | Speed: {tps:.1f} tk/s")
        print(f"Output length: {len(content)} chars")
        print(f"\n--- First 1500 chars ---")
        print(content[:1500])
        print(f"\n--- Last 500 chars ---")
        print(content[-500:])

        # Try parsing
        try:
            # Strip code fences
            text = content.strip()
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            parsed = json.loads(text.strip())
            out_len = len(parsed.get("output", ""))
            print(f"\nJSON parse: OK | output field: {out_len} chars")
        except Exception as e:
            print(f"\nJSON parse: FAILED ({e})")

    except Exception as e:
        print(f"ERROR: {e}")
