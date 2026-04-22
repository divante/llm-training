#!/usr/bin/env python3
"""Phase 3: Extract code from approved repos → multi-turn conversations.

Reads approved_repos.yaml, clones approved repos, extracts code units,
generates multi-turn ShareGPT conversations via LLM.
Target: ~10-20K multi-turn conversations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import tempfile
from pathlib import Path

import yaml

from common import (
    PROVIDER,
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

DATASET = "game-dev-github"
OUTPUT_DIR = RAW_DIR / DATASET
APPROVED_REPOS = OUTPUT_DIR / "approved_repos.yaml"

SYSTEM_PROMPT = "You are a senior game developer who teaches through realistic conversations grounded in production code."

ENGINE_EMPHASIS: dict[str, str] = {
    "unreal": """\
Context: This is Unreal Engine C++ code.
- Use UE5 conventions: UPROPERTY/UFUNCTION macros, UObject lifecycle, component architecture
- Reference relevant UE5 subsystems (GAS, Enhanced Input, Slate, etc.) where the code touches them
- Show Blueprint integration where applicable (BlueprintCallable, BlueprintNativeEvent)
- Discuss memory management (TSharedPtr, garbage collection, weak references) when relevant""",

    "godot": """\
Context: This is Godot engine code.
- Use Godot 4 conventions: @export annotations, signal declarations, node hierarchy
- Reference relevant Godot systems (SceneTree, physics, animation) where the code touches them
- Show both GDScript and C# approaches when the pattern differs between them
- Discuss Godot-specific patterns (scene composition, autoloads, resources) when relevant""",
}

CONVERSATION_PROMPT = """\
Based on this game development source code:

File: {file_path}
Repository: {repo_url}
---
{code}
---

Generate a realistic multi-turn conversation where a developer asks for help
implementing a similar system. The conversation should teach the architectural
patterns and engine-specific best practices demonstrated in this code.

{engine_emphasis}

Conversation guidelines:
- Start with a specific problem the developer is trying to solve (not generic "how do I implement X")
- Vary the conversation structure naturally — don't always follow the same pattern
- The teaching should emerge from the code's actual patterns, not from a checklist of principles
- Include complete, compilable code in implementation turns
- Follow-up questions should explore edge cases, testing, or integration specific to THIS code
- Aim for {n_turns} turns total, but let the conversation flow naturally

Output as JSON:
{{"conversations": [
  {{"from": "human", "value": "..."}},
  {{"from": "gpt", "value": "..."}},
  ...
]}}

CRITICAL RULES:
- Ground the conversation in the specific patterns from the source code above
- DO NOT open with generic principle lectures ("This is a classic case where SRP...")
- Each conversation must be unique — vary the developer's experience level, problem framing, and follow-up direction
- Code must be complete and compilable, inspired by the reference patterns (not copied verbatim)
- Use EXACTLY "human" and "gpt" as the "from" values — never "assistant", "user", or anything else
"""

GEMINI_BREVITY = """
Keep responses focused and practical:
- GPT responses should be 200-500 words max, not multi-page essays
- Lead with the key insight or solution, then show code
- One code block per turn unless the question specifically asks for multiple files
- Skip verbose preambles and summaries — get to the point
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


TURN_COUNTS = [4, 6, 6, 8, 8, 10]


def main():
    parser = argparse.ArgumentParser(description="Extract code from approved repos (Phase 3)")
    add_worker_args(parser)
    args = parser.parse_args()

    repos = load_approved_repos()
    if not repos:
        log.warning("No approved repos. Populate approved_repos.yaml first.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_output = get_output_path(OUTPUT_DIR, args.worker_id)
    processed_keys = load_all_processed_keys(OUTPUT_DIR, ("repo", "file"))
    done = len(read_jsonl(raw_output))

    if done >= args.target:
        log.info("Already have %d conversations", done)
        return

    client = get_client()
    log_progress(DATASET, "extract", "running", progress=f"{done}/{args.target}")

    for i, repo_config in enumerate(repos):
        if (i % args.num_workers) != args.worker_id:
            continue

        url = repo_config["url"]
        slug = url.rstrip("/").split("/")[-1]

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

                if (url, unit["file_path"]) in processed_keys:
                    continue

                engine = unit["engine"]
                engine_emphasis = ENGINE_EMPHASIS.get(engine, "")
                n_turns = random.choice(TURN_COUNTS)

                prompt = CONVERSATION_PROMPT.format(
                    file_path=unit["file_path"],
                    repo_url=url,
                    engine=engine,
                    code=unit["content"][:6000],
                    engine_emphasis=engine_emphasis,
                    n_turns=n_turns,
                )

                if PROVIDER == "gemini":
                    prompt += GEMINI_BREVITY

                temp = 0.5 if PROVIDER == "local" else 0.7

                try:
                    response = generate(client, prompt, system=SYSTEM_PROMPT, temperature=temp)
                except Exception as e:
                    log.warning("Skipping %s/%s after LLM error: %s", slug, unit["file_path"], e)
                    continue

                parsed = repair_json(response)
                if not parsed or not isinstance(parsed, dict) or "conversations" not in parsed:
                    log.warning("Skipping %s/%s: response failed JSON repair", slug, unit["file_path"])
                    continue

                convos = parsed["conversations"]
                valid_convos = []
                for turn in convos:
                    if not isinstance(turn, dict):
                        continue
                    if "from" not in turn and "value" not in turn:
                        continue
                    if "from" not in turn:
                        for candidate in ("human", "gpt", "assistant"):
                            if candidate in turn:
                                turn["value"] = turn.pop(candidate)
                                turn["from"] = "gpt" if candidate in ("gpt", "assistant") else "human"
                                break
                        else:
                            continue
                    if turn.get("from") == "assistant":
                        turn["from"] = "gpt"
                    if turn.get("from") not in ("human", "gpt") or not turn.get("value"):
                        continue
                    valid_convos.append({"from": turn["from"], "value": turn["value"]})
                parsed["conversations"] = valid_convos

                if len(valid_convos) < 4:
                    log.warning("Skipping %s/%s: too few valid turns (%d)", slug, unit["file_path"], len(valid_convos))
                    continue

                append_jsonl(raw_output, {
                    "repo": url,
                    "file": unit["file_path"],
                    "engine": engine,
                    "quality_score": unit["quality_score"],
                    "response": json.dumps(parsed, ensure_ascii=False),
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
