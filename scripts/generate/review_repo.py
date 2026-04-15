#!/usr/bin/env python3
"""Phase 2: Model-based design quality review of screened repos.

Takes repos that passed Phase 1 screening, evaluates code quality via local LLM.
Scores: SRP, OCP, DI, DRY, modularity, naming, error handling, engine best practices.
Threshold: ≥3.5 average to include.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from common import (
    RAW_DIR,
    append_jsonl,
    generate,
    get_client,
    log,
    log_progress,
)

DATASET = "game-dev-github"
OUTPUT_DIR = RAW_DIR / DATASET
REPORTS_DIR = OUTPUT_DIR / "screening_reports"
REVIEWS_DIR = OUTPUT_DIR / "quality_reviews"

REVIEW_PROMPT = """\
Evaluate this codebase for design quality. Score each category 1-5:

1. **Single Responsibility:** Do classes/modules have one clear purpose?
2. **Open/Closed:** Can behavior be extended without modifying existing code?
3. **Dependency Inversion:** Do high-level modules depend on abstractions?
4. **DRY:** Is there significant code duplication? Shared utilities?
5. **Modularity:** Clear separation of concerns? Could you swap out a subsystem?
6. **Naming/Readability:** Are names descriptive? Is the code self-documenting?
7. **Error Handling:** Graceful failure? Or crash-and-pray?
8. **Engine Best Practices:** Does it follow {engine}-specific conventions?

Here is a representative sample of the codebase:
---
{code_sample}
---

Output as JSON:
{{
  "scores": {{
    "single_responsibility": 0,
    "open_closed": 0,
    "dependency_inversion": 0,
    "dry": 0,
    "modularity": 0,
    "naming_readability": 0,
    "error_handling": 0,
    "engine_best_practices": 0
  }},
  "overall": 0.0,
  "strengths": ["...", "...", "..."],
  "weaknesses": ["...", "...", "..."],
  "recommendation": "INCLUDE | PARTIAL | EXCLUDE",
  "partial_include_paths": [],
  "notes": "..."
}}
"""


def collect_code_sample(repo_path: Path, target_exts: set[str], max_chars: int = 12000) -> str:
    """Collect a representative code sample from the repo."""
    files = []
    for f in sorted(repo_path.rglob("*")):
        if f.is_file() and f.suffix in target_exts:
            if not any(p in f.parts for p in [".git", "build", "bin", "third_party", "node_modules"]):
                files.append(f)

    # Pick diverse files (first, middle, last by path sort + some from different dirs)
    sample_files = []
    if len(files) <= 6:
        sample_files = files
    else:
        step = len(files) // 5
        sample_files = [files[i] for i in range(0, len(files), max(step, 1))][:6]

    sample = ""
    for f in sample_files:
        try:
            content = f.read_text(errors="replace")[:2000]
            rel = f.relative_to(repo_path)
            sample += f"\n// === {rel} ===\n{content}\n"
            if len(sample) > max_chars:
                break
        except Exception:
            continue

    return sample[:max_chars]


def review_repo(repo_url: str, engine: str) -> dict:
    """Run model-based quality review on a repo."""
    ext_map = {
        "unreal": {".cpp", ".h", ".hpp"},
        "godot": {".gd", ".cs", ".gdshader"},
    }
    target_exts = ext_map.get(engine, {".cpp", ".h", ".gd"})

    with tempfile.TemporaryDirectory() as tmpdir:
        clone_path = Path(tmpdir) / "repo"
        try:
            result = subprocess.run(
                ["git", "clone", "--depth=1", repo_url, str(clone_path)],
                capture_output=True, text=True, timeout=300,
            )
        except subprocess.TimeoutExpired:
            return {"url": repo_url, "error": "Clone timed out"}
        if result.returncode != 0:
            return {"url": repo_url, "error": f"Clone failed: {result.stderr[:200]}"}

        code_sample = collect_code_sample(clone_path, target_exts)
        if not code_sample.strip():
            return {"url": repo_url, "error": "No source files found"}

    client = get_client()
    prompt = REVIEW_PROMPT.format(engine=engine, code_sample=code_sample)
    response = generate(client, prompt, max_tokens=2048, temperature=0.3)

    return {
        "url": repo_url,
        "engine": engine,
        "review_response": response,
    }


def main():
    parser = argparse.ArgumentParser(description="Model-based quality review (Phase 2)")
    parser.add_argument("--engine", choices=["unreal", "godot"], required=True)
    parser.add_argument("repos", nargs="*", help="Repo URLs (or reads from Phase 1 PASS reports)")
    args = parser.parse_args()

    repos = list(args.repos)

    # Auto-discover repos that passed Phase 1 (filtered by engine)
    if not repos and REPORTS_DIR.exists():
        for report_file in REPORTS_DIR.glob("*.json"):
            report = json.loads(report_file.read_text())
            if report.get("engine") == args.engine and report.get("recommendation") in ("PASS", "REVIEW"):
                repos.append(report["url"])
        log.info("Auto-discovered %d %s repos from Phase 1 screening", len(repos), args.engine)

    if not repos:
        log.error("No repos to review. Run screen_repo.py first or provide URLs.")
        return

    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    for url in repos:
        slug = url.rstrip("/").split("/")[-1]
        review_path = REVIEWS_DIR / f"{slug}.json"
        if review_path.exists():
            log.info("Skipping %s (already reviewed)", slug)
            continue

        log.info("Reviewing: %s", url)
        review = review_repo(url, args.engine)
        review_path.write_text(json.dumps(review, indent=2))
        log.info("  → saved to %s", review_path)

    log_progress(DATASET, "review", "done", records=len(repos))
    log.info("Quality review complete. Check %s for results.", REVIEWS_DIR)
    log.info("Next: human review → populate approved_repos.yaml → run extract_github.py")


if __name__ == "__main__":
    main()
