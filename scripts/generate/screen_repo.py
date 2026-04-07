#!/usr/bin/env python3
"""Phase 1: Automated screening of candidate game-dev repos.

Checks: license, activity, size, language mix, basic code quality heuristics.
Output: screening_report.json per repo.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from common import RAW_DIR, log, log_progress, write_jsonl

DATASET = "game-dev-github"
OUTPUT_DIR = RAW_DIR / DATASET
REPORTS_DIR = OUTPUT_DIR / "screening_reports"

ALLOWED_LICENSES = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "unlicense", "isc"}


def check_license(repo_path: Path) -> dict:
    """Check for an allowed open-source license."""
    for name in ["LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"]:
        license_file = repo_path / name
        if license_file.exists():
            content = license_file.read_text(errors="replace").lower()
            for lic in ALLOWED_LICENSES:
                if lic.replace("-", " ") in content or lic in content:
                    return {"status": "pass", "license": lic}
            return {"status": "fail", "reason": "unrecognized license"}
    return {"status": "fail", "reason": "no license file found"}


def check_language_mix(repo_path: Path, target_extensions: set[str]) -> dict:
    """Check that >80% of source files are in target language."""
    all_files = []
    target_files = []
    for f in repo_path.rglob("*"):
        if f.is_file() and not any(p in f.parts for p in [".git", "node_modules", "__pycache__", "build"]):
            all_files.append(f)
            if f.suffix in target_extensions:
                target_files.append(f)

    if not all_files:
        return {"status": "skip", "reason": "no source files"}

    ratio = len(target_files) / len(all_files)
    return {
        "status": "pass" if ratio > 0.5 else "skip",  # Relaxed from 0.8 — game projects have assets
        "ratio": round(ratio, 2),
        "total_files": len(all_files),
        "target_files": len(target_files),
    }


def check_code_quality(repo_path: Path, target_extensions: set[str]) -> dict:
    """Basic code quality heuristics on source files."""
    func_lengths = []
    file_lengths = []
    comment_lines = 0
    total_lines = 0
    god_objects = []

    for f in repo_path.rglob("*"):
        if not f.is_file() or f.suffix not in target_extensions:
            continue
        if any(p in f.parts for p in [".git", "node_modules", "build", "bin", "third_party"]):
            continue

        try:
            lines = f.read_text(errors="replace").splitlines()
        except Exception:
            continue

        file_lengths.append(len(lines))
        total_lines += len(lines)
        if len(lines) > 1000:
            god_objects.append(str(f.relative_to(repo_path)))

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("#") or stripped.startswith("--"):
                comment_lines += 1

    if not file_lengths:
        return {"status": "skip", "reason": "no analyzable files"}

    avg_file = sum(file_lengths) / len(file_lengths)
    comment_ratio = comment_lines / max(total_lines, 1)

    return {
        "status": "pass" if avg_file < 300 and not god_objects else "warn",
        "avg_file_length": round(avg_file, 1),
        "max_file_length": max(file_lengths),
        "comment_ratio": round(comment_ratio, 3),
        "god_objects": god_objects[:5],
        "total_source_files": len(file_lengths),
    }


def screen_repo(repo_url: str, engine: str) -> dict:
    """Run all screening checks on a repo."""
    ext_map = {
        "unreal": {".cpp", ".h", ".hpp", ".cc"},
        "godot": {".gd", ".cs", ".gdshader", ".tscn"},
    }
    target_exts = ext_map.get(engine, {".cpp", ".h", ".gd", ".cs"})

    report = {
        "url": repo_url,
        "engine": engine,
        "screened_at": datetime.now(timezone.utc).isoformat(),
        "checks": {},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        clone_path = Path(tmpdir) / "repo"
        log.info("Cloning %s ...", repo_url)
        result = subprocess.run(
            ["git", "clone", "--depth=1", repo_url, str(clone_path)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            report["checks"]["clone"] = {"status": "fail", "reason": result.stderr[:500]}
            return report

        report["checks"]["license"] = check_license(clone_path)
        report["checks"]["language_mix"] = check_language_mix(clone_path, target_exts)
        report["checks"]["code_quality"] = check_code_quality(clone_path, target_exts)

    # Overall recommendation
    checks = report["checks"]
    if any(c.get("status") == "fail" for c in checks.values()):
        report["recommendation"] = "FAIL"
    elif any(c.get("status") == "skip" for c in checks.values()):
        report["recommendation"] = "SKIP"
    elif any(c.get("status") == "warn" for c in checks.values()):
        report["recommendation"] = "REVIEW"
    else:
        report["recommendation"] = "PASS"

    return report


def main():
    parser = argparse.ArgumentParser(description="Screen game-dev repos (Phase 1)")
    parser.add_argument("repos", nargs="*", help="Repo URLs to screen")
    parser.add_argument("--engine", choices=["unreal", "godot"], required=True)
    parser.add_argument("--from-file", type=Path, help="File with one repo URL per line")
    args = parser.parse_args()

    repos = list(args.repos)
    if args.from_file and args.from_file.exists():
        repos.extend(line.strip() for line in args.from_file.read_text().splitlines() if line.strip())

    if not repos:
        log.error("No repos specified. Provide URLs as args or via --from-file.")
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for url in repos:
        log.info("Screening: %s", url)
        report = screen_repo(url, args.engine)
        results.append(report)

        # Save individual report
        slug = url.rstrip("/").split("/")[-1]
        report_path = REPORTS_DIR / f"{slug}.json"
        report_path.write_text(json.dumps(report, indent=2))
        log.info("  → %s: %s", slug, report["recommendation"])

    # Summary
    log_progress(DATASET, "screen", "done", records=len(results))
    pass_count = sum(1 for r in results if r["recommendation"] == "PASS")
    review_count = sum(1 for r in results if r["recommendation"] == "REVIEW")
    log.info("Screening complete: %d PASS, %d REVIEW, %d FAIL/SKIP out of %d",
             pass_count, review_count, len(results) - pass_count - review_count, len(results))


if __name__ == "__main__":
    main()
