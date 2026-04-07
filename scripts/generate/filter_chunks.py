#!/usr/bin/env python3
"""Filter game-engine-docs into Tier 1/2/3 before Q&A generation.

Reads markdown doc files from glyph exports, classifies each into tiers
using keyword/filename/UE-naming-convention matching (no model calls),
outputs a filtered manifest JSONL that generate_engine_docs.py consumes.

Usage:
    python filter_chunks.py [--config CONFIG] [--dry-run]

Output per source: datasets/raw/game-engine-docs/{source}_filtered.jsonl
Each line: {"path": "...", "text": "...", "tier": 1, "qa_count": 5, "source": "unreal", ...}
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from common import RAW_DIR, log, write_jsonl

DATASET = "game-engine-docs"
OUTPUT_DIR = RAW_DIR / DATASET
CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "chunk_filter.yaml"

SOURCES = [
    {"name": "unreal", "version": "5.7", "config_key": "unreal"},
    {"name": "godot-api", "version": "4.7", "config_key": "godot_api"},
    {"name": "godot-docs", "version": "4.7", "config_key": "godot_docs"},
]

QA_COUNTS = {1: 5, 2: 2, 3: 0}


def load_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def compile_patterns(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p) for p in patterns]


def _content_stats(content: str) -> dict[str, int]:
    """Extract content metrics."""
    lines = [l for l in content.strip().splitlines() if l.strip()]
    return {
        "bytes": len(content.encode()),
        "lines": len(lines),
        "methods": content.count("### "),
    }


def _is_skipped_unreal(stem: str, module: str, stats: dict, cfg: dict) -> bool:
    """Check if a file matches Unreal skip rules."""
    skip = cfg.get("skip", {})

    if stats["bytes"] < skip.get("min_content_bytes", 100):
        return True
    if stats["lines"] <= skip.get("max_trivial_lines", 8):
        return True

    for pat in compile_patterns(skip.get("module_patterns", [])):
        if pat.search(module):
            return True
    for pat in compile_patterns(skip.get("filename_patterns", [])):
        if pat.search(stem):
            return True

    return False


def classify_unreal(filename: str, content: str, cfg: dict) -> int:
    """Classify an Unreal doc file using module whitelist + UE naming conventions."""
    stem = filename.removesuffix(".md")
    parts = stem.split(".")
    module = parts[0]
    classname = parts[-1] if len(parts) >= 2 else ""
    stats = _content_stats(content)

    # --- Skip rules ---
    if _is_skipped_unreal(stem, module, stats, cfg):
        return 3

    # --- Tier 1: priority module whitelist ---
    t1 = cfg.get("tier1", {})
    if module in t1.get("modules", []):
        return 1

    # --- Tier 1: U/A class promotion (user-facing UObject/AActor classes) ---
    if t1.get("promote_ua_classes") and classname.startswith(("U", "A")):
        if (stats["methods"] >= t1.get("ua_min_methods", 3)
                and stats["lines"] >= t1.get("ua_min_lines", 15)
                and stats["bytes"] >= t1.get("ua_min_bytes", 200)):
            return 1

    # --- Tier 2: module overview files (no dot = top-level module page) ---
    t2 = cfg.get("tier2", {})
    if t2.get("include_module_overviews") and "." not in stem:
        if (stats["lines"] >= t2.get("overview_min_lines", 10)
                and stats["bytes"] >= t2.get("overview_min_bytes", 300)):
            return 2

    # --- Tier 2: substantial F-structs ---
    if t2.get("include_f_structs") and classname.startswith("F"):
        if (stats["methods"] >= t2.get("f_struct_min_methods", 5)
                and stats["lines"] >= t2.get("f_struct_min_lines", 20)
                and stats["bytes"] >= t2.get("f_struct_min_bytes", 500)):
            return 2

    # --- Default ---
    return cfg.get("default_tier", 3)


def classify_godot_api(filename: str, content: str, cfg: dict) -> int:
    """Classify a Godot API doc file."""
    stem = filename.removesuffix(".md")
    stats = _content_stats(content)

    # --- Skip ---
    skip = cfg.get("skip", {})
    if stats["bytes"] < skip.get("min_content_bytes", 80):
        return 3
    if stats["lines"] <= skip.get("max_trivial_lines", 6):
        return 3
    for pat in compile_patterns(skip.get("class_patterns", [])):
        if pat.search(stem):
            return 3

    # --- Tier 1: class whitelist ---
    t1 = cfg.get("tier1", {})
    if stem in t1.get("classes", []):
        return 1
    for kw in t1.get("content_keywords", []):
        if kw in content:
            return 1

    # --- Tier 2: sufficient substance ---
    t2 = cfg.get("tier2", {})
    if (stats["bytes"] >= t2.get("min_content_bytes", 200)
            and stats["lines"] >= t2.get("min_lines", 8)):
        return 2

    return cfg.get("default_tier", 3)


def classify_godot_docs(filename: str, content: str, cfg: dict) -> int:
    """Classify a Godot tutorial/docs file. Default: Tier 1."""
    stem = filename.removesuffix(".md")
    stats = _content_stats(content)

    skip = cfg.get("skip", {})
    if stats["bytes"] < skip.get("min_content_bytes", 200):
        return 3
    for pat in compile_patterns(skip.get("filename_patterns", [])):
        if pat.search(stem):
            return 3

    return cfg.get("default_tier", 1)


CLASSIFIERS = {
    "unreal": classify_unreal,
    "godot_api": classify_godot_api,
    "godot_docs": classify_godot_docs,
}


def filter_source(source: dict, cfg: dict, dry_run: bool = False) -> dict[str, int]:
    """Filter all docs for a single source. Returns tier counts."""
    source_dir = OUTPUT_DIR / source["name"] / source["version"]
    if not source_dir.is_dir():
        log.warning("Missing source dir: %s", source_dir)
        return {}

    config_key = source["config_key"]
    source_cfg = cfg.get(config_key, {})
    classifier = CLASSIFIERS.get(config_key)
    if not classifier:
        log.error("No classifier for config_key=%s", config_key)
        return {}

    md_files = sorted(source_dir.rglob("*.md"))
    output_path = OUTPUT_DIR / f"{source['name']}_filtered.jsonl"

    counts = {1: 0, 2: 0, 3: 0}
    records = []

    for md_file in md_files:
        content = md_file.read_text(errors="replace").strip()
        rel_path = str(md_file.relative_to(source_dir))

        tier = classifier(md_file.name, content, source_cfg)
        counts[tier] += 1

        if tier <= 2:
            records.append({
                "path": rel_path,
                "text": content,
                "tier": tier,
                "qa_count": QA_COUNTS[tier],
                "source": source["name"],
            })

    log.info(
        "%s: %d total → T1: %d (%d Q&A), T2: %d (%d Q&A), T3: %d (skip)",
        source["name"],
        len(md_files),
        counts[1], counts[1] * QA_COUNTS[1],
        counts[2], counts[2] * QA_COUNTS[2],
        counts[3],
    )
    total_qa = counts[1] * QA_COUNTS[1] + counts[2] * QA_COUNTS[2]
    log.info("  → %d filtered docs, ~%d Q&A pairs", counts[1] + counts[2], total_qa)

    if not dry_run:
        write_jsonl(output_path, records)
        log.info("  → Wrote %s", output_path)

    return counts


def main():
    parser = argparse.ArgumentParser(description="Filter game-engine-docs chunks into tiers")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Filter config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log.info("Loaded filter config from %s", args.config)

    grand_total = {1: 0, 2: 0, 3: 0}
    for source in SOURCES:
        counts = filter_source(source, cfg, dry_run=args.dry_run)
        for tier, count in counts.items():
            grand_total[tier] += count

    total_docs = sum(grand_total.values())
    filtered = grand_total[1] + grand_total[2]
    total_qa = grand_total[1] * QA_COUNTS[1] + grand_total[2] * QA_COUNTS[2]

    log.info("=" * 60)
    log.info("TOTAL: %d docs → T1: %d, T2: %d, T3: %d (skip)",
             total_docs, grand_total[1], grand_total[2], grand_total[3])
    log.info("Filtered: %d docs → ~%d Q&A pairs", filtered, total_qa)
    log.info("Reduction: %.0f%% of docs skipped", grand_total[3] / max(total_docs, 1) * 100)


if __name__ == "__main__":
    main()
