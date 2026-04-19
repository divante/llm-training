#!/usr/bin/env python3
"""Export shader/material/render-related chunks from Glyph DB.

Queries the Glyph PostgreSQL database for Unreal Engine and Godot content
related to shaders, materials, rendering, and GPU pipeline.

Outputs:
  datasets/raw/shader-pipeline/unreal_shader_chunks.jsonl
  datasets/raw/shader-pipeline/godot_shader_chunks.jsonl
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import psycopg2

from common import RAW_DIR, append_jsonl, log, write_jsonl

DATASET = "shader-pipeline"
OUTPUT_DIR = RAW_DIR / DATASET

DB_URL = os.environ.get(
    "GLYPH_DB_URL",
    "postgresql://postgres:postgres@192.168.50.117:5432/normandy",
)

# Keywords to match in qualified_name or path (case-insensitive)
SHADER_KEYWORDS = [
    "shader", "material", "render", "rhi", "vulkan", "d3d", "opengl",
    "hlsl", "glsl", "pixel", "vertex", "compute", "mesh", "texture",
    "lighting", "shadow", "postprocess", "post_process", "deferred",
    "forward", "raytracing", "ray_tracing", "lumen", "nanite",
    "canvas_item", "spatial", "fog", "particle", "sky",
    "visual_shader", "shader_material",
]


def _keyword_ilike_clause(column: str) -> str:
    """Build a SQL OR clause matching any keyword (case-insensitive)."""
    conditions = [f"{column} ILIKE '%{kw}%'" for kw in SHADER_KEYWORDS]
    return "(" + " OR ".join(conditions) + ")"


def export_unreal(conn) -> int:
    """Export Unreal shader-related chunks, grouped by parent class."""
    output_path = OUTPUT_DIR / "unreal_shader_chunks.jsonl"
    if output_path.exists() and output_path.stat().st_size > 0:
        log.info("unreal_shader_chunks.jsonl already exists, skipping")
        return sum(1 for _ in open(output_path))

    cur = conn.cursor()

    # Get class overviews
    cur.execute(f"""
        SELECT qualified_name, content, parent_name
        FROM chunks
        WHERE source_name = 'unreal'
          AND chunk_type = 'class_overview'
          AND {_keyword_ilike_clause('qualified_name')}
        ORDER BY qualified_name
    """)
    overviews = {row[0]: row[1] for row in cur.fetchall()}
    log.info("Found %d Unreal shader class overviews", len(overviews))

    # Get methods, grouped by parent class
    cur.execute(f"""
        SELECT parent_name, qualified_name, content
        FROM chunks
        WHERE source_name = 'unreal'
          AND chunk_type = 'method'
          AND {_keyword_ilike_clause('qualified_name')}
        ORDER BY parent_name, qualified_name
    """)
    methods_by_class: dict[str, list[str]] = defaultdict(list)
    for parent, qname, content in cur.fetchall():
        methods_by_class[parent or qname].append(content)
    log.info("Found methods across %d Unreal classes", len(methods_by_class))

    # Build chunks: one per class (overview + methods concatenated)
    records = []
    all_classes = set(overviews.keys()) | set(methods_by_class.keys())
    for cls in sorted(all_classes):
        parts = []
        if cls in overviews:
            parts.append(overviews[cls])
        if cls in methods_by_class:
            # Limit methods per class to keep chunks reasonable
            methods = methods_by_class[cls][:20]
            parts.extend(methods)

        text = "\n\n".join(parts)
        # Skip very short chunks
        if len(text) < 100:
            continue
        # Truncate very long chunks
        if len(text) > 8000:
            text = text[:8000]

        records.append({
            "text": text,
            "source": "unreal",
            "class": cls,
            "has_overview": cls in overviews,
            "method_count": len(methods_by_class.get(cls, [])),
        })

    write_jsonl(output_path, records)
    log.info("Exported %d Unreal shader chunks", len(records))
    return len(records)


def export_godot(conn) -> int:
    """Export Godot shader-related content from docs and API."""
    output_path = OUTPUT_DIR / "godot_shader_chunks.jsonl"
    if output_path.exists() and output_path.stat().st_size > 0:
        log.info("godot_shader_chunks.jsonl already exists, skipping")
        return sum(1 for _ in open(output_path))

    cur = conn.cursor()
    records = []

    # Godot docs (RST tutorials) — use raw_content from documents table
    cur.execute("""
        SELECT d.path, d.title, d.raw_content
        FROM documents d
        JOIN sources s ON d.source_id = s.id
        WHERE s.name = 'godot-docs'
          AND (d.path ILIKE '%shader%' OR d.path ILIKE '%render%'
               OR d.path ILIKE '%visual%' OR d.path ILIKE '%material%')
    """)
    for path, title, content in cur.fetchall():
        if not content or len(content) < 100:
            continue
        text = content[:8000] if len(content) > 8000 else content
        records.append({
            "text": text,
            "source": "godot-docs",
            "path": path,
            "title": title or path,
        })
    log.info("Found %d Godot doc pages", len(records))

    # Godot API entries (class docs for shader/material/render classes)
    api_start = len(records)
    cur.execute(f"""
        SELECT qualified_name, content, chunk_type, parent_name
        FROM chunks
        WHERE source_name = 'godot-api'
          AND {_keyword_ilike_clause('qualified_name')}
        ORDER BY qualified_name
    """)
    # Group by parent/class
    api_by_class: dict[str, list[str]] = defaultdict(list)
    for qname, content, ctype, parent in cur.fetchall():
        key = parent or qname
        api_by_class[key].append(content)

    for cls, contents in sorted(api_by_class.items()):
        text = "\n\n".join(contents[:15])
        if len(text) < 100:
            continue
        if len(text) > 8000:
            text = text[:8000]
        records.append({
            "text": text,
            "source": "godot-api",
            "class": cls,
        })
    log.info("Found %d Godot API class chunks", len(records) - api_start)

    write_jsonl(output_path, records)
    log.info("Exported %d total Godot shader chunks", len(records))
    return len(records)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Connecting to Glyph DB at %s", DB_URL.split("@")[-1])
    conn = psycopg2.connect(DB_URL)

    try:
        unreal_count = export_unreal(conn)
        godot_count = export_godot(conn)
        log.info("Done. Unreal: %d chunks, Godot: %d chunks", unreal_count, godot_count)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
