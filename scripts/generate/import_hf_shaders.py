#!/usr/bin/env python3
"""Import shader datasets from HuggingFace parquet files.

Downloads seanmemery/shader_dataset (45K+ GLSL shaders with descriptions)
and converts to chunks for the shader-pipeline dataset.

Output: datasets/raw/shader-pipeline/hf_shader_chunks.jsonl
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

from common import RAW_DIR, log, write_jsonl

DATASET = "shader-pipeline"
OUTPUT_DIR = RAW_DIR / DATASET

HF_DATASETS = {
    "seanmemery/shader_dataset": {
        "url": "https://huggingface.co/api/datasets/seanmemery/shader_dataset/parquet/default/train/0.parquet",
        "code_col": "code",
        "desc_col": "description",
    },
}


def download_parquet(url: str, dest: Path) -> bool:
    """Download a parquet file if not already cached."""
    if dest.exists():
        log.info("Using cached %s", dest)
        return True
    log.info("Downloading %s ...", url)
    result = subprocess.run(
        ["curl", "-sL", url, "-o", str(dest)],
        capture_output=True, text=True, timeout=300,
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Import HuggingFace shader datasets")
    parser.add_argument("--limit", type=int, default=10000, help="Max shaders to import")
    parser.add_argument("--min-lines", type=int, default=10, help="Min code lines")
    parser.add_argument("--max-lines", type=int, default=500, help="Max code lines")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "hf_shader_chunks.jsonl"

    if output_path.exists() and output_path.stat().st_size > 0:
        line_count = sum(1 for _ in open(output_path))
        log.info("hf_shader_chunks.jsonl already exists (%d chunks), skipping", line_count)
        return

    cache_dir = Path(tempfile.gettempdir()) / "shader_hf"
    cache_dir.mkdir(exist_ok=True)

    all_chunks = []

    for dataset_id, config in HF_DATASETS.items():
        slug = dataset_id.replace("/", "_")
        parquet_path = cache_dir / f"{slug}.parquet"

        if not download_parquet(config["url"], parquet_path):
            log.warning("Failed to download %s", dataset_id)
            continue

        table = pq.read_table(parquet_path)
        log.info("Loaded %d rows from %s", table.num_rows, dataset_id)

        code_col = config["code_col"]
        desc_col = config["desc_col"]

        for i in range(table.num_rows):
            if len(all_chunks) >= args.limit:
                break

            code = table.column(code_col)[i].as_py()
            description = table.column(desc_col)[i].as_py()

            if not code:
                continue

            lines = code.strip().splitlines()
            if len(lines) < args.min_lines or len(lines) > args.max_lines:
                continue

            # Build chunk text
            parts = []
            if description:
                parts.append(f"// Description: {description[:500]}")
                parts.append("// Language: GLSL (Fragment Shader)")
                parts.append("")
            parts.append(code)

            text = "\n".join(parts)
            if len(text) > 8000:
                text = text[:8000]

            all_chunks.append({
                "text": text,
                "source": dataset_id,
                "index": i,
                "line_count": len(lines),
            })

    write_jsonl(output_path, all_chunks)
    log.info("Wrote %d shader chunks to %s", len(all_chunks), output_path)


if __name__ == "__main__":
    main()
