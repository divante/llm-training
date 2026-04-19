#!/usr/bin/env python3
"""Extract shader examples from The Book of Shaders.

Clones (or uses existing) the Book of Shaders repository and extracts
chapter content with inline GLSL code examples.

Output: datasets/raw/shader-pipeline/bookofshaders_chunks.jsonl
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from common import RAW_DIR, log, write_jsonl

DATASET = "shader-pipeline"
OUTPUT_DIR = RAW_DIR / DATASET
REPO_URL = "https://github.com/patriciogonzalezvivo/thebookofshaders.git"

# Regex to find GLSL code blocks in markdown
GLSL_BLOCK = re.compile(
    r'```(?:glsl|c|cpp)?\s*\n(.*?)```',
    re.DOTALL,
)


def find_chapters(repo_path: Path) -> list[Path]:
    """Find chapter markdown files in the repo."""
    chapters = []
    for d in sorted(repo_path.iterdir()):
        if not d.is_dir() or not d.name[:2].isdigit():
            continue
        # Each chapter dir has a README.md or similar
        for md in sorted(d.glob("*.md")):
            chapters.append(md)
    # Also check top-level .md files
    for md in sorted(repo_path.glob("*.md")):
        if md.name.lower() not in ("readme.md", "license.md", "contributing.md"):
            chapters.append(md)
    return chapters


def extract_chunks(chapters: list[Path], repo_path: Path) -> list[dict]:
    """Extract content chunks from chapter files.

    Each chunk is a chapter section that contains GLSL code.
    """
    chunks = []

    for chapter_path in chapters:
        try:
            content = chapter_path.read_text(errors="replace")
        except Exception:
            continue

        # Skip files without shader code
        if not GLSL_BLOCK.search(content) and "shader" not in content.lower():
            continue

        rel_path = str(chapter_path.relative_to(repo_path))
        chapter_name = chapter_path.parent.name if chapter_path.name == "README.md" else chapter_path.stem

        # Split by headings to create smaller chunks
        sections = re.split(r'\n(?=##?\s)', content)

        for section in sections:
            # Only keep sections with code or significant shader content
            has_code = bool(GLSL_BLOCK.search(section))
            has_shader_content = any(
                kw in section.lower()
                for kw in ["uniform", "varying", "gl_frag", "vec2", "vec3", "vec4",
                           "float", "sampler", "texture", "fragment", "vertex"]
            )

            if not has_code and not has_shader_content:
                continue

            text = section.strip()
            if len(text) < 100:
                continue
            if len(text) > 8000:
                text = text[:8000]

            # Extract title from first heading
            title_match = re.match(r'^##?\s+(.+)', text)
            title = title_match.group(1) if title_match else chapter_name

            chunks.append({
                "text": text,
                "source": "book-of-shaders",
                "chapter": chapter_name,
                "title": title,
                "path": rel_path,
            })

    return chunks


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "bookofshaders_chunks.jsonl"

    if output_path.exists() and output_path.stat().st_size > 0:
        line_count = sum(1 for _ in open(output_path))
        log.info("bookofshaders_chunks.jsonl already exists (%d chunks), skipping", line_count)
        return

    # Check for existing local clone
    local_candidates = [
        Path.home() / "git" / "thebookofshaders",
        Path.home() / "git" / "book-of-shaders",
    ]
    repo_path = None
    for candidate in local_candidates:
        if candidate.exists() and (candidate / ".git").exists():
            repo_path = candidate
            log.info("Using existing clone at %s", repo_path)
            break

    if repo_path is None:
        # Clone to temp directory
        tmpdir = tempfile.mkdtemp(prefix="bookofshaders_")
        repo_path = Path(tmpdir) / "repo"
        log.info("Cloning %s ...", REPO_URL)
        result = subprocess.run(
            ["git", "clone", "--depth=1", REPO_URL, str(repo_path)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            log.error("Failed to clone: %s", result.stderr[:300])
            return

    chapters = find_chapters(repo_path)
    log.info("Found %d chapter files", len(chapters))

    chunks = extract_chunks(chapters, repo_path)
    log.info("Extracted %d chunks with shader content", len(chunks))

    write_jsonl(output_path, chunks)
    log.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
