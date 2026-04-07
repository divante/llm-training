"""Shared utilities for dataset generation scripts.

Provides: LLM client, JSONL I/O, resumability, progress logging.
All generation runs through a local llama-server (OpenAI-compatible API).
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(os.environ.get("LLM_TRAINING_ROOT", Path(__file__).resolve().parent.parent.parent))
RAW_DIR = ROOT / "datasets" / "raw"
LOGS_DIR = ROOT / "logs"
GENERATION_LOG = LOGS_DIR / "dataset_generation.jsonl"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("dataset-gen")

# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11437/v1")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "qwen3.5:35b-a3b")
DEFAULT_API_KEY = os.environ.get("LLM_API_KEY", "unused")


def get_client() -> OpenAI:
    """Create an OpenAI-compatible client from LLM_BASE_URL / LLM_API_KEY env vars."""
    return OpenAI(
        base_url=DEFAULT_BASE_URL,
        api_key=DEFAULT_API_KEY,
    )


# ---------------------------------------------------------------------------
# Generation with exponential backoff
# ---------------------------------------------------------------------------

_MAX_RETRIES = 6          # 1s → 2s → 4s → 8s → 16s → 32s (max ~63s total wait)
_BASE_DELAY = 1.0         # seconds
_RETRYABLE_STATUS_CODES = {429, 502, 503, 504}


def generate(
    client: OpenAI,
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    model: str | None = None,
    retries: int = _MAX_RETRIES,
) -> str | None:
    """Single-shot generation with exponential backoff for rate limits."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model or DEFAULT_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_err = e
            status = getattr(e, "status_code", None)
            if status and status not in _RETRYABLE_STATUS_CODES:
                raise  # non-retryable error (auth, bad request, etc.)
            if attempt < retries:
                delay = _BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                log.warning("Retryable error (attempt %d/%d), sleeping %.1fs: %s",
                            attempt + 1, retries, delay, e)
                time.sleep(delay)
            else:
                log.error("All %d retries exhausted: %s", retries, e)
                raise last_err


def generate_batch(
    client: OpenAI,
    prompts: list[str],
    *,
    system: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    log_every: int = 100,
) -> list[str]:
    """Sequential batch generation. Local model — no rate limits, no cost."""
    results = []
    for i, prompt in enumerate(prompts):
        results.append(
            generate(client, prompt, system=system, max_tokens=max_tokens, temperature=temperature)
        )
        if (i + 1) % log_every == 0:
            log.info("Generated %d/%d", i + 1, len(prompts))
    return results


# ---------------------------------------------------------------------------
# JSON Repair
# ---------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences wrapping JSON."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


_VALID_ESCAPES = frozenset('"\\/bfnrtu')


def repair_json(raw: str | None) -> list[dict[str, Any]] | None:
    """Best-effort repair of malformed JSON arrays from LLM output.

    Handles:
    - Markdown code fences
    - Invalid escape sequences (``\\N``, ``\\s``, etc.)
    - Unescaped double quotes inside JSON string values (e.g. C++ #include)
    - Truncated responses (salvages complete objects before the cutoff)

    Returns the parsed list on success, or ``None`` if unrecoverable.
    """
    if not raw:
        return None

    text = _strip_code_fences(raw)

    # ── Fix invalid escape sequences ──
    # JSON only allows: \" \\ \/ \b \f \n \r \t \uXXXX
    # LLMs produce things like \N \s \c — double-escape them.
    fixed_escapes: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == '\\' and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt not in _VALID_ESCAPES:
                fixed_escapes.append('\\\\')  # turn \ into \\
            else:
                fixed_escapes.append('\\')
            fixed_escapes.append(nxt)
            i += 2
        else:
            fixed_escapes.append(text[i])
            i += 1
    text = "".join(fixed_escapes)

    # ── Fast path: try clean parse ──
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # ── Character-level quote repair ──
    # Walk through the text, track whether we're inside a JSON string,
    # and escape interior double quotes that aren't structural.
    repaired: list[str] = []
    i = 0
    in_string = False

    while i < len(text):
        c = text[i]
        if not in_string:
            repaired.append(c)
            if c == '"':
                in_string = True
        else:
            if c == '\\' and i + 1 < len(text):
                repaired.append(c)
                repaired.append(text[i + 1])
                i += 2
                continue
            elif c == '"':
                # Structural close-quote?  Look at what follows (skip whitespace).
                rest = text[i + 1:].lstrip()
                if not rest or rest[0] in ',}]:':
                    repaired.append(c)
                    in_string = False
                elif rest[0] == '"':
                    # end-of-value, start-of-key
                    repaired.append(c)
                    in_string = False
                else:
                    repaired.append('\\"')  # interior quote → escape it
            elif c == '\n':
                repaired.append('\\n')
            elif c == '\r':
                repaired.append('\\r')
            elif c == '\t':
                repaired.append('\\t')
            else:
                repaired.append(c)
        i += 1

    repaired_text = "".join(repaired)
    try:
        data = json.loads(repaired_text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # ── Salvage truncated arrays ──
    # Find the last complete top-level object and close the array there.
    last_complete = -1
    brace_depth = 0
    in_str = False
    for i, c in enumerate(repaired_text):
        if in_str:
            if c == '\\' and i + 1 < len(repaired_text):
                continue
            if c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == '{':
            brace_depth += 1
        elif c == '}':
            brace_depth -= 1
            if brace_depth == 0:
                last_complete = i

    if last_complete > 0:
        candidate = repaired_text[:last_complete + 1].rstrip().rstrip(',')
        if not candidate.lstrip().startswith('['):
            candidate = '[' + candidate
        candidate = candidate.rstrip().rstrip(',') + ']'
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all records from a JSONL file."""
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append a single record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records to a JSONL file (overwrites)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------


def count_lines(path: Path) -> int:
    """Count lines in a file (for resume checkpointing)."""
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


# ---------------------------------------------------------------------------
# Generation Progress Logging
# ---------------------------------------------------------------------------


def log_progress(
    dataset: str,
    phase: str,
    status: str,
    *,
    records: int | None = None,
    progress: str | None = None,
    error: str | None = None,
) -> None:
    """Append a progress entry to dataset_generation.jsonl."""
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "phase": phase,
        "status": status,
    }
    if records is not None:
        entry["records"] = records
    if progress:
        entry["progress"] = progress
    if error:
        entry["error"] = error
    append_jsonl(GENERATION_LOG, entry)
    log.info("[%s/%s] %s%s", dataset, phase, status,
             f" ({progress})" if progress else f" ({records} records)" if records is not None else "")
