"""Shared utilities for dataset generation scripts.

Provides: LLM client, JSONL I/O, resumability, progress logging.
All generation runs through a local llama-server (OpenAI-compatible API).
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
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
# LLM Client — provider-aware via LLM_PROVIDER env var
# ---------------------------------------------------------------------------

PROVIDER = os.environ.get("LLM_PROVIDER", "local")

DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11437/v1")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "qwen3.6:35b-a3b")
DEFAULT_API_KEY = os.environ.get("LLM_API_KEY", "unused")

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

GEMINI_BASE_URL = os.environ.get("GEMINI_BASE_URL", "http://localhost:8317/v1")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "sk-IeEobuE2nRPtWxHK8uxHxYHGfksfeujbFWwgvkhpDpQN5")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def get_client():
    """Return the right LLM client based on LLM_PROVIDER env var."""
    if PROVIDER == "anthropic":
        return anthropic.Anthropic()
    if PROVIDER == "gemini":
        return OpenAI(base_url=GEMINI_BASE_URL, api_key=GEMINI_API_KEY)
    return OpenAI(base_url=DEFAULT_BASE_URL, api_key=DEFAULT_API_KEY)


# ---------------------------------------------------------------------------
# Generation with exponential backoff
# ---------------------------------------------------------------------------

_MAX_RETRIES = 6          # 1s → 2s → 4s → 8s → 16s → 32s (max ~63s total wait)
_BASE_DELAY = 1.0         # seconds
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def generate(
    client,
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int = 16384,
    temperature: float = 0.7,
    model: str | None = None,
    retries: int = _MAX_RETRIES,
) -> str | None:
    """Single-shot generation. Dispatches to OpenAI or Anthropic based on client type."""
    if isinstance(client, anthropic.Anthropic):
        return _generate_anthropic(
            client, prompt, system=system, max_tokens=max_tokens,
            temperature=temperature, model=model, retries=retries,
        )
    return _generate_openai(
        client, prompt, system=system, max_tokens=max_tokens,
        temperature=temperature, model=model, retries=retries,
    )


def _generate_openai(
    client: OpenAI,
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int = 16384,
    temperature: float = 0.7,
    model: str | None = None,
    retries: int = _MAX_RETRIES,
) -> str | None:
    resolved_model = model or (GEMINI_MODEL if PROVIDER == "gemini" else DEFAULT_MODEL)
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    if "qwen3" in resolved_model.lower():
        prompt = "/no_think\n" + prompt
    messages.append({"role": "user", "content": prompt})

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=resolved_model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_err = e
            status = getattr(e, "status_code", None)
            if status and status not in _RETRYABLE_STATUS_CODES:
                raise
            if attempt < retries:
                delay = _BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                log.warning("Retryable error (attempt %d/%d), sleeping %.1fs: %s",
                            attempt + 1, retries, delay, e)
                time.sleep(delay)
            else:
                log.error("All %d retries exhausted: %s", retries, e)
                raise last_err


def _generate_anthropic(
    client: anthropic.Anthropic,
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int = 16384,
    temperature: float = 0.7,
    model: str | None = None,
    retries: int = _MAX_RETRIES,
) -> str | None:
    resolved_model = model or DEFAULT_ANTHROPIC_MODEL
    system_messages = []
    if system:
        system_messages = [{
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }]

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.messages.create(
                model=resolved_model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_messages,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except anthropic.RateLimitError as e:
            last_err = e
            if attempt < retries:
                delay = _BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                log.warning("Rate limited (attempt %d/%d), sleeping %.1fs",
                            attempt + 1, retries, delay)
                time.sleep(delay)
            else:
                log.error("All %d retries exhausted: %s", retries, e)
                raise
        except anthropic.APIStatusError as e:
            last_err = e
            if e.status_code in _RETRYABLE_STATUS_CODES and attempt < retries:
                delay = _BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                log.warning("Retryable error (attempt %d/%d), sleeping %.1fs: %s",
                            attempt + 1, retries, delay, e)
                time.sleep(delay)
            else:
                raise
        except Exception as e:
            raise


def generate_batch(
    client,
    prompts: list[str],
    *,
    system: str | None = None,
    max_tokens: int = 16384,
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


def _fix_escape_sequences(text: str) -> str:
    """Fix invalid escape sequences in JSON text.

    JSON only allows: \" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX
    LLMs produce things like \\N \\s \\c — double-escape them.
    """
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == '\\' and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt not in _VALID_ESCAPES:
                out.append('\\\\')  # turn \ into \\
            else:
                out.append('\\')
            out.append(nxt)
            i += 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _repair_quotes_and_control_chars(text: str) -> str:
    """Walk through JSON text, escape interior quotes and raw control chars.

    Uses a lookahead heuristic: a quote is "structural" (closes a JSON string)
    if followed by whitespace then a JSON structural character. The key insight
    is that after a closing quote the next non-whitespace must be one of:
        , } ] :  "  (comma, brace-close, bracket-close, colon, or next string)
    Anything else means the quote is interior content and should be escaped.
    """
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
                # Structural close-quote? Look at what follows (skip whitespace).
                rest = text[i + 1:].lstrip()
                if not rest or rest[0] in ',}]:':
                    repaired.append(c)
                    in_string = False
                elif rest[0] == '"':
                    # Could be end-of-value→start-of-key, OR an interior quote
                    # followed by more quoted text. Disambiguate: if the next
                    # quote is followed by a colon (after optional whitespace),
                    # this is a key boundary. Otherwise it's interior.
                    after_next_quote = rest[1:].lstrip()
                    if after_next_quote and after_next_quote[0] == ':':
                        # Looks like "value" "key": — but missing comma.
                        # Close string, let the parser handle the missing comma.
                        repaired.append(c)
                        in_string = False
                    else:
                        # Check deeper: scan for the pattern ": after the next quote
                        # to decide if this is a field boundary
                        next_quote_pos = rest.find('"', 1)
                        if next_quote_pos > 0:
                            between = rest[1:next_quote_pos].strip()
                            after = rest[next_quote_pos + 1:].lstrip()
                            if not between and after and after[0] == ':':
                                repaired.append(c)
                                in_string = False
                            else:
                                repaired.append('\\"')
                        else:
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

    return "".join(repaired)


def _salvage_truncated(text: str) -> list[dict[str, Any]] | None:
    """Salvage complete objects from a truncated JSON array."""
    last_complete = -1
    brace_depth = 0
    in_str = False
    i = 0
    while i < len(text):
        c = text[i]
        if in_str:
            if c == '\\' and i + 1 < len(text):
                i += 2
                continue
            if c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == '{':
                brace_depth += 1
            elif c == '}':
                brace_depth -= 1
                if brace_depth == 0:
                    last_complete = i
        i += 1

    if last_complete > 0:
        candidate = text[:last_complete + 1].rstrip().rstrip(',')
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


def _parse_result(text: str) -> list[dict[str, Any]] | dict[str, Any] | None:
    """Try to parse JSON text as a list or single object."""
    try:
        data = json.loads(text)
        if isinstance(data, (list, dict)):
            return data
    except json.JSONDecodeError:
        pass
    return None


def repair_json(raw: str | None) -> list[dict[str, Any]] | dict[str, Any] | None:
    """Best-effort repair of malformed JSON from LLM output.

    Handles:
    - Markdown code fences
    - Invalid escape sequences (``\\N``, ``\\s``, etc.)
    - Unescaped double quotes inside JSON string values (e.g. C++ #include)
    - Raw newlines/tabs inside JSON strings
    - Truncated responses (salvages complete objects before the cutoff)

    Returns the parsed result (list or dict) on success, or ``None`` if
    unrecoverable.
    """
    if not raw:
        return None

    text = _strip_code_fences(raw)

    # ── Fix invalid escape sequences ──
    text = _fix_escape_sequences(text)

    # ── Fast path: try clean parse ──
    result = _parse_result(text)
    if result is not None:
        return result

    # ── Character-level quote + control char repair ──
    repaired_text = _repair_quotes_and_control_chars(text)
    result = _parse_result(repaired_text)
    if result is not None:
        return result

    # ── Salvage truncated arrays ──
    result = _salvage_truncated(repaired_text)
    if result is not None:
        return result

    # ── Structural extraction (last resort for code-heavy content) ──
    result = _extract_conversations_structural(text)
    if result is not None:
        return result

    return None


_TURN_MARKER = re.compile(
    r'"from"\s*:\s*"(human|gpt)"\s*,\s*"value"\s*:\s*"'
)
_TURN_CLOSE_MID = re.compile(r'"\s*}\s*,\s*$')
_TURN_CLOSE_END = re.compile(r'"\s*}\s*\]\s*}?\s*$')


def _unescape_json_content(s: str) -> str:
    """Unescape JSON string escape sequences in extracted content."""
    s = s.replace('\\"', '"')
    s = s.replace('\\n', '\n')
    s = s.replace('\\t', '\t')
    s = s.replace('\\/', '/')
    s = s.replace('\\\\', '\\')
    return s


def _extract_conversations_structural(text: str) -> dict | None:
    """Extract conversations by finding turn boundary markers.

    Falls back to structural parsing when JSON repair fails due to
    unescaped code blocks inside string values. The ``"from": "human/gpt"``
    pattern reliably separates turns even when the content is messy.
    """
    markers = list(_TURN_MARKER.finditer(text))
    if len(markers) < 2:
        return None

    turns = []
    for i, match in enumerate(markers):
        role = match.group(1)
        value_start = match.end()

        if i + 1 < len(markers):
            region = text[value_start:markers[i + 1].start()]
            m = _TURN_CLOSE_MID.search(region)
            if m:
                content = region[:m.start()]
            else:
                content = region.rstrip().rstrip('{').rstrip().rstrip(',').rstrip().rstrip('}').rstrip()
                if content.endswith('"'):
                    content = content[:-1]
        else:
            region = text[value_start:]
            m = _TURN_CLOSE_END.search(region)
            if m:
                content = region[:m.start()]
            else:
                content = region.rstrip().rstrip('}').rstrip().rstrip(']').rstrip().rstrip('}').rstrip()
                if content.endswith('"'):
                    content = content[:-1]

        turns.append({"from": role, "value": _unescape_json_content(content)})

    if len(turns) >= 4:
        return {"conversations": turns}
    return None


# ---------------------------------------------------------------------------
# Provider-aware output routing & cross-provider resumability
# ---------------------------------------------------------------------------


def get_output_path(output_dir: Path, worker_id: int = 0) -> Path:
    """Return per-provider/worker output file path."""
    if PROVIDER == "local":
        tag = f"local_{worker_id}"
    else:
        tag = PROVIDER
    return output_dir / f"raw_responses_{tag}.jsonl"


def load_all_processed_keys(
    output_dir: Path,
    key_fields: tuple[str, ...],
) -> set[tuple]:
    """Scan all raw_responses_*.jsonl in output_dir, extract composite keys."""
    processed: set[tuple] = set()
    for resp_file in output_dir.glob("raw_responses_*.jsonl"):
        for r in read_jsonl(resp_file):
            try:
                key = tuple(r[f] for f in key_fields)
                processed.add(key)
            except KeyError:
                continue
    log.info("Skipping %d items already processed across all providers", len(processed))
    return processed


def add_worker_args(parser) -> None:
    """Add standard worker/target args to any generation script."""
    parser.add_argument("--target", type=int, default=7500, help="Target record count")
    parser.add_argument("--worker-id", type=int, default=0, help="ID of this worker (0-indexed)")
    parser.add_argument("--num-workers", type=int, default=1, help="Total number of parallel workers")


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
