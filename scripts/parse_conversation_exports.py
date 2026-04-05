#!/usr/bin/env python3
"""
Parse conversation exports from Claude.ai and Google Gemini into ShareGPT format
for fine-tuning the personal-style dataset.

Usage:
    # Parse Claude.ai export (JSON array from data export)
    python3 scripts/parse_conversation_exports.py parse \
      --source claude \
      --input datasets/raw/personal-style/claude_export/conversations.json \
      --output datasets/raw/personal-style/claude.jsonl

    # Parse Gemini export (MyActivity.json from Google Takeout)
    python3 scripts/parse_conversation_exports.py parse \
      --source gemini \
      --input datasets/raw/personal-style/gemini_export/MyActivity.json \
      --output datasets/raw/personal-style/gemini.jsonl

    # Show stats
    python3 scripts/parse_conversation_exports.py stats \
      --input datasets/raw/personal-style/claude.jsonl

    # Merge all sources (deduplicates automatically)
    python3 scripts/parse_conversation_exports.py merge \
      --inputs datasets/raw/personal-style/claude.jsonl datasets/raw/personal-style/gemini.jsonl \
      --output datasets/raw/personal-style/all_conversations.jsonl

Output format (ShareGPT, one JSON object per line):
    {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}, ...]}
"""

import argparse
import hashlib
import json
import os
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Generator

# ─── Filters ───────────────────────────────────────────────────────────────

MIN_TURNS = 3
MIN_TOTAL_CHARS = 200
MAX_CODE_RATIO = 0.80

def _update_filters(min_turns: int, max_code_ratio: float) -> None:
    global MIN_TURNS, MAX_CODE_RATIO
    MIN_TURNS = min_turns
    MAX_CODE_RATIO = max_code_ratio


PII_PATTERNS = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b(?:sk-|pk-|key-|token-)[A-Za-z0-9_-]{20,}\b"), "[API_KEY]"),
    (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "[IP_ADDR]"),
    (re.compile(r"(?:password|passwd|secret)\s*[=:]\s*\S+", re.IGNORECASE), "[REDACTED_SECRET]"),
]


def redact_pii(text: str) -> str:
    for pattern, replacement in PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def code_block_ratio(text: str) -> float:
    if not text:
        return 0.0
    code_blocks = re.findall(r"```[\s\S]*?```", text)
    code_chars = sum(len(block) for block in code_blocks)
    return code_chars / len(text)


def passes_filters(conversation: list[dict]) -> tuple[bool, str]:
    if not conversation:
        return False, "empty"

    human_msgs = [m for m in conversation if m["from"] == "human"]
    gpt_msgs = [m for m in conversation if m["from"] == "gpt"]

    if len(human_msgs) < MIN_TURNS or len(gpt_msgs) < MIN_TURNS:
        return False, f"too_few_turns ({len(human_msgs)}h/{len(gpt_msgs)}g)"

    total_chars = sum(len(m["value"]) for m in conversation)
    if total_chars < MIN_TOTAL_CHARS:
        return False, f"too_short ({total_chars} chars)"

    assistant_text = " ".join(m["value"] for m in gpt_msgs)
    ratio = code_block_ratio(assistant_text)
    if ratio > MAX_CODE_RATIO:
        return False, f"mostly_code ({ratio:.0%})"

    return True, "pass"


# ─── HTML stripping ──────────────────────────────────────────────────────

class _HTMLStripper(HTMLParser):
    """Convert HTML to plain text/markdown."""

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._in_code = False
        self._in_pre = False
        self._list_depth = 0
        self._skip = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in ("script", "style"):
            self._skip = True
        elif tag == "code":
            self._in_code = True
            if not self._in_pre:
                self._parts.append("`")
        elif tag == "pre":
            self._in_pre = True
            self._parts.append("\n```\n")
        elif tag in ("p", "div"):
            self._parts.append("\n")
        elif tag == "br":
            self._parts.append("\n")
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag[1])
            self._parts.append("\n" + "#" * level + " ")
        elif tag in ("ul", "ol"):
            self._list_depth += 1
        elif tag == "li":
            self._parts.append("\n" + "  " * (self._list_depth - 1) + "- ")
        elif tag == "strong" or tag == "b":
            self._parts.append("**")
        elif tag == "em" or tag == "i":
            self._parts.append("*")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ("script", "style"):
            self._skip = False
        elif tag == "code":
            self._in_code = False
            if not self._in_pre:
                self._parts.append("`")
        elif tag == "pre":
            self._in_pre = False
            self._parts.append("\n```\n")
        elif tag in ("ul", "ol"):
            self._list_depth = max(0, self._list_depth - 1)
        elif tag == "strong" or tag == "b":
            self._parts.append("**")
        elif tag == "em" or tag == "i":
            self._parts.append("*")

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def handle_entityref(self, name):
        from html import unescape
        self._parts.append(unescape(f"&{name};"))

    def handle_charref(self, name):
        from html import unescape
        self._parts.append(unescape(f"&#{name};"))

    def get_text(self) -> str:
        text = "".join(self._parts)
        # Clean up excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def html_to_markdown(html: str) -> str:
    """Convert HTML to markdown-ish plain text."""
    from html import unescape
    # First unescape HTML entities
    html = unescape(html)
    stripper = _HTMLStripper()
    stripper.feed(html)
    return stripper.get_text()


# ─── Claude.ai Parser ─────────────────────────────────────────────────────

# Artifact that appears in the text field when tool_use blocks are present
_TOOL_USE_ARTIFACT = re.compile(
    r"```\nThis block is not supported on your current device yet\.\n```\s*"
)


def _extract_text_blocks(content: list) -> str:
    """Extract only text-type blocks from Claude's content array.

    Content block types: text, tool_use, tool_result, thinking, token_budget.
    We only want text blocks — tool_use/tool_result are internal,
    thinking is CoT (not user-facing), token_budget is metadata.
    """
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = block.get("text", "")
            if text.strip():
                parts.append(text.strip())
    return "\n\n".join(parts)


def parse_claude_export(input_path: str) -> Generator[list[dict], None, None]:
    """
    Parse Claude.ai data export.

    Format: JSON array of conversation objects.
    Each has chat_messages with sender (human/assistant) and content blocks.
    """
    path = Path(input_path)

    if path.is_dir():
        files = sorted(path.glob("*.json")) + sorted(path.glob("*.jsonl"))
    else:
        files = [path]

    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read().strip()

        # Handle both JSON array and JSONL
        if raw.startswith("["):
            conversations = json.loads(raw)
        else:
            conversations = []
            for line in raw.splitlines():
                if line.strip():
                    conversations.append(json.loads(line))

        for convo in conversations:
            messages = convo.get("chat_messages", [])
            if not messages:
                continue

            sharegpt_messages = []
            for msg in messages:
                sender = msg.get("sender", "")
                if sender == "human":
                    role = "human"
                elif sender == "assistant":
                    role = "gpt"
                else:
                    continue

                # Prefer content blocks (structured) over text field (has artifacts)
                content_blocks = msg.get("content", [])
                if content_blocks and isinstance(content_blocks, list):
                    text = _extract_text_blocks(content_blocks)
                else:
                    text = msg.get("text", "")

                # Clean up tool_use render artifacts from text field
                if text:
                    text = _TOOL_USE_ARTIFACT.sub("", text)

                text = text.strip()
                if not text:
                    continue

                text = redact_pii(text)
                sharegpt_messages.append({"from": role, "value": text})

            if sharegpt_messages:
                yield sharegpt_messages


# ─── Google Gemini Parser ─────────────────────────────────────────────────

# Activity entries that are NOT conversations
_GEMINI_SKIP_PREFIXES = ("Selected", "Used", "Gave", "Created")


def parse_gemini_export(input_path: str) -> Generator[list[dict], None, None]:
    """
    Parse Google Gemini MyActivity.json export.

    Format: JSON array of activity entries (NOT grouped conversations).
    Each entry with title "Prompted ..." is a user prompt.
    safeHtmlItem[0].html is the response (HTML).
    Entries with other prefixes (Selected, Used, Gave, Created) are UI actions.

    We group consecutive "Prompted" entries into multi-turn conversations.
    A new conversation starts after a gap of >30 minutes or a non-Prompted entry.
    """
    path = Path(input_path)

    if path.is_dir():
        json_files = sorted(path.rglob("*.json"))
    else:
        json_files = [path]

    for filepath in json_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            print(f"  WARN: skipping unreadable file {filepath.name}", file=sys.stderr)
            continue

        if not isinstance(data, list):
            continue

        # Filter to Prompted entries only, parse timestamps
        prompted_entries = []
        for entry in data:
            title = entry.get("title", "")
            if not title.startswith("Prompted"):
                continue

            # Extract user prompt (strip "Prompted " prefix)
            user_text = title[len("Prompted"):].strip()
            if not user_text:
                continue

            # Extract response from HTML
            html_items = entry.get("safeHtmlItem", [])
            if not html_items:
                continue
            response_html = html_items[0].get("html", "")
            if not response_html:
                continue

            response_text = html_to_markdown(response_html)
            if not response_text:
                continue

            # Parse timestamp for conversation grouping
            timestamp = entry.get("time", "")

            prompted_entries.append({
                "user": user_text,
                "response": response_text,
                "time": timestamp,
            })

        # Entries are in reverse chronological order — reverse them
        prompted_entries.reverse()

        # Group into conversations based on time gaps
        # New conversation if gap > 30 minutes between entries
        current_conversation: list[dict] = []
        last_time = None

        from datetime import datetime, timedelta

        for entry in prompted_entries:
            entry_time = None
            if entry["time"]:
                try:
                    entry_time = datetime.fromisoformat(
                        entry["time"].replace("Z", "+00:00")
                    )
                except ValueError:
                    pass

            # Check if we should start a new conversation
            start_new = False
            if not current_conversation:
                start_new = False  # first entry always goes into current
            elif last_time and entry_time:
                gap = abs((entry_time - last_time).total_seconds())
                if gap > 1800:  # 30 minutes
                    start_new = True
            elif not entry_time:
                # Can't determine gap — keep in current conversation
                pass

            if start_new and current_conversation:
                # Yield the completed conversation
                sharegpt = _gemini_group_to_sharegpt(current_conversation)
                if sharegpt:
                    yield sharegpt
                current_conversation = []

            current_conversation.append(entry)
            if entry_time:
                last_time = entry_time

        # Don't forget the last group
        if current_conversation:
            sharegpt = _gemini_group_to_sharegpt(current_conversation)
            if sharegpt:
                yield sharegpt


def _gemini_group_to_sharegpt(entries: list[dict]) -> list[dict] | None:
    """Convert a group of Gemini prompted entries to ShareGPT format."""
    messages = []
    for entry in entries:
        user_text = redact_pii(entry["user"])
        response_text = redact_pii(entry["response"])
        messages.append({"from": "human", "value": user_text})
        messages.append({"from": "gpt", "value": response_text})
    return messages if messages else None


# ─── Main ──────────────────────────────────────────────────────────────────

def process_source(source: str, input_path: str) -> list[list[dict]]:
    parsers = {
        "claude": parse_claude_export,
        "gemini": parse_gemini_export,
    }

    if source not in parsers:
        print(f"ERROR: Unknown source '{source}'. Supported: {list(parsers.keys())}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing {source} export from: {input_path}")

    all_conversations = []
    skipped = {"empty": 0, "too_few_turns": 0, "too_short": 0, "mostly_code": 0, "other": 0}

    for messages in parsers[source](input_path):
        passed, reason = passes_filters(messages)
        if passed:
            all_conversations.append(messages)
        else:
            bucket = reason.split(" ")[0]
            if bucket.startswith("too_few"):
                skipped["too_few_turns"] += 1
            elif bucket.startswith("too_short"):
                skipped["too_short"] += 1
            elif bucket.startswith("mostly_code"):
                skipped["mostly_code"] += 1
            else:
                skipped["other"] += 1

    print(f"  Kept: {len(all_conversations)}")
    print(f"  Skipped: {json.dumps(skipped)}")

    return all_conversations


def print_stats(conversations: list[list[dict]]):
    if not conversations:
        print("\n--- Dataset Stats ---")
        print("  No conversations.")
        return

    total_msgs = sum(len(c) for c in conversations)
    human_msgs = sum(1 for c in conversations for m in c if m["from"] == "human")
    gpt_msgs = sum(1 for c in conversations for m in c if m["from"] == "gpt")
    avg_turns = total_msgs / len(conversations)
    total_chars = sum(len(m["value"]) for c in conversations for m in c)
    avg_chars_per_msg = total_chars / total_msgs if total_msgs else 0

    lengths = sorted(len(c) for c in conversations)

    print(f"\n--- Dataset Stats ---")
    print(f"  Conversations: {len(conversations)}")
    print(f"  Total messages: {total_msgs} ({human_msgs} human, {gpt_msgs} gpt)")
    print(f"  Avg turns per conversation: {avg_turns:.1f}")
    print(f"  Avg chars per message: {avg_chars_per_msg:.0f}")
    print(f"  Total chars: {total_chars:,}")
    print(f"  Conversation length distribution:")
    print(f"    Min: {lengths[0]}, Max: {lengths[-1]}")
    p25 = lengths[len(lengths) // 4]
    p50 = lengths[len(lengths) // 2]
    p75 = lengths[3 * len(lengths) // 4]
    print(f"    P25: {p25}, P50: {p50}, P75: {p75}")


def write_output(conversations: list[list[dict]], output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for convo in conversations:
            f.write(json.dumps({"conversations": convo}, ensure_ascii=False) + "\n")
    print(f"\nWritten {len(conversations)} conversations to {output_path}")


def merge_files(input_files: list[str], output_path: str):
    seen: set[str] = set()
    all_lines: list[str] = []

    for filepath in input_files:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                content_hash = hashlib.md5(line.encode()).hexdigest()
                if content_hash not in seen:
                    seen.add(content_hash)
                    all_lines.append(line)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in all_lines:
            f.write(line + "\n")

    print(f"Merged {len(all_lines)} unique conversations from {len(input_files)} files -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse Claude/Gemini conversation exports into ShareGPT format"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Parse command
    parse_cmd = subparsers.add_parser("parse", help="Parse a single export source")
    parse_cmd.add_argument("--source", required=True, choices=["claude", "gemini"])
    parse_cmd.add_argument("--input", required=True, help="Path to export file or directory")
    parse_cmd.add_argument("--output", help="Output JSONL path (omit for stats-only)")
    parse_cmd.add_argument("--min-turns", type=int, default=MIN_TURNS)
    parse_cmd.add_argument("--max-code-ratio", type=float, default=MAX_CODE_RATIO)

    # Merge command
    merge_cmd = subparsers.add_parser("merge", help="Merge multiple parsed JSONL files")
    merge_cmd.add_argument("--inputs", nargs="+", required=True)
    merge_cmd.add_argument("--output", required=True)

    # Stats command
    stats_cmd = subparsers.add_parser("stats", help="Show stats for a JSONL file")
    stats_cmd.add_argument("--input", required=True)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "parse":
        _update_filters(args.min_turns, args.max_code_ratio)
        conversations = process_source(args.source, args.input)
        print_stats(conversations)

        if args.output:
            write_output(conversations, args.output)

    elif args.command == "merge":
        merge_files(args.inputs, args.output)

    elif args.command == "stats":
        conversations = []
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    conversations.append(data.get("conversations", []))
        print_stats(conversations)


if __name__ == "__main__":
    main()
