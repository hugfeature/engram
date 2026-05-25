#!/usr/bin/env python3
"""Collect all local Claude Code sessions into the observation corpus.

Scans ~/.claude/projects for JSONL transcripts, parses basic metadata,
and registers them in the corpus with auto-classification.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from engram.observation.transcript import Transcript
from engram.observation.events import extract_events, EventType
from engram.observation.corpus import Corpus, SessionMeta, SessionKind


CLAUDE_DIR = Path.home() / ".claude"
CODEFUSE_DIR = Path.home() / ".codefuse"
CORPUS_DIR = Path(__file__).parent.parent / "data" / "observation_corpus"

# Skip these path patterns
SKIP_PATTERNS = ["subagents", "plugins", "cache", "backup", "fixtures"]


def find_transcripts() -> list[Path]:
    """Find all main session JSONL files from both Claude Code and Codefuse."""
    results = []
    search_dirs = [
        CLAUDE_DIR / "projects",
        CLAUDE_DIR / "sessions",
        CODEFUSE_DIR / "projects",
        CODEFUSE_DIR / "engine" / "cc" / "projects",
        CODEFUSE_DIR / "fuse" / "engine" / "cc" / "projects",
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for jsonl_file in search_dir.rglob("*.jsonl"):
            path_str = str(jsonl_file)
            if any(skip in path_str for skip in SKIP_PATTERNS):
                continue
            # Skip tiny files (< 500 bytes = likely empty/aborted)
            if jsonl_file.stat().st_size < 500:
                continue
            results.append(jsonl_file)

    return sorted(results, key=lambda p: p.stat().st_mtime, reverse=True)


def extract_session_info(path: Path) -> dict:
    """Extract basic session metadata from a transcript file."""
    info = {
        "session_id": path.stem,
        "file_size": path.stat().st_size,
        "modified": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
        "project": path.parent.name,
        "path": str(path),
    }

    # Count turns and extract first user message as description
    turn_count = 0
    first_user_msg = ""
    tool_use_count = 0
    error_count = 0

    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                top_type = data.get("type", "")

                if top_type in ("user", "assistant"):
                    turn_count += 1

                if top_type == "user" and not first_user_msg:
                    msg = data.get("message", {})
                    content = msg.get("content", "") if isinstance(msg, dict) else ""
                    if isinstance(content, str) and content.strip():
                        first_user_msg = content.strip()[:100]
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                first_user_msg = block.get("text", "")[:100]
                                break

                if top_type == "assistant":
                    msg = data.get("message", {})
                    if isinstance(msg, dict):
                        content = msg.get("content", [])
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict):
                                    if block.get("type") == "tool_use":
                                        tool_use_count += 1
                                    if block.get("type") == "tool_result" and block.get("is_error"):
                                        error_count += 1

    except Exception as exc:
        info["parse_error"] = str(exc)

    info["turn_count"] = turn_count
    info["first_user_msg"] = first_user_msg
    info["tool_use_count"] = tool_use_count
    info["error_count"] = error_count

    return info


def classify_complexity(info: dict) -> str:
    """Heuristic complexity classification."""
    turns = info.get("turn_count", 0)
    tools = info.get("tool_use_count", 0)

    if turns > 100 or tools > 50:
        return "high"
    if turns > 30 or tools > 15:
        return "medium"
    return "low"


def main():
    print("=" * 60)
    print("Claude Code Session Collector")
    print("=" * 60)

    transcripts = find_transcripts()
    print(f"\nFound {len(transcripts)} session transcripts\n")

    # Initialize corpus
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    corpus = Corpus(CORPUS_DIR)
    existing_ids = {s.session_id for s in corpus.list_all()}

    registered = 0
    skipped = 0

    for path in transcripts:
        session_id = path.stem

        if session_id in existing_ids:
            skipped += 1
            continue

        info = extract_session_info(path)
        complexity = classify_complexity(info)

        # Derive project context from path
        project_dir = path.parent.name
        tags = []
        if "engram" in project_dir or "engram" in str(path):
            tags.append("engram")
        if "skill" in project_dir:
            tags.append("skill")

        meta = SessionMeta(
            session_id=session_id,
            kind=SessionKind.NATURALISTIC,  # Default: all real sessions are naturalistic
            transcript_path=str(path),
            description=info.get("first_user_msg", ""),
            tags=tags,
            agent="claude-code",
            task_complexity=complexity,
            session_duration_turns=info.get("turn_count", 0),
        )

        try:
            corpus.register(meta)
            registered += 1
            size_kb = info["file_size"] / 1024
            print(f"  ✓ {session_id[:12]}... | {info['turn_count']:>4} turns | "
                  f"{info['tool_use_count']:>3} tools | {size_kb:>7.1f}KB | "
                  f"{complexity:>6} | {info.get('first_user_msg', '')[:40]}")
        except ValueError as exc:
            print(f"  ✗ {session_id[:12]}... | {exc}")
            skipped += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {registered} registered, {skipped} skipped")
    print(f"Corpus total: {corpus.count()} sessions")
    print(f"\nCorpus summary:")
    summary = corpus.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\nCorpus stored at: {CORPUS_DIR}")


if __name__ == "__main__":
    main()
