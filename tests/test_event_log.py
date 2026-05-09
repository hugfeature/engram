"""Tests for the append-only event log (Tier 1 durability primitive)."""

from __future__ import annotations

import json
import os

import pytest

from engram.event_log import (
    EventLog,
    EventLogError,
    TIER1_KINDS,
    TIER2_KINDS,
)


def _read_lines(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_append_creates_daily_file_and_assigns_monotonic_seq(tmp_path):
    log = EventLog(event_dir=str(tmp_path), engram_version="t.0")
    s1 = log.append("task.create", {"task_id": 1, "name": "a"})
    s2 = log.append("task.update", {"task_id": 1, "status": "done"})
    assert s1 == 1
    assert s2 == 2

    files = sorted(os.listdir(tmp_path))
    jsonl = [f for f in files if f.startswith("events-") and f.endswith(".jsonl")]
    assert len(jsonl) == 1, f"expected single daily file, got {jsonl}"

    events = _read_lines(os.path.join(tmp_path, jsonl[0]))
    assert [e["seq"] for e in events] == [1, 2]
    assert events[0]["kind"] == "task.create"
    assert events[0]["payload"]["name"] == "a"
    assert events[0]["engram_version"] == "t.0"
    assert events[0]["schema_version"] >= 1
    # ts must be ISO8601 UTC.
    assert events[0]["ts"].endswith("Z")


def test_unknown_kind_is_rejected(tmp_path):
    log = EventLog(event_dir=str(tmp_path))
    with pytest.raises(EventLogError):
        log.append("totally.bogus", {})


def test_seq_recovers_from_log_after_restart(tmp_path):
    log1 = EventLog(event_dir=str(tmp_path))
    log1.append("session.start", {"session_id": "s1"})
    log1.append("session.end", {"session_id": "s1", "end_type": "handoff"})
    assert log1.current_seq() == 2

    # Simulate a process restart with no in-memory state.
    log2 = EventLog(event_dir=str(tmp_path))
    assert log2.current_seq() == 2
    next_seq = log2.append("session.outcome", {"session_id": "s1", "outcome": "success"})
    assert next_seq == 3


def test_iter_events_yields_in_seq_order(tmp_path):
    log = EventLog(event_dir=str(tmp_path))
    log.append("task.create", {"task_id": 10, "name": "x"})
    log.append("memory.store", {"memory_id": 1, "content": "hi"})
    log.append("memory.delete", {"memory_id": 1})

    events = list(log.iter_events())
    assert [e["seq"] for e in events] == [1, 2, 3]
    assert [e["kind"] for e in events] == [
        "task.create",
        "memory.store",
        "memory.delete",
    ]


def test_tier1_and_tier2_kind_taxonomy_is_disjoint():
    assert TIER1_KINDS.isdisjoint(TIER2_KINDS)
    assert "task.create" in TIER1_KINDS
    assert "checkpoint.write" in TIER1_KINDS
    assert "memory.store" in TIER2_KINDS


def test_malformed_line_is_skipped_during_iter(tmp_path):
    log = EventLog(event_dir=str(tmp_path))
    log.append("task.create", {"task_id": 1, "name": "ok"})

    # Corrupt the file by appending garbage that isn't valid JSON.
    path = os.path.join(tmp_path, sorted(os.listdir(tmp_path))[-1])
    with open(path, "a", encoding="utf-8") as f:
        f.write("{this is not json}\n")

    events = list(log.iter_events())
    assert len(events) == 1
    assert events[0]["kind"] == "task.create"
