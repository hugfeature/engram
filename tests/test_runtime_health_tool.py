"""Tests for P1-4: get_runtime_health MCP tool.

Verifies:
- Tool is registered in TOOL_SCHEMAS / TOOL_HANDLERS / ARG_MAPPING
- Handler returns ok=True even when DB is fine
- ``advice`` field is populated with actionable hints when degraded/stale/residue
- Schema declares no required input args (zero-arg tool)
"""

from __future__ import annotations

import os

import pytest

from engram.handlers import TOOL_HANDLERS, ARG_MAPPING, handle_get_runtime_health
from engram.tools import TOOL_SCHEMAS


def _get_tool_schema(name: str):
    for t in TOOL_SCHEMAS:
        if t.name == name:
            return t
    return None


def test_tool_is_registered_everywhere():
    assert "get_runtime_health" in TOOL_HANDLERS
    assert "get_runtime_health" in ARG_MAPPING
    assert ARG_MAPPING["get_runtime_health"] == {}

    schema = _get_tool_schema("get_runtime_health")
    assert schema is not None, "get_runtime_health missing from TOOL_SCHEMAS"
    assert schema.inputSchema["properties"] == {}
    # Zero-arg tool: no `required` list, no extra properties.
    assert schema.inputSchema.get("additionalProperties") is False
    assert "required" not in schema.inputSchema


def test_handler_returns_healthy_shape(tmp_path, monkeypatch):
    """Handler must return ok=True and the doctor() payload merged in."""
    # Force doctor() to return a known-clean snapshot of the world so the
    # test doesn't depend on the operator's actual ~/.engram state.
    fake = {
        "db_path": "/tmp/x.duckdb",
        "db_exists": True,
        "event_dir": "/tmp/events",
        "residue_files": [],
        "backups": {"dir": "/tmp/backups", "live_count": 2, "retain": 10,
                    "archive_count": 0, "live_recent": []},
        "snapshots": {"dir": "/tmp/snap", "count": 0, "latest_seq": 0,
                      "latest_path": None, "latest_size_bytes": 0},
        "event_kinds": {"task.create": 1},
        "event_max_seq": 1,
        "meta": {"engram_version": "0.11.0"},
        "counts": {"memories": 0, "tasks": 1, "checkpoints": 0},
        "readonly": False,
        "embedding_stale": False,
    }
    monkeypatch.setattr("engram.recover.doctor", lambda: fake)

    result = handle_get_runtime_health(db=None, graph=None)
    assert result["ok"] is True
    assert result["advice"] == []
    # Original doctor fields are merged in at the top level.
    assert result["db_path"] == "/tmp/x.duckdb"
    assert result["meta"]["engram_version"] == "0.11.0"
    assert result["counts"]["tasks"] == 1


def test_handler_advice_lists_all_problems(monkeypatch):
    """Every problem should add one bullet to advice (not crash, not dedupe wrong)."""
    fake = {
        "db_path": "/tmp/x.duckdb",
        "db_exists": True,
        "event_dir": "/tmp/events",
        "residue_files": ["/tmp/x.corrupt.20260101", "/tmp/x.wal-recovery.20260101"],
        "backups": {"dir": "/tmp/backups", "live_count": 15, "retain": 10,
                    "archive_count": 0, "live_recent": []},
        "snapshots": {"count": 0},
        "event_kinds": {},
        "event_max_seq": 0,
        "meta": {},
        "counts": {"memories": 0, "tasks": 0, "checkpoints": 0},
        "readonly": True,
        "embedding_stale": True,
    }
    monkeypatch.setattr("engram.recover.doctor", lambda: fake)

    result = handle_get_runtime_health(db=None, graph=None)
    assert result["ok"] is True
    advice = result["advice"]
    assert len(advice) == 4
    # All 4 problem types must be mentioned in some bullet.
    joined = " | ".join(advice)
    assert "readonly" in joined.lower()
    assert "residue" in joined.lower()
    assert "embedding" in joined.lower()
    assert "backups" in joined.lower() or "retention" in joined.lower()


def test_handler_signature_accepts_extra_kwargs():
    """Tool dispatcher passes user_id etc. as **kwargs even for zero-arg tools."""
    # Should not raise even when called with extras.
    import unittest.mock as mock
    with mock.patch("engram.recover.doctor", return_value={
        "db_path": "/x", "db_exists": True, "event_dir": "/y",
        "residue_files": [], "backups": {"live_count": 0, "retain": 10},
        "snapshots": {"count": 0}, "event_kinds": {}, "event_max_seq": 0,
        "meta": {}, "counts": {}, "readonly": False, "embedding_stale": False,
    }):
        result = handle_get_runtime_health(
            db=None, graph=None, user_id="alice", extra="ignored"
        )
        assert result["ok"] is True
