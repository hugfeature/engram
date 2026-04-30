"""Unit tests for handlers.py — pure business logic, no transport."""

import pytest

from engram.db import MemoryDB
from engram.graph import MemoryGraph
from engram.handlers import (
    handle_recall, handle_store, handle_update,
    handle_session_handoff, handle_consolidate, handle_stats,
)


@pytest.fixture
def env(tmp_path, monkeypatch):
    db = MemoryDB(str(tmp_path / "h.duckdb"))
    graph = MemoryGraph(str(tmp_path / "h.json"))
    monkeypatch.setattr("engram.handlers.embed", lambda t: [0.1] * 768)
    monkeypatch.setattr("engram.retrieve.embed", lambda t: [0.1] * 768)
    return db, graph


class TestHandleStore:
    def test_store_new(self, env):
        db, graph = env
        result = handle_store(db, graph, content="hello world", importance=0.7)
        assert "memory_id" in result
        assert "Stored new memory" in result["result"]

    def test_store_empty_rejected(self, env):
        db, graph = env
        result = handle_store(db, graph, content="", importance=0.5)
        assert "error" in result

    def test_store_clamps_importance(self, env):
        db, graph = env
        result = handle_store(db, graph, content="test", importance=5.0)
        assert "memory_id" in result

    def test_store_invalid_category(self, env):
        db, graph = env
        result = handle_store(db, graph, content="test", importance=0.5, category="bogus")
        assert "memory_id" in result

    def test_store_with_metadata(self, env):
        db, graph = env
        result = handle_store(db, graph, content="meta test", importance=0.5,
                              metadata={"tag": "v"})
        row = db.get_by_id(result["memory_id"])
        assert row.metadata["tag"] == "v"


class TestHandleRecall:
    def test_recall_empty_rejected(self, env):
        db, graph = env
        result = handle_recall(db, graph, query="")
        assert "error" in result

    def test_recall_returns_stored(self, env):
        db, graph = env
        handle_store(db, graph, content="recall target", importance=0.5)
        result = handle_recall(db, graph, query="recall target")
        assert result["memoriesFound"] >= 1


class TestHandleUpdate:
    def test_update_nonexistent(self, env):
        db, graph = env
        result = handle_update(db, graph, memory_id=9999, new_content="x")
        assert "error" in result

    def test_update_existing(self, env):
        db, graph = env
        mid = db.insert("original", [0.1] * 768, 0.5, "fact", "default")
        result = handle_update(db, graph, memory_id=mid, new_content="updated")
        assert "Updated" in result["result"]
        assert db.get_by_id(mid).content == "updated"


class TestHandleSessionHandoff:
    def test_basic_handoff(self, env):
        db, graph = env
        result = handle_session_handoff(db, graph, summary="Did stuff",
                                        completed=["a"], next_steps=["b"])
        assert "memory_id" in result
        assert "recorded" in result["result"]

    def test_handoff_empty_rejected(self, env):
        db, graph = env
        result = handle_session_handoff(db, graph, summary="")
        assert "error" in result


class TestHandleConsolidate:
    def test_consolidate_empty(self, env):
        db, graph = env
        result = handle_consolidate(db, graph)
        assert "No similar" in result["result"]


class TestHandleStats:
    def test_stats_empty(self, env):
        db, graph = env
        result = handle_stats(db)
        assert result["total"] == 0

    def test_stats_with_data(self, env):
        db, graph = env
        db.insert("m1", [0.1] * 768, 0.5, "fact", "default")
        result = handle_stats(db)
        assert result["total"] == 1
        assert "fact" in result["categories"]
