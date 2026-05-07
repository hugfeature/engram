"""v0.2 feature tests — metadata column, session_handoff, store metadata, backward compat."""

import json

import numpy as np
import pytest

from engram.db import MemoryDB, _parse_metadata
from engram.graph import MemoryGraph


class TestMetadataColumn:
    @pytest.fixture
    def db(self, tmp_path):
        return MemoryDB(str(tmp_path / "test.duckdb"), dim=768)

    def test_default_metadata_is_empty_dict(self, db):
        vec = [0.1] * 768
        mid = db.insert("test content", vec, 0.5, "fact", "default")
        row = db.get_by_id(mid)
        assert row.metadata == {}

    def test_insert_with_metadata(self, db):
        vec = [0.1] * 768
        meta = {"layer": "validation", "fix": "add null check"}
        mid = db.insert("test", vec, 0.5, "failure", "default", metadata=meta)
        row = db.get_by_id(mid)
        assert row.metadata["layer"] == "validation"
        assert row.metadata["fix"] == "add null check"

    def test_metadata_survives_update(self, db):
        vec = [0.1] * 768
        meta = {"source": "test"}
        mid = db.insert("original", vec, 0.5, "fact", "default", metadata=meta)
        db.update(mid, "updated content", vec, 0.6)
        row = db.get_by_id(mid)
        assert row.content == "updated content"
        assert row.metadata["source"] == "test"

    def test_get_metadata_batch(self, db):
        vec = [0.1] * 768
        id1 = db.insert("a", vec, 0.5, "fact", "default", metadata={"k": "v1"})
        id2 = db.insert("b", vec, 0.5, "fact", "default", metadata={"k": "v2"})
        batch = db.get_metadata_batch([id1, id2])
        assert batch[id1]["k"] == "v1"
        assert batch[id2]["k"] == "v2"

    def test_get_metadata_batch_empty(self, db):
        assert db.get_metadata_batch([]) == {}

    def test_search_vector_includes_metadata(self, db):
        vec = np.ones(768)
        vec = (vec / np.linalg.norm(vec)).tolist()
        db.insert("searchable", vec, 0.5, "fact", "default", metadata={"tag": "yes"})
        results = db.search_vector(vec, "default", top_k=5, threshold=0.5)
        assert len(results) == 1
        assert results[0].metadata["tag"] == "yes"

    def test_get_all_includes_metadata(self, db):
        vec = [0.1] * 768
        db.insert("m1", vec, 0.5, "fact", "default", metadata={"x": 1})
        all_rows = db.get_all("default")
        assert len(all_rows) == 1
        assert all_rows[0].metadata["x"] == 1


class TestParseMetadata:
    def test_none(self):
        assert _parse_metadata(None) == {}

    def test_dict(self):
        assert _parse_metadata({"a": 1}) == {"a": 1}

    def test_json_string(self):
        assert _parse_metadata('{"b": 2}') == {"b": 2}

    def test_invalid_string(self):
        assert _parse_metadata("not json") == {}


class TestSessionHandoff:
    @pytest.fixture
    def setup(self, tmp_path, monkeypatch):
        db = MemoryDB(str(tmp_path / "t.duckdb"), dim=768)
        graph = MemoryGraph(str(tmp_path / "t.json"))
        monkeypatch.setattr("engram.shared._db", db)
        monkeypatch.setattr("engram.shared._graph", graph)
        monkeypatch.setattr("engram.handlers.embed", lambda t: [0.1] * 768)
        return db

    @pytest.mark.asyncio
    async def test_basic_handoff(self, setup):
        from engram.server import call_tool
        result = await call_tool("session_handoff", {
            "summary": "Implemented auth flow",
            "completed": ["login endpoint", "token validation"],
            "next_steps": ["add refresh tokens"],
        })
        data = json.loads(result[0].text)
        assert "memory_id" in data
        assert "recorded" in data["result"]

    @pytest.mark.asyncio
    async def test_handoff_empty_summary_rejected(self, setup):
        from engram.server import call_tool
        result = await call_tool("session_handoff", {"summary": ""})
        data = json.loads(result[0].text)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_handoff_uses_strategy_category(self, setup):
        db = setup
        from engram.server import call_tool
        result = await call_tool("session_handoff", {"summary": "test session"})
        data = json.loads(result[0].text)
        mid = data["memory_id"]
        row = db.get_by_id(mid)
        assert row.category == "strategy"
        assert abs(row.importance - 0.9) < 0.01

    @pytest.mark.asyncio
    async def test_handoff_metadata_structure(self, setup):
        db = setup
        from engram.server import call_tool
        result = await call_tool("session_handoff", {
            "summary": "Did X",
            "completed": ["a"],
            "blocked": ["b"],
        })
        data = json.loads(result[0].text)
        row = db.get_by_id(data["memory_id"])
        meta = row.metadata
        assert meta["type"] == "handoff"
        assert meta["summary"] == "Did X"
        assert meta["completed"] == ["a"]
        assert meta["blocked"] == ["b"]
        assert "timestamp" in meta

    @pytest.mark.asyncio
    async def test_handoff_searchable(self, setup):
        db = setup
        from engram.server import call_tool
        await call_tool("session_handoff", {
            "summary": "Implemented auth flow for production",
            "completed": ["login", "logout"],
        })
        row = db.get_all("default")[0]
        assert "auth flow" in row.content
        assert "login" in row.content

    @pytest.mark.asyncio
    async def test_handoff_optional_fields(self, setup):
        from engram.server import call_tool
        result = await call_tool("session_handoff", {"summary": "Minimal handoff"})
        data = json.loads(result[0].text)
        assert "memory_id" in data


class TestStoreWithMetadata:
    @pytest.fixture
    def setup(self, tmp_path, monkeypatch):
        db = MemoryDB(str(tmp_path / "t.duckdb"), dim=768)
        graph = MemoryGraph(str(tmp_path / "t.json"))
        monkeypatch.setattr("engram.shared._db", db)
        monkeypatch.setattr("engram.shared._graph", graph)
        monkeypatch.setattr("engram.handlers.embed", lambda t: [0.1] * 768)
        return db

    @pytest.mark.asyncio
    async def test_store_with_failure_metadata(self, setup):
        db = setup
        from engram.server import call_tool
        result = await call_tool("store_memory", {
            "content": "Login validation failed due to missing CSRF token",
            "importance": 0.7,
            "category": "failure",
            "metadata": {"layer": "validation", "fix": "add CSRF middleware"},
        })
        data = json.loads(result[0].text)
        assert "memory_id" in data
        row = db.get_by_id(data["memory_id"])
        assert row.metadata["layer"] == "validation"

    @pytest.mark.asyncio
    async def test_store_without_metadata_backward_compat(self, setup):
        from engram.server import call_tool
        result = await call_tool("store_memory", {
            "content": "simple memory no metadata",
            "importance": 0.5,
        })
        data = json.loads(result[0].text)
        assert "result" in data

    @pytest.mark.asyncio
    async def test_store_returns_memory_id(self, setup):
        from engram.server import call_tool
        result = await call_tool("store_memory", {
            "content": "test return id",
            "importance": 0.5,
        })
        data = json.loads(result[0].text)
        assert "memory_id" in data
        assert isinstance(data["memory_id"], int)


class TestRecallWithMetadata:
    @pytest.fixture
    def setup(self, tmp_path, monkeypatch):
        db = MemoryDB(str(tmp_path / "t.duckdb"), dim=768)
        graph = MemoryGraph(str(tmp_path / "t.json"))
        monkeypatch.setattr("engram.shared._db", db)
        monkeypatch.setattr("engram.shared._graph", graph)
        monkeypatch.setattr("engram.handlers.embed", lambda t: [0.1] * 768)
        monkeypatch.setattr("engram.retrieve.embed", lambda t: [0.1] * 768)
        return db

    @pytest.mark.asyncio
    async def test_recall_includes_metadata(self, setup):
        db = setup
        from engram.server import call_tool
        await call_tool("store_memory", {
            "content": "test recall meta",
            "importance": 0.5,
            "metadata": {"tag": "recall_test"},
        })
        result = await call_tool("recall_memory", {"query": "test recall meta"})
        data = json.loads(result[0].text)
        assert data["memoriesFound"] >= 1
        found = [m for m in data["memories"] if m.get("metadata", {}).get("tag") == "recall_test"]
        assert len(found) >= 1

    @pytest.mark.asyncio
    async def test_recall_no_metadata_omits_field(self, setup):
        db = setup
        from engram.server import call_tool
        await call_tool("store_memory", {
            "content": "no meta memory",
            "importance": 0.5,
        })
        result = await call_tool("recall_memory", {"query": "no meta memory"})
        data = json.loads(result[0].text)
        if data["memoriesFound"] > 0:
            for m in data["memories"]:
                if m["content"] == "no meta memory":
                    assert "metadata" not in m or m["metadata"] == {}


class TestBackwardCompat:
    @pytest.fixture
    def setup(self, tmp_path, monkeypatch):
        db = MemoryDB(str(tmp_path / "t.duckdb"), dim=768)
        graph = MemoryGraph(str(tmp_path / "t.json"))
        monkeypatch.setattr("engram.shared._db", db)
        monkeypatch.setattr("engram.shared._graph", graph)
        monkeypatch.setattr("engram.handlers.embed", lambda t: [0.1] * 768)
        return db

    @pytest.mark.asyncio
    async def test_original_tools_still_work(self):
        from engram.server import list_tools
        tools = await list_tools()
        names = {t.name for t in tools}
        assert names >= {"recall_memory", "store_memory", "update_memory", "consolidate_memory", "memory_stats"}

    @pytest.mark.asyncio
    async def test_session_handoff_registered(self):
        from engram.server import list_tools
        tools = await list_tools()
        names = {t.name for t in tools}
        assert "session_handoff" in names
