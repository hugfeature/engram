"""Integration tests — server tool handlers, input validation, graph JSON persistence."""

import json
import os
import tempfile

import numpy as np
import pytest

from engram.db import MemoryDB
from engram.graph import MemoryGraph


class TestGraphJsonPersistence:
    def test_save_and_reload(self, tmp_path):
        db_path = str(tmp_path / "g.duckdb")
        path = str(tmp_path / "g.json")
        db = MemoryDB(db_path, dim=768)
        g1 = MemoryGraph(path)
        g1.upsert_node(1, strength=0.9)
        g1.upsert_node(2, strength=0.7)

        vec = np.random.randn(768)
        vec = (vec / np.linalg.norm(vec)).tolist()
        db.insert("memory 1", vec, 0.9, "fact", "default")
        db.insert("memory 2", vec, 0.7, "fact", "default")
        g1.index_memory_incremental(1, vec, db, "default", 0.9, "fact")
        g1.index_memory_incremental(2, vec, db, "default", 0.7, "fact")
        g1.flush()

        g2 = MemoryGraph(path)
        assert 1 in g2._graph
        assert 2 in g2._graph
        neighbors = g2.expand([1], max_depth=1)
        assert any(mid == 2 for mid, _ in neighbors)

    def test_json_file_is_valid_json(self, tmp_path):
        path = str(tmp_path / "g.json")
        g = MemoryGraph(path)
        g.upsert_node(42, strength=1.0)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "nodes" in data or "links" in data

    def test_pkl_migration(self, tmp_path):
        import pickle
        import networkx as nx

        pkl_path = str(tmp_path / "g.pkl")
        json_path = str(tmp_path / "g.json")

        old_graph = nx.DiGraph()
        old_graph.add_node(99, strength=0.5, user_id="default")
        with open(pkl_path, "wb") as f:
            pickle.dump(old_graph, f)

        g = MemoryGraph(json_path)
        assert 99 in g._graph
        assert not os.path.exists(pkl_path)
        assert os.path.exists(json_path)


class TestSearchSimilarForDedup:
    @pytest.fixture
    def db(self, tmp_path):
        return MemoryDB(str(tmp_path / "test.duckdb"), dim=768)

    def test_returns_similar(self, db):
        vec = np.ones(768)
        vec = (vec / np.linalg.norm(vec)).tolist()
        db.insert("hello world", vec, 0.8, "fact", "default")

        results = db.search_similar_for_dedup(vec, "default", top_k=5, threshold=0.5)
        assert len(results) == 1
        assert results[0][0] > 0
        assert results[0][1] == "hello world"

    def test_respects_threshold(self, db):
        vec1 = np.random.randn(768)
        vec1 = (vec1 / np.linalg.norm(vec1)).tolist()
        vec2 = np.random.randn(768)
        vec2 = (vec2 / np.linalg.norm(vec2)).tolist()

        db.insert("memory A", vec1, 0.5, "fact", "default")

        results = db.search_similar_for_dedup(vec2, "default", top_k=5, threshold=0.99)
        assert len(results) == 0


class TestInputValidation:
    """Test input validation via the call_tool handler."""

    @pytest.fixture
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setattr("engram.shared._db", MemoryDB(str(tmp_path / "t.duckdb"), dim=768))
        monkeypatch.setattr("engram.shared._graph", MemoryGraph(str(tmp_path / "t.json")))

    @pytest.mark.asyncio
    async def test_recall_empty_query(self, setup):
        from engram.server import call_tool
        result = await call_tool("recall_memory", {"query": ""})
        data = json.loads(result[0].text)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_store_empty_content(self, setup):
        from engram.server import call_tool
        result = await call_tool("store_memory", {"content": "  ", "importance": 0.5})
        data = json.loads(result[0].text)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_store_clamps_importance(self, setup, monkeypatch):
        monkeypatch.setattr("engram.handlers.embed", lambda t: [0.1] * 768)
        from engram.server import call_tool
        result = await call_tool("store_memory", {"content": "test", "importance": 5.0})
        data = json.loads(result[0].text)
        assert "result" in data

    @pytest.mark.asyncio
    async def test_store_invalid_category_falls_back(self, setup, monkeypatch):
        monkeypatch.setattr("engram.handlers.embed", lambda t: [0.1] * 768)
        from engram.server import call_tool
        result = await call_tool("store_memory", {"content": "test", "importance": 0.5, "category": "invalid"})
        data = json.loads(result[0].text)
        assert "result" in data

    @pytest.mark.asyncio
    async def test_recall_clamps_top_k(self, setup, monkeypatch):
        monkeypatch.setattr("engram.handlers.embed", lambda t: [0.1] * 768)
        from engram.server import call_tool
        result = await call_tool("recall_memory", {"query": "test", "top_k": 999})
        data = json.loads(result[0].text)
        assert data["memoriesFound"] == 0


class TestServerUpdateMemory:
    @pytest.fixture
    def setup(self, tmp_path, monkeypatch):
        db = MemoryDB(str(tmp_path / "t.duckdb"), dim=768)
        graph = MemoryGraph(str(tmp_path / "t.json"))
        monkeypatch.setattr("engram.shared._db", db)
        monkeypatch.setattr("engram.shared._graph", graph)
        monkeypatch.setattr("engram.handlers.embed", lambda t: [0.1] * 768)
        return db

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, setup):
        from engram.server import call_tool
        result = await call_tool("update_memory", {"memory_id": 9999, "new_content": "x"})
        data = json.loads(result[0].text)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_update_existing(self, setup):
        db = setup
        mid = db.insert("original", [0.1] * 768, 0.5, "fact", "default")

        from engram.server import call_tool
        result = await call_tool("update_memory", {"memory_id": mid, "new_content": "updated"})
        data = json.loads(result[0].text)
        assert "Updated" in data["result"]

        m = db.get_by_id(mid)
        assert m.content == "updated"
