"""Core tests for Engram — decay, resolve, db, graph."""

import math
import os
import tempfile
import shutil

import pytest


class TestDecay:
    def test_strength_decreases_over_time(self):
        from engram.decay import compute_strength

        s0 = compute_strength("fact", 0.8, 0)
        s10 = compute_strength("fact", 0.8, 10)
        s30 = compute_strength("fact", 0.8, 30)
        assert s0 > s10 > s30

    def test_higher_importance_decays_slower(self):
        from engram.decay import compute_strength

        high = compute_strength("fact", 0.9, 20)
        low = compute_strength("fact", 0.3, 20)
        assert high > low

    def test_recall_count_boosts(self):
        from engram.decay import compute_strength

        no_recall = compute_strength("fact", 0.5, 10, recall_count=0)
        with_recall = compute_strength("fact", 0.5, 10, recall_count=5)
        assert with_recall > no_recall

    def test_strategy_decays_slowest(self):
        from engram.decay import compute_strength

        strategy = compute_strength("strategy", 0.5, 30)
        failure = compute_strength("failure", 0.5, 30)
        assert strategy > failure

    def test_strength_capped_at_1(self):
        from engram.decay import compute_strength

        s = compute_strength("strategy", 1.0, 0, recall_count=10)
        assert s <= 1.0


class TestResolve:
    def test_new_memory_when_no_existing(self):
        from engram.resolve import resolve, Action

        r = resolve("test", [0.1] * 768, [])
        assert r.action == Action.NEW

    def test_reinforce_very_similar(self):
        from engram.resolve import resolve, Action

        emb = [0.1] * 768
        existing = [(1, "test content", emb)]
        r = resolve("test content exactly", emb, existing)
        assert r.action == Action.REINFORCE

    def test_new_when_dissimilar(self):
        from engram.resolve import resolve, Action
        import numpy as np

        emb_a = np.random.randn(768).tolist()
        emb_b = np.random.randn(768).tolist()
        norm_a = (np.array(emb_a) / np.linalg.norm(emb_a)).tolist()
        norm_b = (np.array(emb_b) / np.linalg.norm(emb_b)).tolist()
        existing = [(1, "totally different", norm_b)]
        r = resolve("new thing", norm_a, existing)
        assert r.action == Action.NEW

    def test_contradiction_detection(self):
        from engram.resolve import _is_contradiction

        assert _is_contradiction("I love Python", "I hate Python")
        assert not _is_contradiction("I love Python", "I love JavaScript")


class TestDB:
    @pytest.fixture
    def db(self, tmp_path):
        from engram.db import MemoryDB

        db_path = str(tmp_path / "test.duckdb")
        return MemoryDB(db_path)

    def test_insert_and_get(self, db):
        emb = [0.1] * 768
        mid = db.insert("test memory", emb, 0.8, "fact", "user1")
        assert mid > 0

        m = db.get_by_id(mid)
        assert m is not None
        assert m.content == "test memory"
        assert abs(m.importance - 0.8) < 1e-6

    def test_update(self, db):
        emb = [0.1] * 768
        mid = db.insert("original", emb, 0.5)
        db.update(mid, "updated", emb, 0.9)

        m = db.get_by_id(mid)
        assert m.content == "updated"
        assert abs(m.importance - 0.9) < 1e-6

    def test_delete(self, db):
        emb = [0.1] * 768
        mid = db.insert("to delete", emb)
        db.delete(mid)
        assert db.get_by_id(mid) is None

    def test_bump_recall(self, db):
        emb = [0.1] * 768
        mid = db.insert("recall me", emb)
        m1 = db.get_by_id(mid)
        assert m1.recall_count == 0

        db.bump_recall(mid)
        m2 = db.get_by_id(mid)
        assert m2.recall_count == 1

    def test_vector_search(self, db):
        import numpy as np

        emb1 = (np.random.randn(768)).tolist()
        emb2 = (np.random.randn(768)).tolist()
        norm1 = (np.array(emb1) / np.linalg.norm(emb1)).tolist()
        norm2 = (np.array(emb2) / np.linalg.norm(emb2)).tolist()

        db.insert("memory one", norm1, 0.8, "fact", "default")
        db.insert("memory two", norm2, 0.5, "fact", "default")

        results = db.search_vector(norm1, "default", top_k=5, threshold=0.0)
        assert len(results) >= 1
        assert results[0].content == "memory one"

    def test_count(self, db):
        emb = [0.1] * 768
        assert db.count() == 0
        db.insert("one", emb)
        db.insert("two", emb)
        assert db.count() == 2


class TestGraph:
    @pytest.fixture
    def graph(self, tmp_path):
        from engram.graph import MemoryGraph

        return MemoryGraph(str(tmp_path / "test_graph.pkl"))

    def test_upsert_and_expand(self, graph):
        import numpy as np

        graph.upsert_node(1, strength=1.0)
        graph.upsert_node(2, strength=0.8)

        vec1 = np.random.randn(768)
        vec1 = (vec1 / np.linalg.norm(vec1)).tolist()
        vec2 = vec1[:]  # identical = high similarity

        graph.index_memory(1, vec1, {2: vec2})
        neighbors = graph.expand([1], max_depth=1)
        assert any(mid == 2 for mid, _ in neighbors)

    def test_chain_safe_prune(self, graph):
        graph.upsert_node(1, strength=0.01)
        graph.upsert_node(2, strength=0.01)
        assert graph.chain_safe_to_prune(1, 0.05) is True

        graph.upsert_node(3, strength=0.8)
        import numpy as np
        vec = np.random.randn(768)
        vec = (vec / np.linalg.norm(vec)).tolist()
        graph.index_memory(1, vec, {3: vec})
        assert graph.chain_safe_to_prune(1, 0.05) is False


class TestConsolidator:
    @pytest.fixture
    def setup(self, tmp_path):
        from engram.db import MemoryDB
        from engram.graph import MemoryGraph

        db = MemoryDB(str(tmp_path / "test.duckdb"))
        graph = MemoryGraph(str(tmp_path / "test_graph.pkl"))
        return db, graph

    def test_no_consolidation_when_dissimilar(self, setup):
        import numpy as np
        from engram.consolidator import run_consolidate

        db, graph = setup
        v1 = np.random.randn(768)
        v1 = (v1 / np.linalg.norm(v1)).tolist()
        v2 = np.random.randn(768)
        v2 = (v2 / np.linalg.norm(v2)).tolist()

        db.insert("Python is great", v1, 0.8, "fact")
        db.insert("DuckDB has file locks", v2, 0.5, "failure")

        results = run_consolidate(db, graph)
        assert results == []
        assert db.count() == 2

    def test_consolidation_merges_similar(self, setup):
        from engram.consolidator import run_consolidate

        db, graph = setup
        vec = [0.1] * 768

        db.insert("User prefers Python", vec, 0.8, "fact")
        db.insert("User likes Python for dev", vec, 0.6, "fact")

        results = run_consolidate(db, graph)
        assert len(results) == 1
        assert db.count() == 1

        remaining = db.get_all()
        assert abs(remaining[0].importance - 0.8) < 1e-6

    def test_consolidation_keeps_higher_importance(self, setup):
        from engram.consolidator import run_consolidate

        db, graph = setup
        vec = [0.1] * 768

        db.insert("low importance", vec, 0.3, "fact")
        db.insert("high importance", vec, 0.9, "fact")
        db.insert("medium importance", vec, 0.5, "fact")

        results = run_consolidate(db, graph)
        assert len(results) == 1
        remaining = db.get_all()
        assert len(remaining) == 1
        assert abs(remaining[0].importance - 0.9) < 1e-6
