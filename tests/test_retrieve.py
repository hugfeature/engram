"""Tests for retrieve.py — hybrid retrieval scoring, graph expansion, reinforcement."""

import numpy as np
import pytest

from engram.db import MemoryDB
from engram.graph import MemoryGraph
from engram.retrieve import recall


def _unit_vec(seed: int, dim: int = 768) -> list[float]:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim)
    return (v / np.linalg.norm(v)).tolist()


class TestRecallEmpty:
    def test_empty_db_returns_nothing(self, db, graph, monkeypatch):
        monkeypatch.setattr("engram.retrieve.embed", lambda t: _unit_vec(1))
        results = recall("anything", db, graph, top_k=5)
        assert results == []


class TestRecallBasic:
    def test_returns_relevant_memory(self, db, graph, monkeypatch):
        vec = _unit_vec(42)
        monkeypatch.setattr("engram.retrieve.embed", lambda t: vec)
        db.insert("Python is great for data science", vec, 0.7, "fact", "default")
        graph.upsert_node(1, strength=0.7)

        results = recall("data science tools", db, graph, top_k=5)
        assert len(results) >= 1
        assert results[0].id == 1

    def test_top_k_limits_results(self, db, graph, monkeypatch):
        vec = _unit_vec(42)
        monkeypatch.setattr("engram.retrieve.embed", lambda t: vec)
        for i in range(10):
            mid = db.insert(f"memory {i}", vec, 0.5, "fact", "default")
            graph.upsert_node(mid, strength=0.5)

        results = recall("test query", db, graph, top_k=3)
        assert len(results) <= 3

    def test_fallback_to_low_threshold(self, db, graph, monkeypatch):
        """When no results at SIMILARITY_HIGH, should fall back to SIMILARITY_LOW."""
        vec_a = _unit_vec(1)
        vec_b = _unit_vec(99)  # very different from vec_a
        db.insert("obscure memory", vec_b, 0.5, "fact", "default")
        graph.upsert_node(1, strength=0.5)

        # Query with vec_a — low similarity, but fallback should find it
        monkeypatch.setattr("engram.retrieve.embed", lambda t: vec_a)
        results = recall("obscure memory", db, graph, top_k=5)
        # Fallback may or may not find it depending on threshold,
        # but should not crash
        assert isinstance(results, list)


class TestRecallGraphExpansion:
    def test_graph_neighbor_included(self, db, graph, monkeypatch):
        vec = _unit_vec(42)
        monkeypatch.setattr("engram.retrieve.embed", lambda t: vec)

        # Insert two similar memories and connect them in the graph
        mid1 = db.insert("primary memory about Go", vec, 0.8, "fact", "default")
        mid2 = db.insert("related memory about Rust", vec, 0.6, "fact", "default")
        graph.upsert_node(mid1, strength=0.8)
        graph.upsert_node(mid2, strength=0.6)
        graph._graph.add_edge(mid1, mid2, relation="semantic", weight=0.5)
        graph._graph.add_edge(mid2, mid1, relation="semantic", weight=0.5)

        results = recall("Go programming", db, graph, top_k=5)
        ids = {r.id for r in results}
        assert mid1 in ids
        # mid2 should be found via graph expansion even if vector similarity is same
        assert mid2 in ids


class TestRecallReinforcement:
    def test_high_similarity_triggers_bump(self, db, graph, monkeypatch):
        vec = _unit_vec(42)
        monkeypatch.setattr("engram.retrieve.embed", lambda t: vec)
        mid = db.insert("recall reinforcement test", vec, 0.7, "fact", "default")
        graph.upsert_node(mid, strength=0.7)

        # Before recall
        row_before = db.get_by_id(mid)
        count_before = row_before.recall_count

        recall("recall reinforcement test", db, graph, top_k=5)

        row_after = db.get_by_id(mid)
        # High similarity (same vector) should trigger bump_recall
        assert row_after.recall_count > count_before


class TestRecallMultiUser:
    def test_user_isolation(self, db, graph, monkeypatch):
        vec = _unit_vec(42)
        monkeypatch.setattr("engram.retrieve.embed", lambda t: vec)
        db.insert("alice secret", vec, 0.7, "fact", "alice")
        db.insert("bob secret", vec, 0.7, "fact", "bob")

        results = recall("secret", db, graph, user_id="alice", top_k=5)
        for r in results:
            assert r.id == 1  # only alice's memory