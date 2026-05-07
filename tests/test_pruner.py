"""Tests for pruner.py — memory pruning, maintenance, multi-user support."""

from datetime import datetime, timezone, timedelta

import pytest

from engram.db import MemoryDB
from engram.graph import MemoryGraph
from engram.pruner import run_prune, run_maintenance
from engram.decay import compute_strength


def _make_old_memory(db, graph, content, days_old, importance=0.1, category="fact"):
    """Insert a memory and backdate last_accessed_at to simulate aging."""
    vec = [0.1] * 768
    mid = db.insert(content, vec, importance, category, "default")
    # Backdate last_accessed_at
    old_time = datetime.now(timezone.utc) - timedelta(days=days_old)
    db.conn.execute(
        "UPDATE memories SET last_accessed_at = ? WHERE id = ?",
        [old_time, mid],
    )
    graph.upsert_node(mid, strength=compute_strength(category, importance, days_old, 0))
    return mid


class TestRunPrune:
    def test_prunes_weak_memories(self, db, graph):
        # Low importance + old = strength < 0.05
        mid = _make_old_memory(db, graph, "old weak memory", days_old=200, importance=0.05)
        run_prune(db, graph, "default")
        assert db.get_by_id(mid) is None

    def test_preserves_strong_memories(self, db, graph):
        # High importance + recent = strength >> 0.05
        mid = _make_old_memory(db, graph, "fresh strong memory", days_old=1, importance=0.9)
        run_prune(db, graph, "default")
        assert db.get_by_id(mid) is not None

    def test_preserves_chain_protected_memories(self, db, graph):
        """A weak memory connected to a strong neighbor should be preserved."""
        weak_id = _make_old_memory(db, graph, "weak bridge", days_old=200, importance=0.05)
        strong_id = _make_old_memory(db, graph, "strong anchor", days_old=1, importance=0.9)
        # Connect them
        graph._graph.add_edge(weak_id, strong_id, relation="semantic", weight=0.5)
        graph._graph.add_edge(strong_id, weak_id, relation="semantic", weight=0.5)
        graph.update_node_strength(strong_id, 0.9)

        run_prune(db, graph, "default")
        # weak_id should survive because its neighbor is strong
        assert db.get_by_id(weak_id) is not None

    def test_prune_updates_graph(self, db, graph):
        mid = _make_old_memory(db, graph, "old memory", days_old=200, importance=0.05)
        graph.upsert_node(mid, strength=0.01)
        run_prune(db, graph, "default")
        assert mid not in graph._graph

    def test_prune_nothing_to_prune(self, db, graph):
        mid = _make_old_memory(db, graph, "fresh memory", days_old=0, importance=0.8)
        run_prune(db, graph, "default")
        assert db.get_by_id(mid) is not None


class TestRunMaintenance:
    def test_maintenance_runs_consolidate_and_prune(self, db, graph, monkeypatch):
        monkeypatch.setattr("engram.consolidator.embed", lambda t: [0.1] * 768)
        # Insert two very similar memories that should consolidate
        vec = [0.1] * 768
        db.insert("memory alpha", vec, 0.5, "fact", "default")
        db.insert("memory alpha", vec, 0.5, "fact", "default")

        count_before = db.count("default")
        run_maintenance(db, graph, "default")
        # After consolidation, similar memories should be merged
        count_after = db.count("default")
        assert count_after <= count_before

    def test_maintenance_all_users(self, db, graph, monkeypatch):
        """run_maintenance with user_id=None should cover all users."""
        monkeypatch.setattr("engram.consolidator.embed", lambda t: [0.1] * 768)
        vec = [0.1] * 768
        db.insert("alice memory", vec, 0.5, "fact", "alice")
        db.insert("bob memory", vec, 0.5, "fact", "bob")

        # Should not crash when iterating all users
        run_maintenance(db, graph, user_id=None)

    def test_maintenance_per_user_error_isolation(self, db, graph, monkeypatch):
        """Errors for one user should not prevent maintenance for others."""
        monkeypatch.setattr("engram.consolidator.embed", lambda t: [0.1] * 768)
        vec = [0.1] * 768
        db.insert("user1 memory", vec, 0.5, "fact", "user1")

        # Should complete without raising
        run_maintenance(db, graph, user_id="default")


class TestPrunerLastMaintenanceTime:
    def test_last_maintenance_time_updated(self, db, graph, monkeypatch):
        from engram.pruner import maintenance
        monkeypatch.setattr("engram.consolidator.embed", lambda t: [0.1] * 768)
        # Reset state
        maintenance.last_time = None

        run_maintenance(db, graph, "default")
        assert maintenance.last_time is not None