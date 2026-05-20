"""Tests for v0.19 features: Drift Nudge + Stats + Adaptive Checkpoint."""

import os
import json
import tempfile

import pytest

from engram.db import MemoryDB
from engram.graph import MemoryGraph
from engram.stats import compute_stats, EngineStats

FAKE_EMBED = [0.1] * 768


@pytest.fixture(autouse=True)
def _disable_sqlite(monkeypatch):
    monkeypatch.setenv("ENGRAM_SQLITE_TIER2", "0")


@pytest.fixture
def env(tmp_path, monkeypatch):
    db = MemoryDB(str(tmp_path / "test.duckdb"), dim=768)
    graph = MemoryGraph(str(tmp_path / "test.json"))
    monkeypatch.setattr("engram.handlers.embed", lambda t: FAKE_EMBED)
    monkeypatch.setattr("engram.retrieve.embed", lambda t: FAKE_EMBED)
    return db, graph


class TestDriftNudge:
    """Phase 3: Drift → Soft Nudge auto-warning."""

    def test_nudge_not_emitted_below_threshold(self, env, monkeypatch):
        """No nudge when drift is below threshold."""
        db, graph = env
        monkeypatch.setenv("ENGRAM_DRIFT_NUDGE_THRESHOLD", "0.9")
        monkeypatch.setattr("engram.embedding.embed", lambda t: FAKE_EMBED)

        from engram.drift import DriftSignal, _maybe_emit_drift_nudge

        signal = DriftSignal(
            goal_drift=0.3, tool_drift=0.1,
            planning_drift=0.2, constraint_drift=0.2,
            composite=0.5,
        )

        initial_count = db.count()
        _maybe_emit_drift_nudge(db, signal, {"goal": "test"}, task_id=1, user_id="default")
        assert db.count() == initial_count  # No new memory stored

    def test_nudge_emitted_above_threshold(self, env, monkeypatch):
        """Nudge emitted when drift exceeds threshold."""
        monkeypatch.setenv("ENGRAM_DRIFT_NUDGE_THRESHOLD", "0.5")
        monkeypatch.setattr("engram.embedding.embed", lambda t: FAKE_EMBED)

        db, graph = env
        from engram.drift import DriftSignal, _maybe_emit_drift_nudge

        signal = DriftSignal(
            goal_drift=0.8, tool_drift=0.5,
            planning_drift=0.6, constraint_drift=0.9,
            composite=0.75,
            violations=["REDO_VIOLATION: 'x' was in must_not_redo but was redone"],
        )

        initial_count = db.count()
        _maybe_emit_drift_nudge(
            db, signal, {"goal": "fix the auth bug"}, task_id=42, user_id="default"
        )
        assert db.count() == initial_count + 1

        # Verify the stored memory content
        memories = db.get_all("default")
        nudge_mem = [m for m in memories if "DRIFT WARNING" in m.content]
        assert len(nudge_mem) == 1
        assert "task#42" in nudge_mem[0].content
        assert "fix the auth bug" in nudge_mem[0].content
        assert nudge_mem[0].category == "failure"
        assert abs(nudge_mem[0].importance - 0.9) < 0.01

    def test_nudge_disabled_via_config(self, env, monkeypatch):
        """Nudge can be disabled via config."""
        monkeypatch.setattr("engram.config.DRIFT_NUDGE_ENABLED", False)
        monkeypatch.setattr("engram.embedding.embed", lambda t: FAKE_EMBED)

        db, graph = env
        from engram.drift import DriftSignal, _maybe_emit_drift_nudge

        signal = DriftSignal(composite=0.99, constraint_drift=1.0)

        initial_count = db.count()
        _maybe_emit_drift_nudge(db, signal, {"goal": "test"}, task_id=1, user_id="default")
        assert db.count() == initial_count


class TestStats:
    """Phase 1: engram stats."""

    def test_compute_stats_empty_dir(self, tmp_path):
        """Stats handles empty/nonexistent event dir gracefully."""
        stats = compute_stats(days=7, event_dir=str(tmp_path / "nonexistent"))
        assert stats.total_sessions == 0
        assert stats.checkpoints_created == 0

    def test_stats_format_table(self):
        """Format table produces readable output."""
        stats = EngineStats(
            period_days=7,
            period_start="2026-05-13",
            period_end="2026-05-20",
            total_sessions=10,
            interrupted_sessions=3,
            recovered_sessions=2,
            checkpoints_created=50,
            checkpoints_restored=5,
        )
        table = stats.format_table()
        assert "Engram Stats" in table
        assert "Total:        10" in table
        assert "Interrupted:  3" in table

    def test_stats_format_report(self):
        """Format report produces valid markdown."""
        stats = EngineStats(
            period_days=7,
            period_start="2026-05-13",
            period_end="2026-05-20",
            total_sessions=10,
            interrupted_sessions=3,
            recovered_sessions=0,
            checkpoints_created=50,
            checkpoints_restored=0,
        )
        report = stats.format_report()
        assert "## Engram Runtime Report" in report
        assert "No recoveries" in report
        assert "Zero checkpoint restores" in report

    def test_stats_to_dict(self):
        """to_dict produces valid JSON-serializable dict."""
        stats = EngineStats(period_days=7)
        d = stats.to_dict()
        assert json.dumps(d)  # serializable
        assert d["period"]["days"] == 7

    def test_stats_with_real_event_log(self, tmp_path):
        """Stats correctly aggregates events from a real event log."""
        from engram.event_log import EventLog

        event_dir = str(tmp_path / "events")
        os.makedirs(event_dir, exist_ok=True)
        log = EventLog(event_dir=event_dir)

        # Simulate events
        log.append("session.start", {"user_id": "default"})
        log.append("memory.store", {"content": "test", "user_id": "default"})
        log.append("checkpoint.write", {
            "task_id": 1, "version": 1, "checkpoint_reason": "FAILURE"
        })
        log.append("session.end", {"end_type": "interrupted"})
        log.append("session.start", {"user_id": "default"})
        log.append("checkpoint.restore", {"task_id": 1, "version": 1})
        log.append("session.memory_recall", {"query": "test"})

        stats = compute_stats(days=7, event_dir=event_dir)
        assert stats.total_sessions == 2
        assert stats.interrupted_sessions == 1
        assert stats.checkpoints_created == 1
        assert stats.checkpoints_restored == 1
        assert stats.recovered_sessions == 1
        assert stats.memories_stored == 1
        assert stats.memories_recalled == 1
        assert stats.checkpoint_reasons == {"FAILURE": 1}


class TestAdaptiveCheckpoint:
    """Phase 2: Adaptive checkpoint interval."""

    def test_default_interval_without_data(self, env, monkeypatch):
        """Returns default interval when not enough event log data (< 5 auto_save)."""
        db, graph = env
        # Mock _get_event_log to return None (no event log)
        db._event_log = None
        monkeypatch.setattr(db, "_get_event_log", lambda: None)

        from engram.checkpoint import _adaptive_auto_save_interval, AUTO_SAVE_FALLBACK_SECONDS

        interval = _adaptive_auto_save_interval(db, "default")
        assert interval == AUTO_SAVE_FALLBACK_SECONDS

    def test_adaptive_disabled_via_config(self, env, monkeypatch):
        """Returns default when adaptive is disabled via config."""
        monkeypatch.setattr("engram.config.ADAPTIVE_CHECKPOINT_ENABLED", False)
        db, graph = env

        from engram.checkpoint import _adaptive_auto_save_interval, AUTO_SAVE_FALLBACK_SECONDS

        interval = _adaptive_auto_save_interval(db, "default")
        assert interval == AUTO_SAVE_FALLBACK_SECONDS

    def test_adaptive_expands_interval_on_low_restore_rate(self, tmp_path, monkeypatch):
        """Interval expands when auto_save checkpoints are rarely restored."""
        monkeypatch.setenv("ENGRAM_SQLITE_TIER2", "0")
        monkeypatch.setenv("ENGRAM_ADAPTIVE_CHECKPOINT", "1")
        monkeypatch.setenv("ENGRAM_ADAPTIVE_LOW_RESTORE_RATE", "0.10")

        from engram.event_log import EventLog
        from engram.checkpoint import _adaptive_auto_save_interval, AUTO_SAVE_FALLBACK_SECONDS

        event_dir = str(tmp_path / "events")
        os.makedirs(event_dir, exist_ok=True)
        event_log = EventLog(event_dir=event_dir)

        # Simulate 10 auto_save checkpoints and 0 restores (0% rate)
        for i in range(10):
            event_log.append("checkpoint.write", {
                "task_id": 1, "version": i + 1, "checkpoint_reason": "AUTO_SAVE"
            })

        db = MemoryDB(str(tmp_path / "test.duckdb"), dim=768)
        monkeypatch.setattr(db, "_get_event_log", lambda: event_log)

        interval = _adaptive_auto_save_interval(db, "default")
        # Should expand (double), capped at max
        assert interval > AUTO_SAVE_FALLBACK_SECONDS
        assert interval <= 600

    def test_config_values_loaded(self):
        """Config values for drift nudge and adaptive checkpoint are accessible."""
        from engram.config import (
            DRIFT_NUDGE_THRESHOLD,
            DRIFT_NUDGE_ENABLED,
            ADAPTIVE_CHECKPOINT_ENABLED,
            ADAPTIVE_LOW_RESTORE_RATE,
            ADAPTIVE_MAX_INTERVAL_SECONDS,
        )
        assert 0.0 < DRIFT_NUDGE_THRESHOLD <= 1.0
        assert isinstance(DRIFT_NUDGE_ENABLED, bool)
        assert isinstance(ADAPTIVE_CHECKPOINT_ENABLED, bool)
        assert 0.0 < ADAPTIVE_LOW_RESTORE_RATE <= 1.0
        assert ADAPTIVE_MAX_INTERVAL_SECONDS >= 300
