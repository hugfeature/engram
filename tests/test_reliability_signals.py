"""Tests for v0.17 — Runtime Reliability Signals.

Covers:
- Interruption Intelligence (enhanced report_interruption)
- Execution Drift Analysis (drift.py)
- Semantic Continuity Scoring (reliability.py)
- Lightweight Recovery Heuristics (recommend_recovery)
"""

import pytest
from engram.db import MemoryDB
from engram.graph import MemoryGraph
from engram.handlers import (
    handle_report_interruption,
    handle_detect_drift,
    handle_score_recovery,
    handle_recommend_recovery,
    handle_start_execution,
    handle_create_task,
)
from engram import checkpoint


@pytest.fixture
def env(tmp_path):
    db = MemoryDB(str(tmp_path / "test.duckdb"), dim=768)
    graph = MemoryGraph(str(tmp_path / "graph.json"))
    return db, graph


@pytest.fixture
def task_with_checkpoint(env):
    """Create a task with a rich checkpoint for drift/scoring tests."""
    db, graph = env
    # Start execution + task
    result = handle_start_execution(db, graph, goal="Fix login bug in auth module")
    task_id = result["task_id"]
    execution_id = result["execution_id"]

        # Create checkpoint with rich state
    checkpoint.create_checkpoint(
        db,
        task_id=task_id,
        reason="PLAN_UPDATE",
        state={
            "goal": "Fix login bug in auth module",
            "completed": ["Identified root cause", "Wrote failing test"],
            "in_progress": ["Implementing fix"],
            "blocked": [],
            "preferred_next": "Apply fix to auth.py",
            "must_not_redo": [
                {"action": "Wrote failing test", "reason": "already_completed"},
                {"action": "Identified root cause", "reason": "already_completed"},
            ],
            "must_preserve": ["Test coverage above 80%"],
            "working_set": {"tools": ["read_file", "file_replace", "shell"]},
            "active_constraints": ["Do not modify user model", "Keep backward compat"],
            "blocked_reasons": [],
        },
        user_id="default",
    )
    return db, graph, task_id, execution_id


# ---- Interruption Intelligence ----

class TestInterruptionIntelligence:
    """report_interruption returns enhanced intelligence signals."""

    def test_overflow_intelligence(self, env):
        db, graph = env
        result = handle_report_interruption(db, graph, reason="overflow")
        assert result["ok"] is True
        intel = result["intelligence"]
        assert intel["severity"] == "medium"
        assert intel["recoverability"] == "high"
        assert intel["data_loss_risk"] == "none"
        assert intel["recommended_action"] == "restore_checkpoint"

    def test_crash_intelligence(self, env):
        db, graph = env
        result = handle_report_interruption(db, graph, reason="crash")
        intel = result["intelligence"]
        assert intel["severity"] == "critical"
        assert intel["recoverability"] == "medium"
        assert intel["data_loss_risk"] == "partial"

    def test_rate_limit_intelligence(self, env):
        db, graph = env
        result = handle_report_interruption(db, graph, reason="rate_limit")
        intel = result["intelligence"]
        assert intel["severity"] == "low"
        assert intel["recoverability"] == "high"
        assert intel["data_loss_risk"] == "none"

    def test_tool_failure_intelligence(self, env):
        db, graph = env
        result = handle_report_interruption(db, graph, reason="tool_failure")
        intel = result["intelligence"]
        assert intel["severity"] == "high"
        assert intel["recoverability"] == "medium"

    def test_unknown_intelligence(self, env):
        db, graph = env
        result = handle_report_interruption(db, graph, reason="unknown")
        intel = result["intelligence"]
        assert intel["severity"] == "medium"
        assert intel["recoverability"] == "low"


# ---- Execution Drift Analysis ----

class TestExecutionDrift:
    """detect_drift measures 4 drift dimensions."""

    def test_no_drift_when_aligned(self, task_with_checkpoint):
        db, graph, task_id, _ = task_with_checkpoint
        result = handle_detect_drift(
            db, graph,
            task_id=task_id,
            current_goal="Fix login bug in auth module",
            tools_used=["read_file", "file_replace"],
            actions_taken=["Implementing fix"],
            in_progress=["Implementing fix"],
        )
        assert result["ok"] is True
        drift = result["drift"]
        assert drift["goal_drift"] < 0.3
        assert drift["composite"] < 0.3
        assert drift["severity"] in ("low", "medium")

    def test_goal_drift_detected(self, task_with_checkpoint):
        db, graph, task_id, _ = task_with_checkpoint
        result = handle_detect_drift(
            db, graph,
            task_id=task_id,
            current_goal="Refactoring the entire authentication system architecture",
            tools_used=["read_file"],
            actions_taken=[],
            in_progress=["Redesigning auth system"],
        )
        assert result["ok"] is True
        drift = result["drift"]
        assert drift["goal_drift"] > 0.4  # Significant goal drift

    def test_constraint_drift_detected(self, task_with_checkpoint):
        db, graph, task_id, _ = task_with_checkpoint
        result = handle_detect_drift(
            db, graph,
            task_id=task_id,
            current_goal="Fix login bug in auth module",
            tools_used=["read_file"],
            actions_taken=["Wrote failing test"],  # This is in must_not_redo!
            in_progress=["Implementing fix"],
        )
        assert result["ok"] is True
        drift = result["drift"]
        assert drift["constraint_drift"] > 0  # Violation detected
        assert any("REDO_VIOLATION" in v for v in (drift.get("violations") or []))

    def test_tool_drift_detected(self, task_with_checkpoint):
        db, graph, task_id, _ = task_with_checkpoint
        result = handle_detect_drift(
            db, graph,
            task_id=task_id,
            current_goal="Fix login bug in auth module",
            tools_used=["web_search", "browser", "screenshot"],  # Completely different tools
            actions_taken=[],
            in_progress=["Implementing fix"],
        )
        assert result["ok"] is True
        drift = result["drift"]
        assert drift["tool_drift"] > 0.5  # Using all new tools

    def test_no_checkpoint_returns_zero_drift(self, env):
        db, graph = env
        task_id = db.create_task(name="fresh task", goal="test")
        result = handle_detect_drift(
            db, graph,
            task_id=task_id,
            current_goal="test",
        )
        assert result["ok"] is True
        assert result["drift"]["composite"] == 0.0

    def test_critical_drift_severity(self, task_with_checkpoint):
        db, graph, task_id, _ = task_with_checkpoint
        result = handle_detect_drift(
            db, graph,
            task_id=task_id,
            current_goal="Building a completely new microservice for payments",
            tools_used=["web_search", "browser", "deploy"],
            actions_taken=["Wrote failing test", "Identified root cause"],  # Both in must_not_redo
            in_progress=["Building payment service"],
            violated_constraints=["Do not modify user model", "Keep backward compat"],
        )
        assert result["ok"] is True
        drift = result["drift"]
        assert drift["composite"] > 0.5
        assert drift["severity"] in ("high", "critical")


# ---- Semantic Continuity Scoring ----

class TestSemanticContinuityScoring:
    """score_recovery measures continuity after restore."""

    def test_perfect_recovery(self, task_with_checkpoint):
        db, graph, task_id, _ = task_with_checkpoint
        result = handle_score_recovery(
            db, graph,
            task_id=task_id,
            goal="Fix login bug in auth module",
            completed=["Identified root cause", "Wrote failing test"],
            in_progress=["Implementing fix"],
            must_not_redo=["Wrote failing test", "Identified root cause"],
            active_constraints=["Do not modify user model", "Keep backward compat"],
            tools_used=["read_file", "file_replace"],
        )
        assert result["ok"] is True
        score = result["score"]
        assert score["goal_alignment"] > 0.8
        assert score["constraint_retention"] > 0.8
        assert score["recovery_confidence"] > 0.7
        assert score["confidence_level"] in ("high", "medium")

    def test_poor_recovery_low_confidence(self, task_with_checkpoint):
        db, graph, task_id, _ = task_with_checkpoint
        result = handle_score_recovery(
            db, graph,
            task_id=task_id,
            goal="Something completely different",
            completed=[],
            in_progress=["Random task"],
            must_not_redo=[],
            active_constraints=[],
            tools_used=["web_search"],
        )
        assert result["ok"] is True
        score = result["score"]
        assert score["goal_alignment"] < 0.5
        assert score["constraint_retention"] < 0.5
        assert score["recovery_confidence"] < 0.5

    def test_partial_recovery(self, task_with_checkpoint):
        db, graph, task_id, _ = task_with_checkpoint
        result = handle_score_recovery(
            db, graph,
            task_id=task_id,
            goal="Fix login bug in auth module",
            completed=["Identified root cause"],  # Missing one
            in_progress=["Implementing fix"],
            must_not_redo=["Wrote failing test"],  # Missing one
            active_constraints=["Do not modify user model"],  # Missing one
            tools_used=["read_file"],
        )
        assert result["ok"] is True
        score = result["score"]
        # Partial retention
        assert 0.3 < score["constraint_retention"] < 0.9
        assert score["confidence_level"] in ("medium", "high")

    def test_retry_degradation(self, task_with_checkpoint):
        """Retry degradation increases with retry depth."""
        db, graph, task_id, execution_id = task_with_checkpoint
        from engram.handlers import handle_retry_task

        # Retry the task twice
        r1 = handle_retry_task(db, graph, task_id=task_id, reason="test retry 1")
        assert r1["ok"] is True
        retry1_id = r1["new_task_id"]
        r2 = handle_retry_task(db, graph, task_id=retry1_id, reason="test retry 2")
        assert r2["ok"] is True
        retry2_id = r2["new_task_id"]

        # Create checkpoint on the latest retry
        checkpoint.create_checkpoint(
            db, task_id=retry2_id, reason="PLAN_UPDATE",
            state={"goal": "Fix login bug", "completed": [], "in_progress": ["Fix"], "blocked": []},
            user_id="default",
        )

        result = handle_score_recovery(
            db, graph, task_id=retry2_id,
            goal="Fix login bug",
        )
        assert result["ok"] is True
        score = result["score"]
        assert score["retry_degradation"] > 0.3  # Should show degradation

    def test_no_checkpoint_returns_default(self, env):
        db, graph = env
        task_id = db.create_task(name="no ckpt", goal="test")
        result = handle_score_recovery(db, graph, task_id=task_id, goal="test")
        assert result["ok"] is True


# ---- Recovery Heuristics ----

class TestRecoveryHeuristics:
    """recommend_recovery returns lightweight heuristic recommendations."""

    def test_overflow_recommends_restore(self, task_with_checkpoint):
        db, graph, task_id, _ = task_with_checkpoint
        result = handle_recommend_recovery(
            db, graph,
            task_id=task_id,
            interruption_reason="overflow",
        )
        assert result["ok"] is True
        assert result["action"] == "restore_checkpoint"
        assert "confidence" in result

    def test_too_many_retries_recommends_abandon(self, task_with_checkpoint):
        db, graph, task_id, _ = task_with_checkpoint
        from engram.handlers import handle_retry_task

        # Create 3 retries
        current = task_id
        for i in range(3):
            r = handle_retry_task(db, graph, task_id=current, reason=f"retry {i}")
            current = r["new_task_id"]

        result = handle_recommend_recovery(db, graph, task_id=current)
        assert result["ok"] is True
        assert result["action"] == "abandon"

    def test_no_checkpoint_recommends_start_fresh(self, env):
        db, graph = env
        task_id = db.create_task(name="no ckpt", goal="test")
        result = handle_recommend_recovery(db, graph, task_id=task_id)
        assert result["ok"] is True
        assert result["action"] == "start_fresh"

    def test_invalid_reason_falls_back_to_unknown(self, task_with_checkpoint):
        db, graph, task_id, _ = task_with_checkpoint
        result = handle_recommend_recovery(
            db, graph,
            task_id=task_id,
            interruption_reason="nonexistent_reason",
        )
        assert result["ok"] is True
        assert "action" in result

    def test_repeated_tool_failure_suggests_alternative(self, task_with_checkpoint):
        db, graph, task_id, _ = task_with_checkpoint
        result = handle_recommend_recovery(
            db, graph,
            task_id=task_id,
            interruption_reason="tool_failure",
            retry_count=2,
        )
        assert result["ok"] is True
        assert result["action"] == "restore_checkpoint"
        assert "alternative" in result or "hint" in result

    def test_nonexistent_task_returns_error(self, env):
        db, graph = env
        result = handle_recommend_recovery(db, graph, task_id=99999)
        assert result["ok"] is False
