"""Chaos Continuity Tests — simulate real interruption scenarios and measure recovery quality.

5 scenarios:
  S1: Normal handoff → restore (baseline, should score near-perfect)
  S2: SIGTERM-style interruption (atexit fires) → restore
  S3: kill -9 style (no atexit) → classify as crash → restore
  S4: Failure mid-session → restore (verify failure recall)
  S5: Working set drift → restore (verify working set stability)
"""

import json
import pytest

from engram.db import MemoryDB, INTERRUPTION_CRASH, INTERRUPTION_TOOL_FAILURE
from engram.graph import MemoryGraph
from engram import checkpoint
from engram.continuity import evaluate, ContinuityScore
from engram.handlers import (
    handle_session_handoff,
    handle_restore_checkpoint,
    handle_track_failure,
    handle_evaluate_continuity,
)

FAKE_EMBED = [0.1] * 768


@pytest.fixture
def env(tmp_path, monkeypatch):
    db = MemoryDB(str(tmp_path / "chaos.duckdb"), dim=768)
    graph = MemoryGraph(str(tmp_path / "chaos.json"))
    monkeypatch.setattr("engram.handlers.embed", lambda t: FAKE_EMBED)
    monkeypatch.setattr("engram.retrieve.embed", lambda t: FAKE_EMBED)
    return db, graph


def _create_task_with_checkpoint(db, goal="implement feature X",
                                 in_progress=None, preferred_next=None,
                                 must_not_redo=None, working_set=None):
    """Helper: create a task + initial checkpoint (simulates Agent A's work)."""
    task_id = db.create_task("chaos-test", goal=goal, user_id="default")
    state = {
        "goal": goal,
        "completed": ["design doc"],
        "in_progress": in_progress or ["implement core logic"],
        "blocked": [],
        "preferred_next": preferred_next or ["write tests", "deploy"],
        "must_not_redo": must_not_redo or [],
        "must_preserve": ["API contract"],
        "working_set": working_set or {
            "files": ["src/main.py", "src/utils.py"],
            "tools": ["pytest"],
        },
    }
    ckpt = checkpoint.create_checkpoint(
        db, task_id, checkpoint.REASON_AUTO_SAVE, state, user_id="default",
    )
    return task_id, state, ckpt


# --- S1: Normal Handoff (baseline) ---

class TestS1NormalHandoff:
    """Agent A does a clean handoff → Agent B restores. Should be near-perfect."""

    def test_handoff_restore_high_continuity(self, env):
        db, graph = env
        task_id, original_state, _ = _create_task_with_checkpoint(db)

        # Agent A does a clean handoff (creates a second checkpoint)
        handle_session_handoff(
            db, graph,
            summary="Completed core logic, ready for tests",
            completed=["design doc", "implement core logic"],
            in_progress=["write tests"],
            next_steps=["deploy"],
            user_id="default",
            task_id=task_id,
        )

        # Agent B restores
        result = handle_restore_checkpoint(db, graph, task_id=task_id)
        assert "continuation" in result

        # Evaluate continuity between the two checkpoints
        eval_result = handle_evaluate_continuity(db, graph, task_id=task_id)
        assert eval_result["ok"] is True
        score = eval_result["continuity_score"]
        assert score["goal_retention"] == 1.0
        assert score["composite"] > 0.6


# --- S2: SIGTERM (atexit fires) ---

class TestS2SigtermInterruption:
    """Simulate SIGTERM: atexit fires, session closes with process_exit."""

    def test_sigterm_session_marked_and_restorable(self, env):
        db, graph = env
        task_id, state, _ = _create_task_with_checkpoint(db)

        # Simulate session start + atexit close
        db.upsert_session("sigterm-sess", "default")
        db.end_session("sigterm-sess", end_type="process_exit")

        # Agent B: session is closed (not interrupted), checkpoint still valid
        result = handle_restore_checkpoint(db, graph, task_id=task_id)
        assert "continuation" in result
        cont = result["continuation"]
        assert cont["goal"] == state["goal"]

    def test_sigterm_continuity_preserved(self, env):
        db, graph = env
        task_id, state, _ = _create_task_with_checkpoint(db)

        # Create second checkpoint (same state = perfect continuity)
        checkpoint.create_checkpoint(
            db, task_id, checkpoint.REASON_AUTO_SAVE, state, user_id="default",
        )

        score = evaluate(state, state, db=db, task_id=task_id)
        assert score.goal_retention == 1.0
        assert score.action_consistency == 1.0
        assert score.composite > 0.9


# --- S3: kill -9 (no atexit) ---

class TestS3CrashInterruption:
    """Simulate kill -9: session never closed, classified as crash by cleanup."""

    def test_crash_session_classified(self, env):
        db, graph = env
        task_id, state, _ = _create_task_with_checkpoint(db)

        # Simulate session start but NO end (process killed)
        db.upsert_session("crash-sess", "default")
        # Make session stale and very short (< 2 min → crash heuristic)
        db.conn.execute(
            "UPDATE session_lifecycle "
            "SET started_at = now() - INTERVAL '60 MINUTES', "
            "    last_active_at = now() - INTERVAL '60 MINUTES' "
            "WHERE session_id = 'crash-sess'"
        )

        # Cleanup should classify as crash
        db.cleanup_stale_sessions("default", stale_minutes=30)
        row = db.conn.execute(
            "SELECT interruption_reason FROM session_lifecycle "
            "WHERE session_id = 'crash-sess'"
        ).fetchone()
        assert row[0] == INTERRUPTION_CRASH

    def test_crash_checkpoint_still_restorable(self, env):
        db, graph = env
        task_id, state, _ = _create_task_with_checkpoint(db)

        # Even after crash, checkpoint is durable (Tier 1)
        result = handle_restore_checkpoint(db, graph, task_id=task_id)
        assert "continuation" in result
        assert result["continuation"]["goal"] == state["goal"]


# --- S4: Failure Mid-Session ---

class TestS4FailureMidSession:
    """Agent encounters failures → track_failure → interrupt → restore."""

    def test_failure_recalled_after_restore(self, env):
        db, graph = env

        must_not = [
            {"action": "deploy without running tests", "reason": "failed_dont_retry"},
        ]
        task_id, state, _ = _create_task_with_checkpoint(
            db, must_not_redo=must_not,
        )

        # Agent A encounters a failure and creates a FAILURE checkpoint
        failure_state = dict(state)
        failure_state["must_not_redo"] = must_not + [
            {"action": "use deprecated API", "reason": "side_effect_emitted"},
        ]
        checkpoint.create_checkpoint(
            db, task_id, checkpoint.REASON_FAILURE, failure_state,
            failure_signature="api:deprecated_call", user_id="default",
        )

        # Agent B restores and evaluates
        eval_result = handle_evaluate_continuity(db, graph, task_id=task_id)
        assert eval_result["ok"] is True
        score = eval_result["continuity_score"]
        # Original must_not_redo preserved in the new checkpoint
        assert score["failure_recall"] == 1.0
        assert score["goal_retention"] == 1.0

    def test_failure_heavy_session_classified(self, env):
        db, graph = env
        task_id, state, _ = _create_task_with_checkpoint(db)

        db.upsert_session("fail-sess", "default")
        db.conn.execute(
            "UPDATE session_lifecycle "
            "SET started_at = now() - INTERVAL '120 MINUTES', "
            "    last_active_at = now() - INTERVAL '60 MINUTES' "
            "WHERE session_id = 'fail-sess'"
        )
        # Insert failure memories within session window
        for i in range(3):
            db.insert(f"tool error #{i}", FAKE_EMBED, 0.8, "failure", "default",
                      metadata={"type": "failure", "component": "api"})
        db.conn.execute(
            "UPDATE memories SET created_at = now() - INTERVAL '90 MINUTES' "
            "WHERE category = 'failure'"
        )

        db.cleanup_stale_sessions("default", stale_minutes=30)
        row = db.conn.execute(
            "SELECT interruption_reason FROM session_lifecycle "
            "WHERE session_id = 'fail-sess'"
        ).fetchone()
        assert row[0] == INTERRUPTION_TOOL_FAILURE


# --- S5: Working Set Drift ---

class TestS5WorkingSetDrift:
    """Working set changes significantly between checkpoints → lower stability score."""

    def test_working_set_drift_detected(self, env):
        db, graph = env

        working_set_before = {
            "files": ["src/auth.py", "src/db.py", "src/api.py"],
            "tools": ["pytest", "mypy"],
        }
        task_id, state, _ = _create_task_with_checkpoint(
            db, working_set=working_set_before,
        )

        # Agent A shifts to completely different files
        drifted_state = dict(state)
        drifted_state["working_set"] = {
            "files": ["src/ui.py", "src/css.py"],
            "tools": ["webpack"],
        }
        checkpoint.create_checkpoint(
            db, task_id, checkpoint.REASON_WORKING_SET_SHIFT,
            drifted_state, user_id="default",
        )

        eval_result = handle_evaluate_continuity(db, graph, task_id=task_id)
        assert eval_result["ok"] is True
        score = eval_result["continuity_score"]
        # Working set completely changed → low stability
        assert score["working_set_stability"] < 0.2
        # But goal and actions should still be preserved
        assert score["goal_retention"] == 1.0

    def test_restore_includes_continuity_score(self, env):
        db, graph = env
        task_id, state, _ = _create_task_with_checkpoint(db)

        # Create a second checkpoint so restore has parent_version
        checkpoint.create_checkpoint(
            db, task_id, checkpoint.REASON_AUTO_SAVE,
            state, user_id="default",
        )

        result = handle_restore_checkpoint(db, graph, task_id=task_id)
        assert "continuation" in result
        # continuity_score should be attached to restore result
        assert "continuity_score" in result
        assert result["continuity_score"]["goal_retention"] == 1.0
