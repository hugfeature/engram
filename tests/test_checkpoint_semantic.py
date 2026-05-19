"""Tests for v0.16 Phase 2 — Checkpoint Semantic Completeness.

Covers:
- active_constraints and blocked_reasons stored in checkpoint
- execution_position derived from lineage on restore
- open_subtasks populated when subtasks exist
- build_continuation returns full semantic context
"""

import pytest

from engram.handlers import (
    handle_start_execution,
    handle_retry_task,
    handle_spawn_subtask,
    handle_restore_checkpoint,
    handle_session_handoff,
)
from engram import checkpoint


class TestActiveConstraints:
    """active_constraints persisted in checkpoint and returned on restore."""

    def test_constraints_stored_and_restored(self, env):
        db, graph = env
        start = handle_start_execution(db, graph, goal="Constrained work")
        task_id = start["task_id"]

        # Create checkpoint with active_constraints
        state = {
            "goal": "Constrained work",
            "completed": ["step 1"],
            "in_progress": ["step 2"],
            "active_constraints": [
                "must not modify production DB directly",
                "all changes require migration script",
            ],
            "blocked_reasons": [],
        }
        checkpoint.create_checkpoint(
            db, task_id, reason="PLAN_UPDATE", state=state,
        )

        # Restore and verify constraints are present
        result = handle_restore_checkpoint(db, graph, task_id=task_id)
        continuation = result["continuation"]
        assert continuation["active_constraints"] == [
            "must not modify production DB directly",
            "all changes require migration script",
        ]

    def test_empty_constraints_default(self, env):
        db, graph = env
        start = handle_start_execution(db, graph, goal="No constraints")
        task_id = start["task_id"]

        state = {"goal": "No constraints", "completed": [], "in_progress": ["work"]}
        checkpoint.create_checkpoint(db, task_id, reason="AUTO_SAVE", state=state)

        result = handle_restore_checkpoint(db, graph, task_id=task_id)
        assert result["continuation"]["active_constraints"] == []


class TestBlockedReasons:
    """blocked_reasons persisted and returned on restore."""

    def test_blocked_reasons_stored(self, env):
        db, graph = env
        start = handle_start_execution(db, graph, goal="Blocked task")
        task_id = start["task_id"]

        state = {
            "goal": "Blocked task",
            "completed": [],
            "in_progress": [],
            "blocked": ["waiting for API key"],
            "blocked_reasons": [
                {"blocker": "API key", "reason": "admin has not provisioned it yet",
                 "since": "2026-05-19"},
            ],
        }
        checkpoint.create_checkpoint(db, task_id, reason="AUTO_SAVE", state=state)

        result = handle_restore_checkpoint(db, graph, task_id=task_id)
        continuation = result["continuation"]
        assert len(continuation["blocked_reasons"]) == 1
        assert continuation["blocked_reasons"][0]["blocker"] == "API key"


class TestExecutionPosition:
    """execution_position derived from lineage on restore."""

    def test_position_for_first_attempt(self, env):
        db, graph = env
        start = handle_start_execution(db, graph, goal="Position test")
        task_id = start["task_id"]

        state = {"goal": "Position test", "in_progress": ["work"]}
        checkpoint.create_checkpoint(db, task_id, reason="AUTO_SAVE", state=state)

        result = handle_restore_checkpoint(db, graph, task_id=task_id)
        pos = result["continuation"]["execution_position"]
        assert pos["in_execution"] is True
        assert pos["execution_id"] == start["execution_id"]
        assert pos["attempt"] == 1
        assert pos["is_retry"] is False
        assert pos["is_subtask"] is False
        assert pos["open_subtasks"] == []

    def test_position_after_retry(self, env):
        db, graph = env
        start = handle_start_execution(db, graph, goal="Retry position")
        tid1 = start["task_id"]

        retry = handle_retry_task(db, graph, task_id=tid1, reason="crashed")
        tid2 = retry["new_task_id"]

        state = {"goal": "Retry position", "in_progress": ["retry work"]}
        checkpoint.create_checkpoint(db, tid2, reason="AUTO_SAVE", state=state)

        result = handle_restore_checkpoint(db, graph, task_id=tid2)
        pos = result["continuation"]["execution_position"]
        assert pos["in_execution"] is True
        assert pos["attempt"] == 2
        assert pos["is_retry"] is True
        assert pos["retry_depth"] == 2  # chain: tid1 -> tid2

    def test_position_with_open_subtasks(self, env):
        db, graph = env
        start = handle_start_execution(db, graph, goal="Parent with subtasks")
        parent_id = start["task_id"]

        # Spawn two subtasks
        s1 = handle_spawn_subtask(db, graph, parent_task_id=parent_id, name="Sub A")
        s2 = handle_spawn_subtask(db, graph, parent_task_id=parent_id, name="Sub B")

        state = {"goal": "Parent with subtasks", "in_progress": ["coordinating"]}
        checkpoint.create_checkpoint(db, parent_id, reason="AUTO_SAVE", state=state)

        result = handle_restore_checkpoint(db, graph, task_id=parent_id)
        pos = result["continuation"]["execution_position"]
        assert len(pos["open_subtasks"]) == 2
        subtask_names = [s["name"] for s in pos["open_subtasks"]]
        assert "Sub A" in subtask_names
        assert "Sub B" in subtask_names

    def test_position_without_execution(self, env):
        db, graph = env
        # Plain task without execution lineage
        task_id = db.create_task(name="standalone", goal="no exec")

        state = {"goal": "no exec", "in_progress": ["work"]}
        checkpoint.create_checkpoint(db, task_id, reason="AUTO_SAVE", state=state)

        result = handle_restore_checkpoint(db, graph, task_id=task_id)
        pos = result["continuation"]["execution_position"]
        assert pos["in_execution"] is False

    def test_position_total_attempts(self, env):
        db, graph = env
        start = handle_start_execution(db, graph, goal="Many attempts")
        tid1 = start["task_id"]

        r2 = handle_retry_task(db, graph, task_id=tid1, reason="fail1")
        r3 = handle_retry_task(db, graph, task_id=r2["new_task_id"], reason="fail2")
        tid3 = r3["new_task_id"]

        state = {"goal": "Many attempts", "in_progress": ["attempt 3"]}
        checkpoint.create_checkpoint(db, tid3, reason="AUTO_SAVE", state=state)

        result = handle_restore_checkpoint(db, graph, task_id=tid3)
        pos = result["continuation"]["execution_position"]
        assert pos["total_attempts_in_execution"] == 3
        assert pos["attempt"] == 3


class TestSemanticDiffIncludesNewFields:
    """Verify that shallow diff detects changes in active_constraints/blocked_reasons."""

    def test_diff_detects_constraint_change(self, env):
        db, graph = env
        start = handle_start_execution(db, graph, goal="Diff test")
        task_id = start["task_id"]

        state1 = {"goal": "Diff test", "in_progress": ["a"], "active_constraints": ["rule1"]}
        checkpoint.create_checkpoint(db, task_id, reason="PLAN_UPDATE", state=state1)

        state2 = {"goal": "Diff test", "in_progress": ["a"], "active_constraints": ["rule1", "rule2"]}
        ckpt = checkpoint.create_checkpoint(db, task_id, reason="PLAN_UPDATE", state=state2)

        # Verify the diff was computed
        latest = checkpoint.get_checkpoint(db, task_id)
        diff = latest["state_diff"]
        assert "active_constraints" in diff.get("changed_fields", {})
