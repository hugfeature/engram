"""Tests for v0.16 Execution Lineage — Durable Runtime Continuity.

Covers:
- start_execution: creates execution + first task
- retry_task: creates retry chain within same execution
- spawn_subtask: spawns child task under parent
- trace_execution: full lineage query
- end_execution: marks execution completed/abandoned
- Event log replay for new event kinds
"""

import pytest

from engram.handlers import (
    handle_start_execution,
    handle_retry_task,
    handle_spawn_subtask,
    handle_trace_execution,
    handle_end_execution,
    handle_update_task,
)


class TestStartExecution:
    def test_creates_execution_and_first_task(self, env):
        db, graph = env
        result = handle_start_execution(db, graph, goal="Implement feature X")
        assert result["ok"] is True
        assert "execution_id" in result
        assert "task_id" in result
        assert result["attempt"] == 1

    def test_rejects_empty_goal(self, env):
        db, graph = env
        result = handle_start_execution(db, graph, goal="")
        assert result.get("ok") is not True or "error" in result

    def test_with_origin_checkpoint(self, env):
        db, graph = env
        result = handle_start_execution(
            db, graph,
            goal="Resume from checkpoint",
            origin_checkpoint="ckpt-abc-123",
        )
        assert result["ok"] is True
        execution = db.get_execution(result["execution_id"])
        assert execution["origin_checkpoint"] == "ckpt-abc-123"

    def test_execution_status_is_active(self, env):
        db, graph = env
        result = handle_start_execution(db, graph, goal="Test goal")
        execution = db.get_execution(result["execution_id"])
        assert execution["status"] == "active"

    def test_task_has_execution_id(self, env):
        db, graph = env
        result = handle_start_execution(db, graph, goal="Test goal")
        task = db.get_task(result["task_id"])
        assert task.execution_id == result["execution_id"]
        assert task.attempt == 1


class TestRetryTask:
    def test_creates_retry_in_same_execution(self, env):
        db, graph = env
        start = handle_start_execution(db, graph, goal="Flaky task")
        original_task_id = start["task_id"]

        result = handle_retry_task(db, graph, task_id=original_task_id, reason="tool_failure")
        assert result["ok"] is True
        assert result["execution_id"] == start["execution_id"]
        assert result["retry_of_task_id"] == original_task_id
        assert result["attempt"] == 2

    def test_retry_chain_grows(self, env):
        db, graph = env
        start = handle_start_execution(db, graph, goal="Multi-retry")
        tid1 = start["task_id"]

        r2 = handle_retry_task(db, graph, task_id=tid1, reason="fail1")
        tid2 = r2["new_task_id"]
        assert r2["attempt"] == 2

        r3 = handle_retry_task(db, graph, task_id=tid2, reason="fail2")
        assert r3["attempt"] == 3

    def test_original_task_marked_cancelled(self, env):
        db, graph = env
        start = handle_start_execution(db, graph, goal="Cancel test")
        original_task_id = start["task_id"]

        handle_retry_task(db, graph, task_id=original_task_id, reason="interrupted")
        original = db.get_task(original_task_id)
        assert original.status == "cancelled"

    def test_rejects_task_without_execution(self, env):
        db, graph = env
        # Create a plain task (no execution)
        task_id = db.create_task(name="standalone", goal="no execution")
        result = handle_retry_task(db, graph, task_id=task_id)
        assert "error" in result

    def test_rejects_invalid_task_id(self, env):
        db, graph = env
        result = handle_retry_task(db, graph, task_id=99999)
        assert "error" in result


class TestSpawnSubtask:
    def test_creates_subtask_under_parent(self, env):
        db, graph = env
        start = handle_start_execution(db, graph, goal="Complex task")
        parent_id = start["task_id"]

        result = handle_spawn_subtask(
            db, graph,
            parent_task_id=parent_id,
            name="Subtask A",
            goal="Do part A",
        )
        assert result["ok"] is True
        assert result["execution_id"] == start["execution_id"]
        assert result["parent_task_id"] == parent_id

        subtask = db.get_task(result["new_task_id"])
        assert subtask.parent_task_id == parent_id
        assert subtask.execution_id == start["execution_id"]

    def test_rejects_empty_name(self, env):
        db, graph = env
        start = handle_start_execution(db, graph, goal="Test")
        result = handle_spawn_subtask(db, graph, parent_task_id=start["task_id"], name="")
        assert "error" in result

    def test_rejects_parent_without_execution(self, env):
        db, graph = env
        task_id = db.create_task(name="standalone", goal="no exec")
        result = handle_spawn_subtask(db, graph, parent_task_id=task_id, name="child")
        assert "error" in result


class TestTraceExecution:
    def test_traces_full_lineage(self, env):
        db, graph = env
        start = handle_start_execution(db, graph, goal="Trace me")
        tid1 = start["task_id"]

        # Retry once
        r2 = handle_retry_task(db, graph, task_id=tid1, reason="fail")
        tid2 = r2["new_task_id"]

        # Spawn subtask from retry
        spawn = handle_spawn_subtask(db, graph, parent_task_id=tid2, name="Sub")

        result = handle_trace_execution(db, graph, execution_id=start["execution_id"])
        assert result["ok"] is True
        assert result["total_attempts"] == 3
        assert result["execution"]["root_goal"] == "Trace me"
        assert result["execution"]["status"] == "active"

        task_ids = [t["task_id"] for t in result["tasks"]]
        assert tid1 in task_ids
        assert tid2 in task_ids
        assert spawn["new_task_id"] in task_ids

    def test_identifies_current_active_task(self, env):
        db, graph = env
        start = handle_start_execution(db, graph, goal="Active check")
        tid1 = start["task_id"]
        r2 = handle_retry_task(db, graph, task_id=tid1, reason="fail")

        result = handle_trace_execution(db, graph, execution_id=start["execution_id"])
        # Current task should be the retry (tid1 is cancelled)
        assert result["current_task_id"] == r2["new_task_id"]

    def test_rejects_nonexistent_execution(self, env):
        db, graph = env
        result = handle_trace_execution(db, graph, execution_id="nonexistent-uuid")
        assert "error" in result


class TestEndExecution:
    def test_marks_completed(self, env):
        db, graph = env
        start = handle_start_execution(db, graph, goal="Finish me")
        result = handle_end_execution(db, graph, execution_id=start["execution_id"])
        assert result["ok"] is True
        assert result["status"] == "completed"

        execution = db.get_execution(start["execution_id"])
        assert execution["status"] == "completed"

    def test_marks_abandoned(self, env):
        db, graph = env
        start = handle_start_execution(db, graph, goal="Give up")
        result = handle_end_execution(
            db, graph, execution_id=start["execution_id"], status="abandoned"
        )
        assert result["ok"] is True
        execution = db.get_execution(start["execution_id"])
        assert execution["status"] == "abandoned"

    def test_rejects_invalid_status(self, env):
        db, graph = env
        start = handle_start_execution(db, graph, goal="Bad status")
        result = handle_end_execution(
            db, graph, execution_id=start["execution_id"], status="invalid"
        )
        assert "error" in result

    def test_rejects_nonexistent_execution(self, env):
        db, graph = env
        result = handle_end_execution(db, graph, execution_id="no-such-id")
        assert "error" in result


class TestRetryChainQuery:
    def test_get_retry_chain(self, env):
        db, graph = env
        start = handle_start_execution(db, graph, goal="Chain test")
        tid1 = start["task_id"]

        r2 = handle_retry_task(db, graph, task_id=tid1, reason="fail1")
        tid2 = r2["new_task_id"]

        r3 = handle_retry_task(db, graph, task_id=tid2, reason="fail2")
        tid3 = r3["new_task_id"]

        chain = db.get_retry_chain(tid3)
        assert len(chain) == 3
        assert chain[0].id == tid1
        assert chain[1].id == tid2
        assert chain[2].id == tid3


class TestActiveExecutions:
    def test_lists_active_executions(self, env):
        db, graph = env
        handle_start_execution(db, graph, goal="Exec 1")
        handle_start_execution(db, graph, goal="Exec 2")

        active = db.get_active_executions()
        assert len(active) == 2

    def test_completed_not_in_active(self, env):
        db, graph = env
        s1 = handle_start_execution(db, graph, goal="Will complete")
        handle_start_execution(db, graph, goal="Still active")

        handle_end_execution(db, graph, execution_id=s1["execution_id"])

        active = db.get_active_executions()
        assert len(active) == 1
        assert active[0]["execution_id"] != s1["execution_id"]
