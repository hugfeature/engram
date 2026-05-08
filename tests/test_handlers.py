"""Unit tests for handlers.py — pure business logic, no transport."""

import pytest

from engram.db import MemoryDB
from engram.graph import MemoryGraph
from engram.handlers import (
    handle_recall, handle_store, handle_update,
    handle_session_handoff, handle_consolidate, handle_stats,
    handle_session_outcome,
    handle_track_failure, handle_track_progress,
    handle_create_task, handle_update_task, handle_get_task, handle_list_tasks,
)


@pytest.fixture
def env(tmp_path, monkeypatch):
    db = MemoryDB(str(tmp_path / "h.duckdb"), dim=768)
    graph = MemoryGraph(str(tmp_path / "h.json"))
    monkeypatch.setattr("engram.handlers.embed", lambda t: [0.1] * 768)
    monkeypatch.setattr("engram.retrieve.embed", lambda t: [0.1] * 768)
    return db, graph


class TestHandleStore:
    def test_store_new(self, env):
        db, graph = env
        result = handle_store(db, graph, content="hello world", importance=0.7)
        assert "memory_id" in result
        assert "Stored new memory" in result["result"]

    def test_store_empty_rejected(self, env):
        db, graph = env
        result = handle_store(db, graph, content="", importance=0.5)
        assert "error" in result

    def test_store_clamps_importance(self, env):
        db, graph = env
        result = handle_store(db, graph, content="test", importance=5.0)
        assert "memory_id" in result

    def test_store_invalid_category(self, env):
        db, graph = env
        result = handle_store(db, graph, content="test", importance=0.5, category="bogus")
        assert "memory_id" in result

    def test_store_with_metadata(self, env):
        db, graph = env
        result = handle_store(db, graph, content="meta test", importance=0.5,
                              metadata={"tag": "v"})
        row = db.get_by_id(result["memory_id"])
        assert row.metadata["tag"] == "v"


class TestHandleRecall:
    def test_recall_empty_rejected(self, env):
        db, graph = env
        result = handle_recall(db, graph, query="")
        assert "error" in result

    def test_recall_returns_stored(self, env):
        db, graph = env
        handle_store(db, graph, content="recall target", importance=0.5)
        result = handle_recall(db, graph, query="recall target")
        assert result["memoriesFound"] >= 1


class TestHandleUpdate:
    def test_update_nonexistent(self, env):
        db, graph = env
        result = handle_update(db, graph, memory_id=9999, new_content="x")
        assert "error" in result

    def test_update_existing(self, env):
        db, graph = env
        mid = db.insert("original", [0.1] * 768, 0.5, "fact", "default")
        result = handle_update(db, graph, memory_id=mid, new_content="updated")
        assert "Updated" in result["result"]
        assert db.get_by_id(mid).content == "updated"


class TestHandleSessionHandoff:
    def test_basic_handoff(self, env):
        db, graph = env
        result = handle_session_handoff(db, graph, summary="Did stuff",
                                        completed=["a"], next_steps=["b"])
        assert "memory_id" in result
        assert "recorded" in result["result"]

    def test_handoff_empty_rejected(self, env):
        db, graph = env
        result = handle_session_handoff(db, graph, summary="")
        assert "error" in result


class TestHandleConsolidate:
    def test_consolidate_empty(self, env):
        db, graph = env
        result = handle_consolidate(db, graph)
        assert "No similar" in result["result"]


class TestHandleStats:
    def test_stats_empty(self, env):
        db, graph = env
        result = handle_stats(db)
        assert result["total"] == 0

    def test_stats_with_data(self, env):
        db, graph = env
        db.insert("m1", [0.1] * 768, 0.5, "fact", "default")
        result = handle_stats(db)
        assert result["total"] == 1
        assert "fact" in result["categories"]


class TestHandleSessionOutcome:
    def test_outcome_success(self, env):
        db, graph = env
        mid = db.insert("test memory", [0.1] * 768, 0.5, "fact", "default")
        db.log_session_recall("sess_1", [mid], "default")
        result = handle_session_outcome(db, graph, session_id="sess_1", outcome="success")
        assert "recorded" in result["result"]
        assert result["outcome"] == "success"
        assert result["memories_adjusted"] > 0
        # Importance should have increased
        m = db.get_by_id(mid)
        assert m.importance > 0.5

    def test_outcome_failure(self, env):
        db, graph = env
        mid = db.insert("test memory", [0.1] * 768, 0.5, "fact", "default")
        db.log_session_recall("sess_2", [mid], "default")
        result = handle_session_outcome(db, graph, session_id="sess_2",
                                        outcome="failure", notes="missing context")
        assert "recorded" in result["result"]
        assert result["outcome"] == "failure"
        # Importance should have decreased
        m = db.get_by_id(mid)
        assert m.importance < 0.5
        # Failure lesson should be stored
        all_memories = db.get_all("default")
        lesson = [m for m in all_memories if "missing context" in m.content]
        assert len(lesson) >= 1

    def test_outcome_empty_session_id_rejected(self, env):
        db, graph = env
        result = handle_session_outcome(db, graph, session_id="", outcome="success")
        assert "error" in result

    def test_outcome_invalid_outcome(self, env):
        db, graph = env
        result = handle_session_outcome(db, graph, session_id="s1", outcome="bogus")
        assert "error" in result

    def test_outcome_no_memories_found(self, env):
        db, graph = env
        result = handle_session_outcome(db, graph, session_id="empty_sess", outcome="success")
        assert result["memories_adjusted"] == 0


class TestHandleRecallSessionLog:
    def test_recall_with_session_id_logs(self, env):
        db, graph = env
        handle_store(db, graph, content="session test memory", importance=0.5)
        result = handle_recall(db, graph, query="session test", session_id="sess_log")
        if result["memoriesFound"] > 0:
            ids = db.get_session_memories("sess_log", "default")
            assert len(ids) > 0

    def test_recall_without_session_id_no_log(self, env):
        db, graph = env
        handle_store(db, graph, content="no session test", importance=0.5)
        handle_recall(db, graph, query="no session test")
        ids = db.get_session_memories("nonexistent", "default")
        assert ids == []


class TestHandleCreateTask:
    def test_create_basic(self, env):
        db, graph = env
        result = handle_create_task(db, graph, name="test task", goal="do stuff")
        assert "task_id" in result
        assert "Task created" in result["result"]

    def test_create_empty_name_rejected(self, env):
        db, graph = env
        result = handle_create_task(db, graph, name="")
        assert "error" in result

    def test_create_invalid_status_rejected(self, env):
        db, graph = env
        result = handle_create_task(db, graph, name="t", status="bogus")
        assert "error" in result

    def test_create_with_metadata(self, env):
        db, graph = env
        result = handle_create_task(db, graph, name="meta task", metadata={"priority": "high"})
        task = db.get_task(result["task_id"])
        assert task.metadata["priority"] == "high"


class TestHandleUpdateTask:
    def test_update_status(self, env):
        db, graph = env
        tid = handle_create_task(db, graph, name="update me")["task_id"]
        result = handle_update_task(db, graph, task_id=tid, status="in_progress")
        assert "Task updated" in result["result"]
        task = db.get_task(tid)
        assert task.status == "in_progress"

    def test_update_nonexistent_rejected(self, env):
        db, graph = env
        result = handle_update_task(db, graph, task_id=9999)
        assert "error" in result

    def test_update_wrong_user_rejected(self, env):
        db, graph = env
        tid = handle_create_task(db, graph, name="alice task", user_id="alice")["task_id"]
        result = handle_update_task(db, graph, task_id=tid, status="done", user_id="bob")
        assert "error" in result

    def test_update_invalid_task_id(self, env):
        db, graph = env
        result = handle_update_task(db, graph, task_id="abc")
        assert "error" in result


class TestHandleGetTask:
    def test_get_basic(self, env):
        db, graph = env
        tid = handle_create_task(db, graph, name="get me")["task_id"]
        result = handle_get_task(db, graph, task_id=tid)
        assert result["task"]["name"] == "get me"
        assert result["total_memories"] == 0

    def test_get_with_associated_memories(self, env):
        db, graph = env
        tid = handle_create_task(db, graph, name="with memories")["task_id"]
        handle_track_failure(db, graph, error="err", component="comp", task_id=tid)
        handle_track_progress(db, graph, feature="feat", status="in_progress", task_id=tid)
        result = handle_get_task(db, graph, task_id=tid)
        assert result["total_memories"] == 2
        assert len(result["failures"]) == 1
        assert len(result["progress"]) == 1

    def test_get_nonexistent_rejected(self, env):
        db, graph = env
        result = handle_get_task(db, graph, task_id=9999)
        assert "error" in result

    def test_get_wrong_user_rejected(self, env):
        db, graph = env
        tid = handle_create_task(db, graph, name="alice task", user_id="alice")["task_id"]
        result = handle_get_task(db, graph, task_id=tid, user_id="bob")
        assert "error" in result


class TestHandleListTasks:
    def test_list_empty(self, env):
        db, graph = env
        result = handle_list_tasks(db, graph)
        assert result["total"] == 0
        assert result["tasks"] == []

    def test_list_with_tasks(self, env):
        db, graph = env
        handle_create_task(db, graph, name="task 1")
        handle_create_task(db, graph, name="task 2", status="in_progress")
        result = handle_list_tasks(db, graph)
        assert result["total"] == 2

    def test_list_filter_by_status(self, env):
        db, graph = env
        handle_create_task(db, graph, name="planning task")
        handle_create_task(db, graph, name="active task", status="in_progress")
        result = handle_list_tasks(db, graph, status="in_progress")
        assert result["total"] == 1
        assert result["tasks"][0]["name"] == "active task"


class TestConsolidatorProtectsStructured:
    """Consolidator should not merge handoff/failure/progress memories."""

    def test_handoff_not_consolidated(self, env):
        db, graph = env
        handle_session_handoff(db, graph, summary="handoff A",
                               completed=["x"], next_steps=["y"])
        handle_session_handoff(db, graph, summary="handoff A",
                               completed=["x"], next_steps=["y"])
        from engram.consolidator import run_consolidate
        results = run_consolidate(db, graph)
        # Both handoffs should survive (not merged)
        all_mems = db.get_all("default")
        handoffs = [m for m in all_mems if (m.metadata or {}).get("type") == "handoff"]
        assert len(handoffs) == 2

    def test_failure_not_consolidated(self, env):
        db, graph = env
        handle_track_failure(db, graph, error="same error", component="comp")
        handle_track_failure(db, graph, error="same error", component="comp")
        from engram.consolidator import run_consolidate
        results = run_consolidate(db, graph)
        all_mems = db.get_all("default")
        failures = [m for m in all_mems if (m.metadata or {}).get("type") == "failure"]
        assert len(failures) == 2
