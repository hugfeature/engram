"""Tests for v0.18 SQLite Tier 2 Runtime State Store.

Validates:
1. RuntimeStateStore CRUD operations
2. MemoryDB proxy to SQLite when ENGRAM_SQLITE_TIER2=1
3. Auto-migration from DuckDB to SQLite
4. Checkpoint read/write through SQLite path
"""

import os
import json
import pytest

from engram.projection import RuntimeStateStore, migrate_from_duckdb
from engram.db import MemoryDB
from engram.checkpoint import (
    create_checkpoint, get_checkpoint, list_checkpoints, build_continuation,
    REASON_MANUAL_HANDOFF, REASON_PLAN_UPDATE, REASON_AUTO_SAVE,
)


@pytest.fixture
def store(tmp_path):
    s = RuntimeStateStore(str(tmp_path / "test.state.sqlite"))
    yield s
    s.close()


@pytest.fixture
def sqlite_db(tmp_path, monkeypatch):
    """MemoryDB with SQLite Tier 2 enabled."""
    monkeypatch.setenv("ENGRAM_SQLITE_TIER2", "1")
    db = MemoryDB(str(tmp_path / "test.duckdb"), dim=768)
    yield db


# ============================================================
# RuntimeStateStore unit tests
# ============================================================

class TestRuntimeStateStore:
    def test_create_and_get_task(self, store):
        tid = store.create_task(name="test", goal="verify")
        task = store.get_task(tid)
        assert task is not None
        assert task.name == "test"
        assert task.goal == "verify"
        assert task.status == "in_progress"

    def test_update_task(self, store):
        tid = store.create_task(name="t1", goal="g1")
        store.update_task(tid, status="done", goal="updated")
        task = store.get_task(tid)
        assert task.status == "done"
        assert task.goal == "updated"

    def test_list_tasks(self, store):
        store.create_task(name="a", status="in_progress")
        store.create_task(name="b", status="done")
        all_tasks = store.list_tasks()
        assert len(all_tasks) == 2
        done_tasks = store.list_tasks(status="done")
        assert len(done_tasks) == 1
        assert done_tasks[0].name == "b"

    def test_execution_lifecycle(self, store):
        store.create_execution("exec-1", root_goal="test goal")
        ex = store.get_execution("exec-1")
        assert ex["root_goal"] == "test goal"
        assert ex["status"] == "active"

        store.end_execution("exec-1", status="completed")
        ex2 = store.get_execution("exec-1")
        assert ex2["status"] == "completed"

    def test_execution_tasks(self, store):
        store.create_execution("exec-2", root_goal="multi-task")
        tid1 = store.create_task_in_execution(
            name="step1", goal="do step1", execution_id="exec-2"
        )
        tid2 = store.create_task_in_execution(
            name="step2", goal="do step2", execution_id="exec-2",
            previous_task_id=tid1, attempt=2,
        )
        tasks = store.get_execution_tasks("exec-2")
        assert len(tasks) == 2
        assert tasks[0].name == "step1"
        assert tasks[1].previous_task_id == tid1

    def test_checkpoint_insert_and_get(self, store):
        tid = store.create_task(name="ckpt-test", goal="verify checkpoint")
        ckpt_id = store.insert_checkpoint(
            task_id=tid, version=1, kind="auto",
            checkpoint_reason="PLAN_UPDATE",
            goal="verify checkpoint",
            completed=["step1"],
            in_progress=["step2"],
            continuation_confidence=0.85,
            confidence_breakdown={"state_completeness": 0.8},
            user_id="default",
        )
        assert ckpt_id > 0

        ckpt = store.get_latest_checkpoint(tid)
        assert ckpt is not None
        assert ckpt["version"] == 1
        assert ckpt["state"]["goal"] == "verify checkpoint"
        assert ckpt["state"]["completed"] == ["step1"]
        assert ckpt["continuation_confidence"] == 0.85

    def test_checkpoint_versioning(self, store):
        tid = store.create_task(name="versioned")
        store.insert_checkpoint(
            task_id=tid, version=1, kind="auto",
            checkpoint_reason="PLAN_UPDATE", user_id="default",
            goal="v1",
        )
        store.insert_checkpoint(
            task_id=tid, version=2, kind="handoff",
            checkpoint_reason="MANUAL_HANDOFF", user_id="default",
            goal="v2",
        )
        latest = store.get_latest_checkpoint(tid)
        assert latest["version"] == 2
        assert latest["state"]["goal"] == "v2"

        v1 = store.get_checkpoint_by_version(tid, 1)
        assert v1["state"]["goal"] == "v1"

    def test_session_lifecycle(self, store):
        store.start_session("sess-1")
        sessions = store.get_recent_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "sess-1"

        store.end_session("sess-1", end_type="interrupted",
                          interruption_reason="overflow")
        sessions2 = store.get_recent_sessions()
        assert sessions2[0]["end_type"] == "interrupted"

    def test_task_checkpoint_cache(self, store):
        tid = store.create_task(name="cache-test")
        store.update_task_checkpoint_cache(tid, 1)
        store.update_task_checkpoint_cache(tid, 2)
        ver, count = store.get_task_checkpoint_cache(tid)
        assert ver == 2
        assert count == 2

    def test_is_empty(self, store):
        assert store.is_empty()
        store.create_task(name="non-empty")
        assert not store.is_empty()


# ============================================================
# MemoryDB with SQLite Tier 2 proxy tests
# ============================================================

class TestMemoryDBSQLiteProxy:
    def test_state_store_initialized(self, sqlite_db):
        assert sqlite_db._state_store is not None

    def test_create_and_get_task(self, sqlite_db):
        tid = sqlite_db.create_task(name="proxy-test", goal="verify proxy")
        task = sqlite_db.get_task(tid)
        assert task is not None
        assert task.name == "proxy-test"

    def test_update_task(self, sqlite_db):
        tid = sqlite_db.create_task(name="update-test")
        sqlite_db.update_task(tid, status="done")
        task = sqlite_db.get_task(tid)
        assert task.status == "done"

    def test_list_tasks(self, sqlite_db):
        sqlite_db.create_task(name="a")
        sqlite_db.create_task(name="b")
        tasks = sqlite_db.list_tasks()
        assert len(tasks) == 2

    def test_execution_through_proxy(self, sqlite_db):
        eid = sqlite_db.create_execution("exec-proxy", root_goal="proxy test")
        assert eid == "exec-proxy"
        ex = sqlite_db.get_execution("exec-proxy")
        assert ex["root_goal"] == "proxy test"

        tid = sqlite_db.create_task_in_execution(
            name="sub", goal="sub goal", execution_id="exec-proxy"
        )
        tasks = sqlite_db.get_execution_tasks("exec-proxy")
        assert len(tasks) == 1

        sqlite_db.end_execution("exec-proxy")
        ex2 = sqlite_db.get_execution("exec-proxy")
        assert ex2["status"] == "completed"

    def test_checkpoint_through_proxy(self, sqlite_db):
        tid = sqlite_db.create_task(name="ckpt-proxy")
        result = create_checkpoint(
            sqlite_db, tid, REASON_MANUAL_HANDOFF,
            {"goal": "test checkpoint", "completed": ["step1"]},
        )
        assert result["version"] == 1
        assert result["continuation_confidence"] > 0

        ckpt = get_checkpoint(sqlite_db, tid)
        assert ckpt is not None
        assert ckpt["state"]["goal"] == "test checkpoint"
        assert ckpt["state"]["completed"] == ["step1"]

    def test_checkpoint_versioning_through_proxy(self, sqlite_db):
        tid = sqlite_db.create_task(name="multi-ckpt")
        create_checkpoint(sqlite_db, tid, REASON_PLAN_UPDATE, {"in_progress": ["s1"]})
        create_checkpoint(sqlite_db, tid, REASON_PLAN_UPDATE, {"in_progress": ["s2"]})
        create_checkpoint(sqlite_db, tid, REASON_AUTO_SAVE, {"in_progress": ["s3"]})

        history = list_checkpoints(sqlite_db, tid)
        assert len(history) == 3
        assert history[0]["version"] == 3  # DESC order

    def test_build_continuation_through_proxy(self, sqlite_db):
        tid = sqlite_db.create_task(name="continuation-test")
        create_checkpoint(sqlite_db, tid, REASON_MANUAL_HANDOFF, {
            "goal": "fix bug",
            "completed": ["investigate"],
            "in_progress": ["implement fix"],
            "preferred_next": ["write tests"],
            "must_not_redo": [{"action": "investigate", "reason": "already_completed"}],
        })
        ckpt = get_checkpoint(sqlite_db, tid)
        cont = build_continuation(ckpt, db=sqlite_db)
        assert cont["goal"] == "fix bug"
        assert cont["completed"] == ["investigate"]
        assert cont["in_progress"] == ["implement fix"]
        assert cont["continuation_confidence"] > 0

    def test_session_through_proxy(self, sqlite_db):
        sqlite_db.upsert_session("test-sess")
        sqlite_db.end_session("test-sess", end_type="interrupted",
                              interruption_reason="overflow")
        # No error = success (session operations are fire-and-forget)


# ============================================================
# Auto-migration tests
# ============================================================

class TestAutoMigration:
    def test_migrate_tasks(self, tmp_path):
        """DuckDB tasks are migrated to SQLite on first enable."""
        # 1. Create data in DuckDB (without SQLite)
        db = MemoryDB(str(tmp_path / "migrate.duckdb"), dim=768)
        tid = db.create_task(name="migrate-me", goal="test migration")
        db.update_task(tid, status="done")

        # 2. Create empty SQLite store and migrate
        store = RuntimeStateStore(str(tmp_path / "migrate.state.sqlite"))
        result = migrate_from_duckdb(db.conn, store)

        assert not result["skipped"]
        assert result["counts"]["tasks"] >= 1

        # 3. Verify data in SQLite
        task = store.get_task(tid)
        assert task is not None
        assert task.name == "migrate-me"
        assert task.status == "done"
        store.close()

    def test_migrate_skips_when_not_empty(self, tmp_path):
        """Migration is skipped if SQLite already has data."""
        db = MemoryDB(str(tmp_path / "skip.duckdb"), dim=768)
        db.create_task(name="existing")

        store = RuntimeStateStore(str(tmp_path / "skip.state.sqlite"))
        store.create_task(name="already-here")
        result = migrate_from_duckdb(db.conn, store)

        assert result["skipped"]
        store.close()

    def test_migrate_checkpoints(self, tmp_path):
        """Checkpoints are migrated from DuckDB to SQLite."""
        db = MemoryDB(str(tmp_path / "ckpt-mig.duckdb"), dim=768)
        tid = db.create_task(name="ckpt-task")
        create_checkpoint(db, tid, REASON_MANUAL_HANDOFF, {
            "goal": "migrate checkpoint",
            "completed": ["step1"],
        })

        store = RuntimeStateStore(str(tmp_path / "ckpt-mig.state.sqlite"))
        result = migrate_from_duckdb(db.conn, store)

        assert result["counts"]["checkpoints"] >= 1
        ckpt = store.get_latest_checkpoint(tid)
        assert ckpt is not None
        assert ckpt["state"]["goal"] == "migrate checkpoint"
        store.close()

    def test_auto_migrate_on_init(self, tmp_path, monkeypatch):
        """MemoryDB auto-migrates when ENGRAM_SQLITE_TIER2=1 and SQLite is empty."""
        # 1. Create data without SQLite
        db1 = MemoryDB(str(tmp_path / "auto.duckdb"), dim=768)
        tid = db1.create_task(name="auto-migrate")
        create_checkpoint(db1, tid, REASON_AUTO_SAVE, {"goal": "auto"})
        db1.conn.close()

        # 2. Re-open with SQLite enabled
        monkeypatch.setenv("ENGRAM_SQLITE_TIER2", "1")
        db2 = MemoryDB(str(tmp_path / "auto.duckdb"), dim=768)

        assert db2._state_store is not None
        task = db2.get_task(tid)
        assert task is not None
        assert task.name == "auto-migrate"

        ckpt = get_checkpoint(db2, tid)
        assert ckpt is not None
        assert ckpt["state"]["goal"] == "auto"
