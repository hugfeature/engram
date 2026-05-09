"""Tests for `engram recover`: replay event log into a fresh DB.

The point of this test file is to verify the architectural promise:
> Even if the DuckDB file is destroyed, every Tier 1 event written through
> the runtime can be replayed back into a working DB with identical state.
"""

from __future__ import annotations

import os

import pytest

from engram.db import MemoryDB
from engram.event_log import EventLog, reset_event_log_for_tests
from engram.recover import recover, doctor

FAKE_EMBED = [0.1] * 768


@pytest.fixture
def isolated_event_log(tmp_path, monkeypatch):
    """Wire MemoryDB.write paths to a per-test event log directory."""
    event_dir = tmp_path / "events"
    event_dir.mkdir()
    log = reset_event_log_for_tests(event_dir=str(event_dir))
    yield log
    # Reset back to default singleton so other tests don't see this fixture.
    reset_event_log_for_tests()


def _open_db(tmp_path, name: str) -> MemoryDB:
    return MemoryDB(str(tmp_path / name), dim=768)


def test_recover_rebuilds_tasks_and_sessions(tmp_path, isolated_event_log):
    # 1. Original runtime writes some Tier 1 state.
    db = _open_db(tmp_path, "live.duckdb")
    tid = db.create_task("ship v0.10", goal="durability", user_id="u")
    db.update_task(tid, status="in_progress")
    db.upsert_session("sess-1", user_id="u")
    db.log_session_outcome("sess-1", "success", user_id="u")
    db.end_session("sess-1", end_type="handoff")
    db.checkpoint()
    db.close()

    # 2. Disaster: pretend the DB file is gone.
    os.remove(str(tmp_path / "live.duckdb"))

    # 3. Recover into a fresh location (dry-run).
    out_dir = tmp_path / "recovered"
    report = recover(
        event_dir=isolated_event_log._dir,
        output_dir=str(out_dir),
        promote=False,
    )
    assert report.errors == []
    assert report.counts.get("task.create") == 1
    assert report.counts.get("task.update") == 1
    assert report.counts.get("session.start") == 1
    assert report.counts.get("session.outcome") == 1
    assert report.counts.get("session.end") == 1

    # 4. The recovered DB has the same logical state.
    rebuilt = MemoryDB(report.output_db, dim=768, log_writes=False)
    try:
        task = rebuilt.get_task(tid)
        assert task is not None
        assert task.name == "ship v0.10"
        assert task.status == "in_progress"
        sess = rebuilt._fetchone_dict(
            "SELECT session_id, end_type FROM session_lifecycle WHERE session_id = ?",
            ["sess-1"],
        )
        assert sess["end_type"] == "handoff"
        outcomes = rebuilt._fetchall_dicts(
            "SELECT outcome FROM session_outcome_log WHERE session_id = ?",
            ["sess-1"],
        )
        assert [o["outcome"] for o in outcomes] == ["success"]
    finally:
        rebuilt.close()


def test_recover_rebuilds_checkpoints_with_full_state(tmp_path, isolated_event_log):
    from engram import checkpoint as ckpt_mod

    db = _open_db(tmp_path, "live.duckdb")
    tid = db.create_task("with-checkpoints", user_id="u")
    state = {
        "goal": "make recover work",
        "completed": ["design", "code"],
        "in_progress": ["test"],
        "blocked": [],
        "preferred_next": ["ship"],
        "must_not_redo": [],
        "must_preserve": ["main branch"],
        "working_set": {"files": ["recover.py"]},
    }
    written = ckpt_mod.create_checkpoint(
        db,
        task_id=tid,
        state=state,
        reason=ckpt_mod.REASON_MANUAL_HANDOFF,
        user_id="u",
        source_session_id="sess-cp",
    )
    db.checkpoint()
    db.close()

    os.remove(str(tmp_path / "live.duckdb"))
    report = recover(
        event_dir=isolated_event_log._dir,
        output_dir=str(tmp_path / "recovered"),
        promote=False,
    )
    assert report.errors == []
    assert report.counts.get("checkpoint.write") == 1

    rebuilt = MemoryDB(report.output_db, dim=768, log_writes=False)
    try:
        row = rebuilt._fetchone_dict(
            "SELECT goal, completed, in_progress, must_preserve, version "
            "FROM checkpoints WHERE task_id = ? AND version = ?",
            [tid, written["version"]],
        )
        assert row["goal"] == "make recover work"
        assert "design" in row["completed"]
        assert "test" in row["in_progress"]
        assert "main branch" in row["must_preserve"]
        # tasks summary cache also rebuilt.
        task_cache = rebuilt._fetchone_dict(
            "SELECT latest_checkpoint_version, checkpoint_count FROM tasks WHERE id = ?",
            [tid],
        )
        assert task_cache["latest_checkpoint_version"] == written["version"]
        assert task_cache["checkpoint_count"] >= 1
    finally:
        rebuilt.close()


def test_recover_dry_run_does_not_touch_target(tmp_path, isolated_event_log):
    db = _open_db(tmp_path, "live.duckdb")
    tid = db.create_task("keep-original", user_id="u")
    db.checkpoint()
    db.close()

    target = str(tmp_path / "live.duckdb")
    original_size = os.path.getsize(target)

    report = recover(
        event_dir=isolated_event_log._dir,
        output_dir=str(tmp_path / "recovered"),
        promote=False,  # default
        target_db=target,
    )
    assert report.promoted is False
    assert report.backup_path is None
    # Target file untouched.
    assert os.path.exists(target)
    assert os.path.getsize(target) == original_size
    # Recovered DB must be a separate file.
    assert os.path.exists(report.output_db)
    assert os.path.dirname(report.output_db) != os.path.dirname(target)


def test_recover_promote_backs_up_original(tmp_path, isolated_event_log):
    db = _open_db(tmp_path, "live.duckdb")
    db.create_task("about-to-be-replaced", user_id="u")
    db.checkpoint()
    db.close()

    target = str(tmp_path / "live.duckdb")
    report = recover(
        event_dir=isolated_event_log._dir,
        output_dir=str(tmp_path / "recovered"),
        promote=True,
        target_db=target,
    )
    assert report.promoted is True
    assert report.backup_path is not None
    assert os.path.exists(report.backup_path), "original must be preserved"
    assert os.path.exists(target), "promoted file must occupy the target path"


def test_doctor_reports_event_kinds_and_meta(tmp_path, isolated_event_log):
    db = _open_db(tmp_path, "live.duckdb")
    db.create_task("doctor-task", user_id="u")
    db.upsert_session("sess-doc", user_id="u")
    db.checkpoint()
    db.close()

    info = doctor(
        db_path=str(tmp_path / "live.duckdb"),
        event_dir=isolated_event_log._dir,
    )
    assert info["db_exists"] is True
    assert info["readonly"] is False
    assert info["event_kinds"].get("task.create") == 1
    assert info["event_kinds"].get("session.start") == 1
    assert info["event_max_seq"] >= 2
    assert info["meta"]["engram_version"]
    assert info["counts"]["tasks"] == 1
