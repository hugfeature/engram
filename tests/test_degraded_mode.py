"""Tests for readonly degraded mode and the no-silent-fresh-DB contract."""

from __future__ import annotations

import os

import pytest

from engram.db import (
    MemoryDB,
    DegradedModeError,
    DatabaseCorruptionError,
    _connect_with_retry,
    _scan_residue,
    ENV_ALLOW_RESET,
)

FAKE_EMBED = [0.1] * 768


def _make_db(tmp_path) -> MemoryDB:
    return MemoryDB(str(tmp_path / "test.duckdb"), dim=768, log_writes=False)


def test_writes_blocked_in_readonly_mode(tmp_path):
    db = _make_db(tmp_path)
    db.enter_degraded_mode("test forced")
    assert db.readonly is True

    with pytest.raises(DegradedModeError) as info:
        db.create_task("hello", goal="g", user_id="u")
    assert "test forced" in str(info.value)
    assert info.value.recover_command == "engram recover"

    with pytest.raises(DegradedModeError):
        db.upsert_session("s1", user_id="u")
    with pytest.raises(DegradedModeError):
        db.log_session_outcome("s1", "success", user_id="u")
    with pytest.raises(DegradedModeError):
        db.insert("hi", FAKE_EMBED, user_id="u")
    with pytest.raises(DegradedModeError):
        db.delete(1)


def test_reads_still_work_in_readonly_mode(tmp_path):
    db = _make_db(tmp_path)
    # Write something while writable.
    tid = db.create_task("alive", user_id="u")
    db.enter_degraded_mode("simulated post-corruption")

    # Reads must keep working — degraded mode is "read-only", not "no-op".
    row = db.get_task(tid)
    assert row is not None
    assert row.name == "alive"
    assert db.list_tasks(user_id="u")[0].id == tid


def test_corrupt_db_raises_instead_of_silent_reset(tmp_path):
    """The old code would silently rename to .corrupt and create a fresh DB.
    The new contract MUST raise DatabaseCorruptionError unless ENV_ALLOW_RESET=1."""
    db_path = str(tmp_path / "broken.duckdb")
    # Write garbage that is definitely not a DuckDB file.
    with open(db_path, "wb") as f:
        f.write(b"not a duckdb file at all, just random bytes for the test")

    # Make sure the env escape hatch is OFF.
    os.environ.pop(ENV_ALLOW_RESET, None)

    with pytest.raises(DatabaseCorruptionError) as info:
        _connect_with_retry(db_path)

    err = info.value
    assert err.recover_command == "engram recover"
    assert err.db_path == db_path
    # Original file must be isolated, not deleted.
    assert err.backup_path is not None
    assert os.path.exists(err.backup_path)
    assert not os.path.exists(db_path)  # moved aside
    # And residue scanner picks it up for the next start.
    residue = _scan_residue(db_path)
    assert any(r == err.backup_path for r in residue)


def test_env_allow_reset_restores_old_behaviour(tmp_path, monkeypatch):
    db_path = str(tmp_path / "broken.duckdb")
    with open(db_path, "wb") as f:
        f.write(b"definitely not a real duckdb file")

    monkeypatch.setenv(ENV_ALLOW_RESET, "1")
    conn = _connect_with_retry(db_path)
    # Should now be a fresh, working connection.
    conn.execute("CREATE TABLE x (id INTEGER)")
    conn.execute("INSERT INTO x VALUES (1)")
    assert conn.execute("SELECT COUNT(*) FROM x").fetchone()[0] == 1
    conn.close()


def test_engram_meta_records_runtime_identity(tmp_path):
    db = _make_db(tmp_path)
    meta = db.all_meta()
    # The contract MCP clients depend on:
    for required in (
        "schema_version",
        "engram_version",
        "duckdb_version",
        "embedding_model",
        "embedding_dim",
        "embedding_stale",
        "last_boot_at",
    ):
        assert required in meta, f"engram_meta missing key {required!r}"
    assert meta["embedding_dim"] == "768"
    assert meta["embedding_stale"] in ("0", "1")


def test_tasks_table_has_execution_graph_columns(tmp_path):
    """Schema must reserve parent_task_id / retry_of_task_id for v0.10+ even
    though the runtime doesn't read them yet — avoids a future destructive
    migration."""
    db = _make_db(tmp_path)
    rows = db.conn.execute("DESCRIBE tasks").fetchall()
    cols = {r[0] for r in rows}
    assert "parent_task_id" in cols
    assert "retry_of_task_id" in cols
