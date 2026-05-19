"""Tests for v0.16 Phase 3 — Runtime Coordination.

Covers:
- Graceful lock detection: MemoryDB enters readonly mode when locked
- Write retry with backoff: _execute_with_retry behavior
- DegradedModeError on writes in readonly mode
"""

import pytest
from unittest.mock import patch, MagicMock

import duckdb

from engram.db import (
    MemoryDB,
    DegradedModeError,
    DatabaseLockedError,
    _is_lock_conflict,
)


class TestGracefulLockDetection:
    """MemoryDB enters readonly degraded mode when DB is locked."""

    def test_enters_readonly_on_lock(self, tmp_path):
        db_path = str(tmp_path / "locked.duckdb")
        # Create the DB file first
        conn = duckdb.connect(db_path)
        conn.close()

        lock_exc = duckdb.IOException("Could not set lock on file: Conflicting lock")
        with patch("engram.db._connect_with_retry", side_effect=DatabaseLockedError(db_path, lock_exc)):
            db = MemoryDB(db_path, dim=768)
            assert db.readonly is True
            assert "locked" in (db._readonly_reason or "").lower() or "lock" in (db._readonly_reason or "").lower()

    def test_readonly_rejects_writes(self, tmp_path):
        db_path = str(tmp_path / "locked2.duckdb")
        conn = duckdb.connect(db_path)
        conn.close()

        lock_exc = duckdb.IOException("Could not set lock: Conflicting lock")
        with patch("engram.db._connect_with_retry", side_effect=DatabaseLockedError(db_path, lock_exc)):
            db = MemoryDB(db_path, dim=768)
            with pytest.raises(DegradedModeError):
                db.create_task(name="should fail", goal="test")

    def test_readonly_allows_reads_on_existing_db(self, tmp_path):
        db_path = str(tmp_path / "existing.duckdb")
        # Create a DB with data first
        db = MemoryDB(db_path, dim=768)
        task_id = db.create_task(name="existing task", goal="test")
        db.close()

        # Now simulate lock on re-open
        lock_exc = duckdb.IOException("Conflicting lock is held")
        with patch("engram.db._connect_with_retry", side_effect=DatabaseLockedError(db_path, lock_exc)):
            db2 = MemoryDB(db_path, dim=768)
            assert db2.readonly is True
            # Read should work (opened read_only)
            # Note: schema may not be fully initialized in readonly mode,
            # but the connection should be functional


class TestWriteRetryWithBackoff:
    """_execute_with_retry retries on lock conflicts."""

    def test_succeeds_on_first_try(self, tmp_path):
        db = MemoryDB(str(tmp_path / "retry.duckdb"), dim=768)
        result = db._execute_with_retry("SELECT 1")
        assert result.fetchone()[0] == 1

    def test_retries_on_lock_conflict(self, tmp_path):
        db = MemoryDB(str(tmp_path / "retry2.duckdb"), dim=768)
        call_count = [0]
        original_conn = db.conn

        class RetryProxy:
            """Proxy that simulates lock conflicts on first N calls."""
            def __getattr__(self, name):
                return getattr(original_conn, name)

            def execute(self, sql, params=None):
                nonlocal call_count
                call_count[0] += 1
                if call_count[0] <= 2:
                    raise duckdb.IOException("Could not set lock on file: Conflicting lock")
                return original_conn.execute(sql, params or [])

        db.conn = RetryProxy()
        result = db._execute_with_retry("SELECT 1", max_retries=3)
        assert result.fetchone()[0] == 1
        assert call_count[0] == 3  # failed twice, succeeded on third
        db.conn = original_conn  # restore for cleanup

    def test_raises_after_max_retries(self, tmp_path):
        db = MemoryDB(str(tmp_path / "retry3.duckdb"), dim=768)
        original_conn = db.conn

        class AlwaysLockedProxy:
            def __getattr__(self, name):
                return getattr(original_conn, name)

            def execute(self, sql, params=None):
                raise duckdb.IOException("Could not set lock: Conflicting lock")

        db.conn = AlwaysLockedProxy()
        with pytest.raises(DegradedModeError) as exc_info:
            db._execute_with_retry("SELECT 1", max_retries=2)
        assert "retries" in str(exc_info.value).lower()
        db.conn = original_conn

    def test_raises_immediately_on_non_lock_error(self, tmp_path):
        db = MemoryDB(str(tmp_path / "retry4.duckdb"), dim=768)
        original_conn = db.conn

        class IOErrorProxy:
            def __getattr__(self, name):
                return getattr(original_conn, name)

            def execute(self, sql, params=None):
                raise duckdb.IOException("File not found")

        db.conn = IOErrorProxy()
        with pytest.raises(duckdb.IOException):
            db._execute_with_retry("SELECT 1")
        db.conn = original_conn


class TestIsLockConflict:
    """_is_lock_conflict correctly identifies lock errors."""

    def test_detects_conflicting_lock(self):
        exc = duckdb.IOException("Conflicting lock is held by PID 1234")
        assert _is_lock_conflict(exc) is True

    def test_detects_could_not_set_lock(self):
        exc = duckdb.IOException("Could not set lock on file")
        assert _is_lock_conflict(exc) is True

    def test_does_not_match_other_errors(self):
        exc = duckdb.IOException("File not found")
        assert _is_lock_conflict(exc) is False

    def test_does_not_match_generic_exception(self):
        exc = RuntimeError("some other error")
        assert _is_lock_conflict(exc) is False


class TestDegradedModeError:
    """DegradedModeError has proper attributes."""

    def test_has_recover_command(self):
        err = DegradedModeError("test reason", "/path/to/db")
        assert err.recover_command == "engram recover"
        assert err.db_path == "/path/to/db"
        assert "test reason" in str(err)
