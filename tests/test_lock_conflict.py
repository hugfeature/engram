"""Tests for v0.11.1 hotfix: lock conflict must NOT be misclassified as DB corruption.

Regression context: in v0.10/v0.11, when two engram processes raced for the
same DuckDB file, the loser's ``_connect_with_retry`` caught DuckDB's
"Conflicting lock" IOException, walked the WAL-isolation path, and finally
renamed the (perfectly healthy) main DB to ``.corrupt.<ts>`` before failing.

The fix:
  - Recognize lock conflicts via ``_is_lock_conflict``.
  - Raise ``DatabaseLockedError`` immediately, BEFORE touching any file.

Note on test design: DuckDB's file lock is **process-level**, so two
``duckdb.connect()`` calls inside the same Python process share the same
lock and never conflict. To exercise the real conflict path we spawn a
holder subprocess that grabs the lock and stays alive while the main
process attempts the conflicting connect.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import duckdb
import pytest

from engram.db import (
    DatabaseLockedError,
    DatabaseCorruptionError,
    _connect_with_retry,
    _force_checkpoint,
    _is_lock_conflict,
    _scan_residue,
)


# ---------------------------------------------------------------------------
# _is_lock_conflict — exception classifier (pure unit, no subprocess needed)
# ---------------------------------------------------------------------------

def test_is_lock_conflict_recognises_duckdb_messages():
    # Real-world DuckDB messages we observed.
    e1 = duckdb.IOException(
        'IO Error: Could not set lock on file "/tmp/x.duckdb": '
        'Conflicting lock is held in /usr/bin/python (PID 12345)'
    )
    e2 = duckdb.IOException(
        'Could not set lock on file "/var/lib/x.duckdb"'
    )
    assert _is_lock_conflict(e1) is True
    assert _is_lock_conflict(e2) is True


def test_is_lock_conflict_rejects_unrelated_io_errors():
    # Genuine corruption / IO errors must NOT match.
    cases = [
        duckdb.IOException("Block checksum mismatch"),
        duckdb.IOException("Cannot open file: No such file or directory"),
        duckdb.IOException("Disk full"),
        RuntimeError("totally unrelated"),
    ]
    for exc in cases:
        assert _is_lock_conflict(exc) is False, f"false positive on {exc!r}"


def test_database_locked_error_is_not_corruption():
    """A locked DB is healthy. The two error types must be distinct so HTTP
    handlers / MCP dispatch can branch on the type and avoid suggesting
    `engram recover` for a file that doesn't need recovery."""
    assert not issubclass(DatabaseLockedError, DatabaseCorruptionError)
    assert not issubclass(DatabaseCorruptionError, DatabaseLockedError)


# ---------------------------------------------------------------------------
# Subprocess-based lock conflict tests
# ---------------------------------------------------------------------------

_HOLDER_SCRIPT = """
import duckdb, os, sys, time
db_path = sys.argv[1]
ready_path = sys.argv[2]
conn = duckdb.connect(db_path)
conn.execute('CREATE TABLE IF NOT EXISTS t (id INT)')
conn.execute('INSERT INTO t VALUES (1), (2), (3)')
# Signal readiness to parent.
with open(ready_path, 'w') as f:
    f.write('ready')
# Hold the lock until killed.
while True:
    time.sleep(1)
"""


@pytest.fixture
def lock_holder(tmp_path):
    """Spawn a subprocess that holds the DuckDB file lock for the test."""
    db_path = tmp_path / "live.duckdb"
    ready_path = tmp_path / ".ready"
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER_SCRIPT, str(db_path), str(ready_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait up to 10s for the holder to grab the lock.
    deadline = time.time() + 10
    while time.time() < deadline:
        if ready_path.exists():
            break
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode("utf-8", errors="replace")
            pytest.fail(f"holder subprocess exited early: {stderr}")
        time.sleep(0.05)
    else:
        proc.kill()
        pytest.fail("holder subprocess did not become ready in time")

    yield str(db_path)

    proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def test_lock_conflict_raises_clean_error_without_touching_files(lock_holder):
    """Critical regression: file state is preserved on lock conflict.

    Before v0.11.1 this test would fail because ``_connect_with_retry`` would:
      1. catch the IOException
      2. attempt WAL isolation (renaming the WAL aside)
      3. retry the connect (still locked, still fails)
      4. rename the main DB to ``.corrupt.<ts>``

    All four steps are now suppressed for lock conflicts.
    """
    db_path = lock_holder

    # Snapshot on-disk state before the conflicting connect attempt.
    before_size = os.path.getsize(db_path)
    before_residue = set(_scan_residue(db_path))
    wal_path = db_path + ".wal"
    before_wal_exists = os.path.exists(wal_path)
    before_wal_size = os.path.getsize(wal_path) if before_wal_exists else 0

    with pytest.raises(DatabaseLockedError) as info:
        _connect_with_retry(db_path)

    # Error carries the context an operator needs.
    assert info.value.db_path == db_path
    assert "locked" in str(info.value).lower()

    # Critical: NO file mutation occurred.
    assert os.path.exists(db_path), "main DB must not be renamed/moved"
    assert os.path.getsize(db_path) == before_size
    # No NEW residue files were created.
    after_residue = set(_scan_residue(db_path))
    new_residue = after_residue - before_residue
    assert not new_residue, f"lock conflict created residue: {new_residue}"
    # WAL preserved exactly as it was.
    if before_wal_exists:
        assert os.path.exists(wal_path), "WAL must not be moved aside"
        assert os.path.getsize(wal_path) == before_wal_size


def test_force_checkpoint_propagates_lock_conflict(lock_holder):
    """If the DB is locked by another process, _force_checkpoint must NOT
    return False — that would mislead ``_recover_wal`` into renaming the
    WAL. Instead it raises DatabaseLockedError so the recovery walk halts."""
    db_path = lock_holder
    if not os.path.exists(db_path + ".wal"):
        # WAL only exists if holder is mid-write; some duckdb versions
        # checkpoint before this test sees it. The other tests cover the
        # same code path via _connect_with_retry, so skip rather than fake.
        pytest.skip("no WAL present — _force_checkpoint short-circuits to True")

    with pytest.raises(DatabaseLockedError):
        _force_checkpoint(db_path)


def test_force_checkpoint_returns_true_with_no_wal(tmp_path):
    """Sanity: clean state -> nothing to do, returns True."""
    db_path = str(tmp_path / "fresh.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE t (id INT)")
    conn.close()
    assert _force_checkpoint(db_path) is True
