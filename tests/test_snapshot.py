"""Tests for P1-1: periodic snapshot + replay acceleration.

Verifies:
- _parse_seq parses our naming pattern and rejects junk
- take_snapshot returns None on missing src, None when copy fails, and a
  valid SnapshotInfo on success
- list_snapshots / latest_snapshot ignore unrelated files and order by seq
- _prune_snapshots keeps only the configured retain count
- SnapshotScheduler triggers when event count or wall clock cross threshold
- recover._try_seed_from_snapshot copies the snapshot in for full replay
  but skips it when since_date is set
- recover() with snapshot present produces the same logical state as without
"""

from __future__ import annotations

import os
import time

import duckdb
import pytest

from engram.snapshot import (
    DEFAULT_RETAIN,
    SnapshotScheduler,
    _parse_seq,
    _prune_snapshots,
    latest_snapshot,
    list_snapshots,
    take_snapshot,
)


def _seed_duckdb(path: str, rows: int = 3) -> None:
    conn = duckdb.connect(path)
    try:
        conn.execute("CREATE TABLE t (id INT)")
        for i in range(rows):
            conn.execute("INSERT INTO t VALUES (?)", [i])
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# _parse_seq
# ---------------------------------------------------------------------------

def test_parse_seq_accepts_valid_names():
    assert _parse_seq("snapshot-seq42-20260509-100000.duckdb") == 42
    assert _parse_seq("snapshot-seq0-anything.duckdb") == 0
    assert _parse_seq("snapshot-seq999999-x.duckdb") == 999999


def test_parse_seq_rejects_junk():
    assert _parse_seq("not-a-snapshot.duckdb") is None
    assert _parse_seq("snapshot-seqXXX-x.duckdb") is None
    assert _parse_seq("snapshot-seq42-x.txt") is None  # wrong suffix
    assert _parse_seq("readme.md") is None


# ---------------------------------------------------------------------------
# take_snapshot
# ---------------------------------------------------------------------------

def test_take_snapshot_no_op_when_src_missing(tmp_path):
    snap_dir = tmp_path / "snapshots"
    result = take_snapshot(
        db_path=str(tmp_path / "missing.duckdb"),
        seq=10,
        snapshot_dir=str(snap_dir),
        checkpoint_first=False,
    )
    assert result is None
    assert not snap_dir.exists()


def test_take_snapshot_copies_bytes(tmp_path):
    db_path = tmp_path / "live.duckdb"
    _seed_duckdb(str(db_path))
    original_bytes = db_path.read_bytes()
    snap_dir = tmp_path / "snapshots"

    info = take_snapshot(
        db_path=str(db_path),
        seq=42,
        snapshot_dir=str(snap_dir),
        checkpoint_first=False,
    )
    assert info is not None
    assert info.seq == 42
    assert info.size_bytes == len(original_bytes)
    assert os.path.exists(info.path)
    # Source DB must be untouched.
    assert db_path.read_bytes() == original_bytes
    # Snapshot is a real readable DuckDB file with the same data.
    conn = duckdb.connect(info.path, read_only=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 3
    finally:
        conn.close()


def test_take_snapshot_with_checkpoint_first(tmp_path):
    """checkpoint_first=True must still produce a working snapshot."""
    db_path = tmp_path / "live.duckdb"
    _seed_duckdb(str(db_path))
    snap_dir = tmp_path / "snapshots"

    info = take_snapshot(
        db_path=str(db_path),
        seq=99,
        snapshot_dir=str(snap_dir),
        checkpoint_first=True,
    )
    assert info is not None
    assert info.seq == 99


# ---------------------------------------------------------------------------
# list / latest / prune
# ---------------------------------------------------------------------------

def test_list_snapshots_ignores_unrelated_files_and_sorts_by_seq(tmp_path):
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()
    # Real snapshots, out-of-order.
    for seq in (100, 5, 50):
        (snap_dir / f"snapshot-seq{seq}-anytime.duckdb").write_bytes(b"x")
    # Noise that must be ignored.
    (snap_dir / "README.txt").write_text("ignore me")
    (snap_dir / "snapshot-seqXX-malformed.duckdb").write_bytes(b"x")
    (snap_dir / "random.duckdb").write_bytes(b"x")

    snaps = list_snapshots(str(snap_dir))
    assert [s.seq for s in snaps] == [5, 50, 100]


def test_latest_snapshot_returns_highest_seq(tmp_path):
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()
    for seq in (10, 20, 5):
        (snap_dir / f"snapshot-seq{seq}-x.duckdb").write_bytes(b"x")
    assert latest_snapshot(str(snap_dir)).seq == 20


def test_latest_snapshot_returns_none_when_empty(tmp_path):
    assert latest_snapshot(str(tmp_path / "missing")) is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert latest_snapshot(str(empty)) is None


def test_prune_keeps_retain_count(tmp_path, monkeypatch):
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()
    for seq in range(10):
        (snap_dir / f"snapshot-seq{seq}-x.duckdb").write_bytes(b"x")

    monkeypatch.setenv("ENGRAM_SNAPSHOT_RETAIN", "3")
    _prune_snapshots(str(snap_dir))

    remaining_seqs = sorted(s.seq for s in list_snapshots(str(snap_dir)))
    # The 3 highest seqs survive (newest snapshots = most useful base).
    assert remaining_seqs == [7, 8, 9]


def test_prune_default_retain_is_safe(tmp_path):
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()
    # Fewer than DEFAULT_RETAIN snapshots → noop.
    for seq in range(DEFAULT_RETAIN - 1):
        (snap_dir / f"snapshot-seq{seq}-x.duckdb").write_bytes(b"x")
    _prune_snapshots(str(snap_dir))
    assert len(list_snapshots(str(snap_dir))) == DEFAULT_RETAIN - 1


# ---------------------------------------------------------------------------
# SnapshotScheduler
# ---------------------------------------------------------------------------

def test_scheduler_triggers_on_event_count(tmp_path, monkeypatch):
    db_path = tmp_path / "live.duckdb"
    _seed_duckdb(str(db_path))
    snap_dir = tmp_path / "snapshots"

    # Fake event log: pretend many events have happened.
    class FakeLog:
        def current_seq(self):
            return 5000
    monkeypatch.setattr("engram.event_log.get_event_log", lambda: FakeLog())

    sched = SnapshotScheduler(
        db_path=str(db_path),
        snapshot_dir=str(snap_dir),
        interval_events=1000,
        interval_hours=999,  # disable time path
    )
    info = sched.trigger_now()
    assert info is not None
    assert info.seq == 5000


def test_scheduler_triggers_on_time_window(tmp_path, monkeypatch):
    db_path = tmp_path / "live.duckdb"
    _seed_duckdb(str(db_path))
    snap_dir = tmp_path / "snapshots"

    class FakeLog:
        def current_seq(self):
            return 7  # fewer than interval_events; only time path applies
    monkeypatch.setattr("engram.event_log.get_event_log", lambda: FakeLog())

    sched = SnapshotScheduler(
        db_path=str(db_path),
        snapshot_dir=str(snap_dir),
        interval_events=999_999,
        interval_hours=0.001,  # ~3.6s
    )
    # Pretend last snapshot was a long time ago.
    sched._last_at = time.time() - 60
    info = sched.trigger_now()
    assert info is not None
    assert info.seq == 7


def test_scheduler_skips_when_no_new_events(tmp_path, monkeypatch):
    db_path = tmp_path / "live.duckdb"
    _seed_duckdb(str(db_path))
    snap_dir = tmp_path / "snapshots"

    class FakeLog:
        def current_seq(self):
            return 100
    monkeypatch.setattr("engram.event_log.get_event_log", lambda: FakeLog())

    sched = SnapshotScheduler(
        db_path=str(db_path),
        snapshot_dir=str(snap_dir),
        interval_events=10,
        interval_hours=0.001,
    )
    # Already snapshotted at seq=100; now it's "later" but no new events.
    sched._last_seq = 100
    sched._last_at = time.time() - 60
    info = sched.trigger_now()
    assert info is None  # noop when current_seq == last_seq


# ---------------------------------------------------------------------------
# recover._try_seed_from_snapshot integration
# ---------------------------------------------------------------------------

def test_seed_from_snapshot_copies_when_full_replay(tmp_path, monkeypatch):
    from engram.recover import _try_seed_from_snapshot
    from engram import snapshot as snap_mod

    src = tmp_path / "live.duckdb"
    _seed_duckdb(str(src))
    snap_dir = tmp_path / "snapshots"

    # Create one real snapshot and point latest_snapshot at it.
    info = take_snapshot(str(src), seq=200, snapshot_dir=str(snap_dir),
                        checkpoint_first=False)
    assert info is not None
    monkeypatch.setattr(snap_mod, "latest_snapshot", lambda *a, **kw: info)

    output_db = tmp_path / "output.duckdb"
    seq = _try_seed_from_snapshot(str(output_db), since_date=None)
    assert seq == 200
    assert output_db.exists()
    # Output is a working DB.
    conn = duckdb.connect(str(output_db), read_only=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 3
    finally:
        conn.close()


def test_seed_from_snapshot_skipped_when_since_date_set(tmp_path, monkeypatch):
    from engram.recover import _try_seed_from_snapshot
    from engram import snapshot as snap_mod

    src = tmp_path / "live.duckdb"
    _seed_duckdb(str(src))
    info = take_snapshot(
        str(src), seq=99, snapshot_dir=str(tmp_path / "snapshots"),
        checkpoint_first=False,
    )
    monkeypatch.setattr(snap_mod, "latest_snapshot", lambda *a, **kw: info)

    output_db = tmp_path / "output.duckdb"
    seq = _try_seed_from_snapshot(str(output_db), since_date="20260101")
    # Partial replay window must NOT use a snapshot — would skip events.
    assert seq == 0
    assert not output_db.exists()


def test_seed_from_snapshot_no_op_without_snapshot(tmp_path, monkeypatch):
    from engram.recover import _try_seed_from_snapshot
    from engram import snapshot as snap_mod

    monkeypatch.setattr(snap_mod, "latest_snapshot", lambda *a, **kw: None)
    output_db = tmp_path / "output.duckdb"
    seq = _try_seed_from_snapshot(str(output_db), since_date=None)
    assert seq == 0
    assert not output_db.exists()


# ---------------------------------------------------------------------------
# Recover end-to-end: with snapshot vs. without should produce same state
# ---------------------------------------------------------------------------

def test_recover_with_snapshot_matches_full_replay(tmp_path, monkeypatch):
    """Critical correctness test: snapshot fast-path must NOT change output."""
    from engram.event_log import reset_event_log_for_tests
    from engram.db import MemoryDB
    from engram.recover import recover
    from engram import snapshot as snap_mod

    event_dir = tmp_path / "events"
    log = reset_event_log_for_tests(event_dir=str(event_dir))

    try:
        # 1. Original runtime: create some Tier 1 state.
        live_db = tmp_path / "live.duckdb"
        db = MemoryDB(str(live_db), dim=768, log_writes=True)
        try:
            tid = db.create_task("snapshot-recover-test", goal="ok", user_id="u")
            db.update_task(tid, status="in_progress")
            db.upsert_session("sess-1", user_id="u")
            db.checkpoint()
        finally:
            db.close()

        # 2. Take a snapshot at the current event seq.
        snap_dir = tmp_path / "snapshots"
        snap = take_snapshot(
            str(live_db), seq=log.current_seq(),
            snapshot_dir=str(snap_dir), checkpoint_first=False,
        )
        assert snap is not None

        # 3. Make MORE state after the snapshot.
        db = MemoryDB(str(live_db), dim=768, log_writes=True)
        try:
            db.update_task(tid, status="done")
            db.log_session_outcome("sess-1", "success", user_id="u")
        finally:
            db.close()

        monkeypatch.setattr(snap_mod, "latest_snapshot", lambda *a, **kw: snap)

        # 4a. Recover WITH snapshot fast-path.
        out_with = tmp_path / "with-snapshot"
        report_with = recover(
            event_dir=str(event_dir),
            output_dir=str(out_with),
            promote=False,
        )
        assert report_with.snapshot_used is True
        assert report_with.snapshot_seq == snap.seq

        # 4b. Recover WITHOUT snapshot (force fallback).
        monkeypatch.setattr(snap_mod, "latest_snapshot", lambda *a, **kw: None)
        out_full = tmp_path / "full-replay"
        report_full = recover(
            event_dir=str(event_dir),
            output_dir=str(out_full),
            promote=False,
        )
        assert report_full.snapshot_used is False

        # 5. Both DBs must agree on the final logical state.
        def _state(db_path: str) -> dict:
            d = MemoryDB(db_path, dim=768, log_writes=False)
            try:
                row = d._fetchone_dict(
                    "SELECT name, status FROM tasks WHERE id = ?", [tid]
                )
                outcomes = d._fetchall_dicts(
                    "SELECT outcome FROM session_outcome_log WHERE session_id = ?",
                    ["sess-1"],
                )
                return {
                    "task": row,
                    "outcomes": [o["outcome"] for o in outcomes],
                }
            finally:
                d.close()

        assert _state(report_with.output_db) == _state(report_full.output_db)
    finally:
        reset_event_log_for_tests()
