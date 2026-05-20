"""Periodic DB snapshots — replay accelerator for ``engram recover``.

Without snapshots, ``recover()`` scans every event ever written. After a few
months in production that's ~minutes per recover. With snapshots, recover
loads the most recent snapshot DB (already a fully-formed projection) and
only replays events with ``seq > snapshot_seq``.

Design notes:

- A snapshot is a **byte-for-byte copy** of the live DuckDB file (after a
  CHECKPOINT to flush WAL). Cheaper than EXPORT DATABASE and yields an
  identical schema.
- Snapshots are taken on a daemon thread driven by a tiny scheduler in
  this module — NOT inside ``EventLog.append()`` — so log writes never
  pay snapshot latency.
- The trigger is an OR of two cheap conditions polled once per minute:
    * ``current_seq - last_snapshot_seq >= ENGRAM_SNAPSHOT_INTERVAL_EVENTS``
    * ``now - last_snapshot_at >= ENGRAM_SNAPSHOT_INTERVAL_HOURS``
- Snapshot files live at ``~/.engram/snapshots/snapshot-seq{N}-{ts}.duckdb``
  with a small retention budget (oldest pruned to keep the dir bounded).

Recover integration: see ``recover.py::_load_snapshot_base``.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time as _time
from dataclasses import dataclass

log = logging.getLogger(__name__)

ENGRAM_DIR = os.path.join(os.path.expanduser("~"), ".engram")
SNAPSHOT_DIR = os.path.join(ENGRAM_DIR, "snapshots")

ENV_INTERVAL_EVENTS = "ENGRAM_SNAPSHOT_INTERVAL_EVENTS"
ENV_INTERVAL_HOURS = "ENGRAM_SNAPSHOT_INTERVAL_HOURS"
ENV_RETAIN = "ENGRAM_SNAPSHOT_RETAIN"

DEFAULT_INTERVAL_EVENTS = 1000
DEFAULT_INTERVAL_HOURS = 1.0
DEFAULT_RETAIN = 5

# Poll period for the scheduler thread. Cheap (just an int compare).
_POLL_SECONDS = 60.0

_SNAPSHOT_PREFIX = "snapshot-seq"
_SNAPSHOT_SUFFIX = ".duckdb"


@dataclass(frozen=True)
class SnapshotInfo:
    path: str
    seq: int
    ts: str
    size_bytes: int


# ---------------------------------------------------------------------------
# config helpers
# ---------------------------------------------------------------------------

def _read_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        n = int(raw)
        return max(1, n)
    except ValueError:
        log.warning("Invalid %s=%r; using default %d", name, raw, default)
        return default


def _read_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
        return max(0.01, v)
    except ValueError:
        log.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default


def _ts_suffix() -> str:
    return _time.strftime("%Y%m%d-%H%M%S")


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def _parse_seq(name: str) -> int | None:
    """Parse '1234' out of 'snapshot-seq1234-20260509-111200.duckdb'."""
    if not name.startswith(_SNAPSHOT_PREFIX) or not name.endswith(_SNAPSHOT_SUFFIX):
        return None
    middle = name[len(_SNAPSHOT_PREFIX):-len(_SNAPSHOT_SUFFIX)]
    seq_str = middle.split("-", 1)[0]
    try:
        return int(seq_str)
    except ValueError:
        return None


def list_snapshots(snapshot_dir: str = SNAPSHOT_DIR) -> list[SnapshotInfo]:
    """Return all snapshots sorted by seq ascending. Bad files are skipped."""
    if not os.path.isdir(snapshot_dir):
        return []
    out: list[SnapshotInfo] = []
    for name in os.listdir(snapshot_dir):
        seq = _parse_seq(name)
        if seq is None:
            continue
        path = os.path.join(snapshot_dir, name)
        try:
            size = os.path.getsize(path)
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        out.append(SnapshotInfo(
            path=path,
            seq=seq,
            ts=_time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime(mtime)),
            size_bytes=size,
        ))
    out.sort(key=lambda s: s.seq)
    return out


def latest_snapshot(snapshot_dir: str = SNAPSHOT_DIR) -> SnapshotInfo | None:
    snaps = list_snapshots(snapshot_dir)
    return snaps[-1] if snaps else None


# ---------------------------------------------------------------------------
# core: take a snapshot
# ---------------------------------------------------------------------------

def take_snapshot(
    db_path: str,
    seq: int,
    snapshot_dir: str = SNAPSHOT_DIR,
    *,
    checkpoint_first: bool = True,
    shared_conn=None,
) -> SnapshotInfo | None:
    """Copy the live DB file to a new snapshot.

    Returns the new SnapshotInfo, or None if the source DB doesn't exist
    or copy failed. Never raises into the caller.

    ``checkpoint_first``: when True (default), flush WAL into the main
    file BEFORE copying so the snapshot includes all committed writes.

    ``shared_conn``: an existing DuckDB connection to reuse for the
    CHECKPOINT command. When provided, avoids opening a second write
    connection which would cause a segfault (DuckDB does not support
    multiple concurrent write connections within the same process).
    If None and checkpoint_first is True, we skip the explicit checkpoint
    rather than risk a crash — the server already checkpoints periodically.
    """
    if not os.path.exists(db_path):
        log.debug("snapshot skipped: db_path missing %s", db_path)
        return None

    if checkpoint_first:
        try:
            if shared_conn is not None:
                shared_conn.execute("CHECKPOINT")
            else:
                # Do NOT open a second connection — DuckDB C++ layer crashes
                # with SIGSEGV when two write connections exist in-process.
                # The live server already does periodic WAL flushes; worst
                # case the snapshot misses the last few unflushed writes,
                # which recover replays from the event log.
                log.debug("pre-snapshot CHECKPOINT skipped: no shared_conn provided")
        except Exception as exc:
            log.debug("pre-snapshot CHECKPOINT skipped: %s", exc)

    try:
        os.makedirs(snapshot_dir, exist_ok=True)
    except OSError as exc:
        log.warning("Cannot create snapshot dir %s: %s", snapshot_dir, exc)
        return None

    name = f"{_SNAPSHOT_PREFIX}{seq}-{_ts_suffix()}{_SNAPSHOT_SUFFIX}"
    target = os.path.join(snapshot_dir, name)
    try:
        shutil.copy2(db_path, target)
        size = os.path.getsize(target)
    except OSError as exc:
        log.warning("snapshot copy failed: %s", exc)
        return None

    info = SnapshotInfo(
        path=target,
        seq=seq,
        ts=_time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        size_bytes=size,
    )
    log.info("snapshot taken: seq=%d size=%d path=%s", seq, size, target)

    # Best-effort: emit event for operator visibility.
    _try_emit_event("snapshot.create", {
        "snapshot_path": target,
        "seq": seq,
        "db_size_bytes": size,
    })

    _prune_snapshots(snapshot_dir)
    return info


def _prune_snapshots(snapshot_dir: str) -> None:
    retain = _read_int_env(ENV_RETAIN, DEFAULT_RETAIN)
    snaps = list_snapshots(snapshot_dir)
    if len(snaps) <= retain:
        return
    surplus = snaps[: len(snaps) - retain]
    for s in surplus:
        try:
            os.remove(s.path)
            log.info("pruned old snapshot: %s", s.path)
        except OSError as exc:
            log.debug("prune snapshot %s failed: %s", s.path, exc)


# ---------------------------------------------------------------------------
# scheduler: poll every minute, snapshot if either threshold hit
# ---------------------------------------------------------------------------

class SnapshotScheduler:
    """Daemon thread polling event-log seq + wall clock to trigger snapshots.

    One scheduler per process; created lazily by ``ensure_started()``.
    Tests can use ``trigger_now()`` to bypass the timer.
    """

    def __init__(
        self,
        db_path: str,
        snapshot_dir: str = SNAPSHOT_DIR,
        interval_events: int | None = None,
        interval_hours: float | None = None,
        poll_seconds: float = _POLL_SECONDS,
    ):
        self._db_path = db_path
        self._snapshot_dir = snapshot_dir
        self._interval_events = interval_events or _read_int_env(
            ENV_INTERVAL_EVENTS, DEFAULT_INTERVAL_EVENTS
        )
        self._interval_hours = interval_hours or _read_float_env(
            ENV_INTERVAL_HOURS, DEFAULT_INTERVAL_HOURS
        )
        self._poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Initialize from disk so reboots don't re-snapshot immediately.
        last = latest_snapshot(snapshot_dir)
        self._last_seq = last.seq if last else 0
        self._last_at = _time.time() if last is None else os.path.getmtime(last.path)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="engram-snapshot", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self._poll_seconds):
            try:
                self._maybe_snapshot()
            except Exception as exc:
                log.warning("snapshot scheduler tick failed: %s", exc)

    def _maybe_snapshot(self) -> SnapshotInfo | None:
        try:
            from .event_log import get_event_log
            current_seq = get_event_log().current_seq()
        except Exception:
            return None

        events_due = (current_seq - self._last_seq) >= self._interval_events
        hours_elapsed = (_time.time() - self._last_at) / 3600.0
        time_due = hours_elapsed >= self._interval_hours

        if not (events_due or time_due):
            return None
        if current_seq == self._last_seq:
            # Nothing happened since last snapshot; skip even if time elapsed.
            self._last_at = _time.time()
            return None

        # Pass the shared DB connection to avoid opening a second write
        # connection which causes SIGSEGV in DuckDB's C++ layer.
        conn = None
        try:
            from .shared import _db
            if _db is not None:
                conn = _db.conn
        except Exception:
            pass

        info = take_snapshot(
            self._db_path, current_seq, self._snapshot_dir, shared_conn=conn
        )
        if info is not None:
            self._last_seq = info.seq
            self._last_at = _time.time()
        return info

    def trigger_now(self) -> SnapshotInfo | None:
        """Force a snapshot check immediately. Tests + manual ops only."""
        return self._maybe_snapshot()


# ---------------------------------------------------------------------------
# module singleton + start hook
# ---------------------------------------------------------------------------

_scheduler: SnapshotScheduler | None = None
_scheduler_lock = threading.Lock()


def ensure_started(db_path: str, snapshot_dir: str = SNAPSHOT_DIR) -> SnapshotScheduler:
    """Idempotent: start the scheduler if not already running."""
    global _scheduler
    with _scheduler_lock:
        if _scheduler is None:
            _scheduler = SnapshotScheduler(db_path=db_path, snapshot_dir=snapshot_dir)
            _scheduler.start()
            log.info(
                "snapshot scheduler started: every %d events OR %.2fh, dir=%s",
                _scheduler._interval_events,
                _scheduler._interval_hours,
                snapshot_dir,
            )
        return _scheduler


def reset_scheduler_for_tests() -> None:
    global _scheduler
    with _scheduler_lock:
        if _scheduler is not None:
            _scheduler.stop()
        _scheduler = None


# ---------------------------------------------------------------------------
# best-effort event emit
# ---------------------------------------------------------------------------

def _try_emit_event(kind: str, payload: dict) -> None:
    try:
        from .event_log import get_event_log
        get_event_log().append(kind, payload)
    except Exception as exc:
        log.debug("event emit %s skipped: %s", kind, exc)


# ---------------------------------------------------------------------------
# Periodic interrupt checkpoint scheduler
# ---------------------------------------------------------------------------

_checkpoint_scheduler = None
_checkpoint_scheduler_lock = threading.Lock()


def _run_interrupt_checkpoint() -> None:
    """Call trigger_interrupt_checkpoint() safely for scheduler."""
    try:
        from .shared import trigger_interrupt_checkpoint
        result = trigger_interrupt_checkpoint()
        if result:
            log.info("Periodic interrupt checkpoint completed")
    except Exception as exc:
        log.warning("Periodic interrupt checkpoint failed: %s", exc)


def start_checkpoint_scheduler() -> None:
    """Start a 2-minute interrupt checkpoint cycle using APScheduler.

    This creates a periodic checkpoint independent of exit signals,
    ensuring continuity state is persisted even during long-running sessions.
    """
    global _checkpoint_scheduler
    with _checkpoint_scheduler_lock:
        if _checkpoint_scheduler is not None:
            return
        try:
            from apscheduler.schedulers.background import BackgroundScheduler

            scheduler = BackgroundScheduler()
            scheduler.add_job(
                _run_interrupt_checkpoint,
                "interval",
                minutes=2,
                id="interrupt_checkpoint",
            )
            scheduler.start()
            _checkpoint_scheduler = scheduler
            log.info("Interrupt checkpoint scheduler started: every 2 minutes")
        except ImportError:
            log.warning("APScheduler not installed, periodic checkpoint disabled")


def stop_checkpoint_scheduler() -> None:
    """Stop the periodic checkpoint scheduler."""
    global _checkpoint_scheduler
    with _checkpoint_scheduler_lock:
        if _checkpoint_scheduler is not None:
            _checkpoint_scheduler.shutdown()
            _checkpoint_scheduler = None
            log.info("Interrupt checkpoint scheduler stopped")
