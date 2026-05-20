"""Append-only event log — Engram's only durability primitive.

Two laws this module enforces:
1. Event log is the only durability primitive.
2. If it cannot be replayed, it is not critical state.

Tier 1 (Source of Truth) writes MUST go through `EventLog.append` BEFORE
the corresponding DuckDB write. DuckDB is a projection layer; if it is
lost, Tier 1 state is reconstructed by replaying this log.

Storage layout:
    ~/.engram/events/events-YYYYMMDD.jsonl    (one file per UTC day, append-only)
    ~/.engram/events/.seq                      (monotonic sequence number cache)

Each line is a JSON object:
    {
        "ts": "2026-05-09T02:43:00.123456Z",
        "seq": 12345,
        "kind": "task.create",
        "payload": { ... },
        "engram_version": "0.10.0",
        "schema_version": 1
    }

Durability strategy: every append calls flush() + os.fsync() before
returning. This trades throughput for "no lost events on power loss".
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Iterator

log = logging.getLogger("engram.event_log")

EVENT_SCHEMA_VERSION = 1
DEFAULT_EVENT_DIR = os.path.join(os.path.expanduser("~"), ".engram", "events")
SEQ_FILE_NAME = ".seq"

# Tier 1 event kinds — these MUST be replayable to reconstruct runtime state.
TIER1_KINDS = frozenset({
    "task.create",
    "task.update",
    "task.retry",
    "task.spawn",
    "checkpoint.write",
    "execution.start",
    "execution.end",
    "session.start",
    "session.end",
    "session.outcome",
    "session.memory_recall",
    "bootstrap.from_legacy_db",
})

# Tier 2 event kinds — logged for replay completeness, but degradation-safe.
TIER2_KINDS = frozenset({
    "memory.store",
    "memory.update",
    "memory.delete",
})

# Runtime/maintenance event kinds — operational anchors, not state.
# These are NOT replayed into the DB; they exist so an operator can answer
# "what did the runtime do, and when?" by reading the event log alone.
RUNTIME_KINDS = frozenset({
    "snapshot.create",          # payload: {snapshot_path, seq, db_size_bytes}
    "runtime.duckdb_upgrade",   # payload: {old_version, new_version, backup_path}
    "maintenance.backup_pruned",  # payload: {archived: [...], kept: int, dir}
    "checkpoint.restore",       # payload: {task_id, version}
    "drift.detected",           # payload: {task_id, composite, violations}
    "drift.nudge",              # payload: {task_id, composite, constraint_drift, ...}
    "continuity.evaluated",     # payload: {task_id, composite}
    "continuity.redundant_exploration",  # payload: {task_id, action}
})

VALID_KINDS = TIER1_KINDS | TIER2_KINDS | RUNTIME_KINDS


class EventLogError(RuntimeError):
    """Raised when the event log cannot fulfill its durability contract."""


class EventLog:
    """Append-only event log with daily file rotation and fsync durability.

    Thread-safe: a single internal lock serializes append() so seq numbers
    are monotonic across threads in one process.
    """

    def __init__(self, event_dir: str = DEFAULT_EVENT_DIR, engram_version: str = "0.0.0"):
        self._dir = event_dir
        self._engram_version = engram_version
        self._lock = threading.Lock()
        self._seq = 0
        os.makedirs(self._dir, exist_ok=True)
        self._seq = self._load_seq()

    # ---- public API ----

    def append(self, kind: str, payload: dict[str, Any]) -> int:
        """Append one event. Returns the assigned sequence number.

        Raises:
            EventLogError: if the kind is unknown or write fails.
        """
        if kind not in VALID_KINDS:
            raise EventLogError(f"Unknown event kind: {kind!r}")

        with self._lock:
            self._seq += 1
            seq = self._seq
            event = {
                "ts": _utc_now_iso(),
                "seq": seq,
                "kind": kind,
                "payload": payload,
                "engram_version": self._engram_version,
                "schema_version": EVENT_SCHEMA_VERSION,
            }
            line = json.dumps(event, ensure_ascii=False, default=_json_default)
            try:
                self._write_line(line)
                self._persist_seq(seq)
            except OSError as exc:
                # Roll back the in-memory seq so we never report success on a
                # failed write; the next attempt will retry the same number.
                self._seq -= 1
                raise EventLogError(f"Failed to append event {kind!r}: {exc}") from exc
            return seq

    def current_seq(self) -> int:
        with self._lock:
            return self._seq

    def iter_events(self, since_date: str | None = None) -> Iterator[dict]:
        """Yield events in chronological order (across all daily files).

        Args:
            since_date: lower bound 'YYYYMMDD' (inclusive). None = from beginning.
        """
        for path in self._sorted_event_files(since_date):
            yield from self._iter_file(path)

    def rotate_old_files(self) -> list[str]:
        """Gzip-compress event files older than today.

        Only compresses ``.jsonl`` files whose date stamp is strictly before
        today (UTC).  The active file (today) is never touched.

        Safety: after writing the ``.jsonl.gz`` file we verify the line
        count matches the original before removing the uncompressed copy.

        Returns:
            List of paths that were successfully compressed.
        """
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        compressed: list[str] = []

        try:
            names = [
                n for n in os.listdir(self._dir)
                if n.startswith("events-") and n.endswith(".jsonl")
            ]
        except FileNotFoundError:
            return compressed

        for name in sorted(names):
            date_part = name[len("events-"):-len(".jsonl")]
            if date_part >= today:
                continue  # never compress today's active file

            src_path = os.path.join(self._dir, name)
            gz_path = src_path + ".gz"

            if os.path.exists(gz_path):
                # Already compressed — just remove the leftover .jsonl
                # (could happen if a previous rotate was interrupted after
                # writing .gz but before removing the source).
                try:
                    os.remove(src_path)
                    log.info("Removed leftover %s (gz already exists)", name)
                except OSError as exc:
                    log.warning("Failed to remove leftover %s: %s", name, exc)
                continue

            try:
                source_lines = _count_lines(src_path)
                _gzip_file(src_path, gz_path)
                gz_lines = _count_gz_lines(gz_path)

                if gz_lines != source_lines:
                    log.error(
                        "Line count mismatch for %s: src=%d gz=%d — keeping original",
                        name, source_lines, gz_lines,
                    )
                    try:
                        os.remove(gz_path)
                    except OSError:
                        pass
                    continue

                os.remove(src_path)
                compressed.append(gz_path)
                log.info("Rotated %s → %s.gz (%d lines)", name, name, source_lines)
            except Exception as exc:
                log.warning("Failed to rotate %s: %s", name, exc)
                # Clean up partial .gz to avoid confusion
                try:
                    if os.path.exists(gz_path):
                        os.remove(gz_path)
                except OSError:
                    pass

        return compressed

    # ---- internals ----

    def _current_file_path(self) -> str:
        date_part = datetime.now(timezone.utc).strftime("%Y%m%d")
        return os.path.join(self._dir, f"events-{date_part}.jsonl")

    def _write_line(self, line: str) -> None:
        path = self._current_file_path()
        # O_APPEND guarantees atomic positioning under POSIX; fsync guarantees
        # durability across power loss.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, (line + "\n").encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)

    def _seq_path(self) -> str:
        return os.path.join(self._dir, SEQ_FILE_NAME)

    def _load_seq(self) -> int:
        """Recover the last seq number on startup.

        Strategy: read the .seq cache; if missing or stale, scan the most
        recent event file and take its max seq. Replay always trusts the
        log over the cache.
        """
        cached = self._read_seq_cache()
        scanned = self._scan_max_seq()
        if scanned > cached:
            return scanned
        return cached

    def _read_seq_cache(self) -> int:
        try:
            with open(self._seq_path(), "r", encoding="utf-8") as f:
                return int(f.read().strip() or 0)
        except (FileNotFoundError, ValueError):
            return 0

    def _persist_seq(self, seq: int) -> None:
        # Best-effort cache; the source of truth is the jsonl files themselves.
        try:
            tmp = self._seq_path() + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(str(seq))
            os.replace(tmp, self._seq_path())
        except OSError as exc:
            log.warning("Failed to persist seq cache (non-fatal): %s", exc)

    def _scan_max_seq(self) -> int:
        files = self._sorted_event_files(None)
        if not files:
            return 0
        # Only scan the latest file; seq is monotonic across files because
        # we never go back in time.
        latest = files[-1]
        max_seq = 0
        try:
            for event in self._iter_file(latest):
                if event.get("seq", 0) > max_seq:
                    max_seq = event["seq"]
        except OSError as exc:
            log.warning("Failed to scan %s for seq recovery: %s", latest, exc)
        return max_seq

    def _sorted_event_files(self, since_date: str | None) -> list[str]:
        """Return event file paths in chronological order (.jsonl and .jsonl.gz).

        When both ``events-YYYYMMDD.jsonl`` and ``events-YYYYMMDD.jsonl.gz``
        exist for the same date, the uncompressed file takes precedence
        (it is the actively-written or not-yet-cleaned-up copy).
        """
        try:
            all_names = os.listdir(self._dir)
        except FileNotFoundError:
            return []

        date_to_name: dict[str, str] = {}
        for name in all_names:
            if not name.startswith("events-"):
                continue
            if name.endswith(".jsonl.gz"):
                date_part = name[len("events-"):-len(".jsonl.gz")]
                # .jsonl takes precedence over .jsonl.gz for the same date
                if date_part not in date_to_name:
                    date_to_name[date_part] = name
            elif name.endswith(".jsonl"):
                date_part = name[len("events-"):-len(".jsonl")]
                date_to_name[date_part] = name  # always overwrite .gz entry

        if since_date:
            date_to_name = {d: n for d, n in date_to_name.items() if d >= since_date}

        sorted_dates = sorted(date_to_name)
        return [os.path.join(self._dir, date_to_name[d]) for d in sorted_dates]

    def _iter_file(self, path: str) -> Iterator[dict]:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError as exc:
                    log.error("Skipping malformed event %s:%d: %s", path, lineno, exc)


def _count_lines(path: str) -> int:
    """Count non-empty lines in a plain text file."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _gzip_file(src: str, dst: str) -> None:
    """Compress *src* into *dst* using gzip level 6 (good balance)."""
    with open(src, "rb") as f_in, gzip.open(dst, "wb", compresslevel=6) as f_out:
        while True:
            chunk = f_in.read(1 << 16)  # 64 KiB
            if not chunk:
                break
            f_out.write(chunk)


def _count_gz_lines(path: str) -> int:
    """Count non-empty lines in a gzip-compressed text file."""
    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ---- module-level singleton (lazy) ----

_singleton: EventLog | None = None
_singleton_lock = threading.Lock()


def get_event_log() -> EventLog:
    """Process-wide singleton. Lazy-initialized to allow test isolation via reset()."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                from . import __version__
                _singleton = EventLog(engram_version=__version__)
    return _singleton


def reset_event_log_for_tests(event_dir: str | None = None) -> EventLog:
    """Replace the singleton — tests only."""
    global _singleton
    with _singleton_lock:
        from . import __version__
        _singleton = EventLog(
            event_dir=event_dir or DEFAULT_EVENT_DIR,
            engram_version=__version__,
        )
    return _singleton
