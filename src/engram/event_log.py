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
    "checkpoint.write",
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

VALID_KINDS = TIER1_KINDS | TIER2_KINDS


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

    # ---- internals ----

    def _current_file_path(self) -> str:
        date_part = datetime.now(timezone.utc).strftime("%Y%m%d")
        return os.path.join(self._dir, f"events-{date_part}.jsonl")

    def _write_line(self, line: str) -> None:
        path = self._current_file_path()
        # O_APPEND guarantees atomic positioning under POSIX; fsync guarantees
        # durability across power loss.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
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
        try:
            names = [
                n for n in os.listdir(self._dir)
                if n.startswith("events-") and n.endswith(".jsonl")
            ]
        except FileNotFoundError:
            return []
        if since_date:
            names = [n for n in names if n[len("events-"):-len(".jsonl")] >= since_date]
        names.sort()
        return [os.path.join(self._dir, n) for n in names]

    def _iter_file(self, path: str) -> Iterator[dict]:
        with open(path, "r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError as exc:
                    log.error("Skipping malformed event %s:%d: %s", path, lineno, exc)


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
