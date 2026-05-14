"""DuckDB storage layer — schema, CRUD, vector search, FTS.

Engram architecture (v0.10+):
    Tier 1 (Source of Truth): tasks / checkpoints / session_lifecycle / ...
        backed by ``~/.engram/events/*.jsonl`` (append-only event log).
    Tier 2 (Degradable):      memories content / metadata / graph
        materialized in this DuckDB file.
    Tier 3 (Disposable):      embeddings / FTS / vector index
        rebuildable from Tier 2.

Two laws this module enforces:
    1. Event log is the only durability primitive.
    2. If it cannot be replayed, it is not critical state.

If DuckDB cannot be opened or schema is unsafe to use, MemoryDB enters
**readonly degraded mode** instead of silently creating a fresh database.
The previous "rename to .corrupt and continue" behaviour caused users to
lose runtime state without warning and is now removed.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time as _time
from datetime import datetime, timezone
from dataclasses import dataclass, field

import duckdb

from .embedding import get_dimensions, MODEL_NAME
from .config import DEDUP_SEARCH_THRESHOLD, SIMILARITY_LOW

log = logging.getLogger("engram.db")

ENGRAM_DIR = os.path.join(os.path.expanduser("~"), ".engram")
DB_PATH = os.path.join(ENGRAM_DIR, "memories.duckdb")
_BACKUP_DIR = os.path.join(ENGRAM_DIR, "backups")

# Engram-internal schema version. Bump when Tier-1 / Tier-2 SQL schema
# changes in a way that needs explicit handling.
ENGRAM_SCHEMA_VERSION = 1

# Toggle: only when explicitly set, MemoryDB is allowed to wipe a corrupt
# database file and start fresh. Default = forbidden (we prefer degraded
# mode + explicit ``engram recover`` over silent data loss).
ENV_ALLOW_RESET = "ENGRAM_ALLOW_RESET"

_DEFAULT_DIM = 768


class DegradedModeError(RuntimeError):
    """Raised when a write is attempted while MemoryDB is in readonly degraded mode.

    The error message and ``recover_command`` attribute should be surfaced
    to the user / MCP client so they can run ``engram recover`` explicitly.
    """

    def __init__(self, reason: str, db_path: str):
        super().__init__(reason)
        self.reason = reason
        self.db_path = db_path
        self.recover_command = "engram recover"


class DatabaseCorruptionError(RuntimeError):
    """Raised when DuckDB cannot be opened and silent recovery is disabled."""

    def __init__(self, db_path: str, original: BaseException, backup_path: str | None = None):
        msg = (
            f"DuckDB file at {db_path} cannot be opened "
            f"({original.__class__.__name__}: {original}). "
            f"Engram will not silently create a fresh database "
            f"(would lose runtime state). "
        )
        if backup_path:
            msg += f"Original file isolated at {backup_path}. "
        msg += (
            "Run `engram recover` to rebuild from the event log, or set "
            f"{ENV_ALLOW_RESET}=1 to allow a fresh start."
        )
        super().__init__(msg)
        self.db_path = db_path
        self.original = original
        self.backup_path = backup_path
        self.recover_command = "engram recover"


class DatabaseLockedError(RuntimeError):
    """Raised when DuckDB refuses to open because another process holds the lock.

    This is NOT corruption — the file is healthy, someone else is just using
    it. Callers MUST NOT rename, move, or alter the DB file in this case.
    The right action is to refuse to start the second process and surface
    a clear "engram is already running" message to the operator.
    """

    def __init__(self, db_path: str, original: BaseException):
        super().__init__(
            f"DuckDB file at {db_path} is locked by another process "
            f"({original}). The DB file is healthy — do NOT delete or "
            "rename it. Stop the other engram process first "
            "(`engram-server stop` or `launchctl unload`), then retry."
        )
        self.db_path = db_path
        self.original = original


def _is_lock_conflict(exc: BaseException) -> bool:
    """Recognize DuckDB's "file lock held by another process" error.

    DuckDB raises this as ``IOException`` with a message containing
    "Conflicting lock". We match on substring because the exception type
    is shared with real "file is corrupt" IO errors.
    """
    msg = str(exc).lower()
    return "conflicting lock" in msg or "could not set lock" in msg

def _dim() -> int:
    """Detect embedding dimension, with mismatch detection.

    If the DB already exists and has memories, check that the cached/stored
    dimension matches the current model dimension.  If not, return the DB's
    existing dimension so _init_schema doesn't CREATE TABLE with the wrong
    FLOAT[N] and cause all INSERTs to fail.
    """
    cache = os.path.join(ENGRAM_DIR, ".dim_cache")

    # 1) Try reading from dim_cache first
    cached_dim = None
    try:
        with open(cache) as f:
            line = f.read().strip()
            if ":" in line:
                model, dim_str = line.rsplit(":", 1)
                if model == MODEL_NAME:
                    cached_dim = int(dim_str)
            else:
                cached_dim = int(line)
    except Exception:
        pass

    # 2) Detect actual dimension: try reading from existing DB
    db_dim = _read_db_dimension(DB_PATH)

    if db_dim is not None and cached_dim is not None and db_dim != cached_dim:
        log.warning(
            "Dimension mismatch! DB has FLOAT[%d] but dim_cache says %d. "
           "Trusting DB dimension to avoid INSERT failures.",
            db_dim, cached_dim,
        )
        # Update cache to match reality
        try:
            os.makedirs(ENGRAM_DIR, exist_ok=True)
            with open(cache, "w") as f:
                f.write(f"{MODEL_NAME}:{db_dim}")
        except Exception:
            pass
        return db_dim

    if db_dim is not None:
        return db_dim

    if cached_dim is not None:
        return cached_dim

    # 3) No existing data — prefer fast startup over eager model probing.
    # Loading sentence-transformers here can block MCP boot and trigger
    # proxy startup timeouts. Use default dim unless explicitly opted into
    # eager detection.
    if os.environ.get("ENGRAM_EAGER_DIM_DETECT", "0") == "1":
        try:
            dim = get_dimensions()
            os.makedirs(ENGRAM_DIR, exist_ok=True)
            with open(cache, "w") as f:
                f.write(f"{MODEL_NAME}:{dim}")
            return dim
        except Exception:
            return _DEFAULT_DIM
    return _DEFAULT_DIM


def _read_db_dimension(db_path: str) -> int | None:
    """Read the actual embedding dimension from an existing DB.

    Returns None if DB doesn't exist, is empty, or can't be read.
    """
    if not os.path.exists(db_path):
        return None
    try:
        conn = duckdb.connect(db_path, read_only=True)
        try:
            row = conn.execute(
                "SELECT array_length(embedding) FROM memories LIMIT 1"
            ).fetchone()
            if row is not None and row[0] is not None:
                return int(row[0])
        finally:
            conn.close()
    except Exception:
        log.debug("Could not read DB dimension: %s", exc_info=True)
    return None


def _force_checkpoint(db_path: str) -> bool:
    """Try to open the DB read-write and run FORCE CHECKPOINT.

    Returns True if checkpoint succeeded (WAL absorbed into main file).
    This preserves data that would otherwise be lost when we move the WAL.

    Raises:
        DatabaseLockedError: if the DB is held by another process. We
            propagate this rather than swallowing it because the caller
            must NOT proceed with WAL isolation in that case (the WAL
            is healthy and being actively written by another engram).
    """
    wal = db_path + ".wal"
    if not os.path.exists(wal) or os.path.getsize(wal) == 0:
        return True  # nothing to checkpoint
    try:
        conn = duckdb.connect(db_path)
        try:
            conn.execute("FORCE CHECKPOINT")
            log.info("Forced checkpoint succeeded — WAL data preserved")
            return True
        finally:
            conn.close()
    except Exception as e:
        if _is_lock_conflict(e):
            raise DatabaseLockedError(db_path, e) from e
        log.warning("Force checkpoint failed: %s", e)
        return False


def _ts_suffix() -> str:
    return _time.strftime("%Y%m%d-%H%M%S")


def _isolate_corrupt_file(path: str, label: str) -> str | None:
    """Move a corrupt artifact aside with a timestamped suffix.

    Never overwrites a previous isolation: each call gets a unique suffix.
    """
    if not os.path.exists(path):
        return None
    target = f"{path}.{label}.{_ts_suffix()}"
    try:
        os.replace(path, target)
        log.warning("Isolated %s artifact: %s -> %s", label, path, target)
        return target
    except OSError as exc:
        log.error("Failed to isolate %s file %s: %s", label, path, exc)
        return None


def _recover_wal(db_path: str) -> str | None:
    """Try hard to keep WAL data, only move it aside as last resort.

    Sequence:
      1. If no WAL or empty WAL → nothing to do.
      2. Try ``_force_checkpoint()`` to absorb WAL into the main file.
         Success means we kept all unflushed writes.
      3. If checkpoint fails (truly corrupt WAL), move the WAL aside with
         a timestamped suffix so an operator can inspect it later.
         Returns the backup path so callers can surface it.
    """
    wal = db_path + ".wal"
    if not os.path.exists(wal) or os.path.getsize(wal) == 0:
        return None

    # Note: _force_checkpoint may raise DatabaseLockedError. We deliberately
    # let that propagate — moving a WAL aside while another process is
    # actively writing to it would silently lose that process's data.
    if _force_checkpoint(db_path):
        # WAL was successfully absorbed; DuckDB removes the file itself.
        return None

    log.warning(
        "WAL file present and could not be checkpointed: %s (%d bytes). "
        "Moving aside; any unflushed writes will be lost.",
        wal, os.path.getsize(wal),
    )
    return _isolate_corrupt_file(wal, "wal-recovery")


def _scan_residue(db_path: str) -> list[str]:
    """Find leftover .corrupt.* / .wal-recovery.* artifacts to warn about."""
    db_dir = os.path.dirname(db_path) or "."
    base = os.path.basename(db_path)
    try:
        names = os.listdir(db_dir)
    except FileNotFoundError:
        return []
    return sorted(
        os.path.join(db_dir, n)
        for n in names
        if n.startswith(base + ".") and (".corrupt." in n or ".wal-recovery." in n)
    )


def _connect_with_retry(db_path: str) -> duckdb.DuckDBPyConnection:
    """Connect to DuckDB.

    Failure policy (Engram v0.11.1+):
        1. Try to open the file as-is.
        2. **If the failure is a lock conflict** (another process holds
           the DB), raise ``DatabaseLockedError`` IMMEDIATELY. The DB
           file is healthy; we MUST NOT touch it in this case.
        3. Otherwise, if a WAL is present, try to checkpoint or move it
           aside (timestamped) and retry — trading unflushed writes for
           the ability to read previously committed data.
        4. If it still fails, raise ``DatabaseCorruptionError``. We DO NOT
           silently rename the file and create a fresh database.

    Escape hatch: setting env ``ENGRAM_ALLOW_RESET=1`` re-enables the
    "create fresh DB" behaviour for users who explicitly want a clean slate.
    Lock conflicts are NOT affected by this env — a locked file is healthy
    and can never benefit from a reset.
    """
    try:
        return duckdb.connect(db_path)
    except (duckdb.IOException, duckdb.InternalException) as first_err:
        # CRITICAL: lock conflict means another engram is already running.
        # The file is fine — fail loudly without touching it.
        if _is_lock_conflict(first_err):
            raise DatabaseLockedError(db_path, first_err) from first_err

        log.warning("DuckDB connect failed: %s — attempting WAL isolation", first_err)
        try:
            _recover_wal(db_path)
        except DatabaseLockedError:
            # Lock contention surfaced during WAL recovery — same story:
            # don't touch the file, surface a clear error.
            raise
        try:
            return duckdb.connect(db_path)
        except Exception as second_err:
            if _is_lock_conflict(second_err):
                raise DatabaseLockedError(db_path, second_err) from second_err
            allow_reset = os.environ.get(ENV_ALLOW_RESET, "").strip() in ("1", "true", "yes")
            backup_path: str | None = None
            if os.path.exists(db_path):
                backup_path = _isolate_corrupt_file(db_path, "corrupt")
            if allow_reset:
                log.error(
                    "DuckDB unrecoverable; %s=1 set, creating fresh database. "
                    "Original isolated at %s",
                    ENV_ALLOW_RESET, backup_path,
                )
                return duckdb.connect(db_path)
            raise DatabaseCorruptionError(
                db_path=db_path,
                original=second_err,
                backup_path=backup_path,
            ) from second_err


def _schema_sql(dim: int) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY DEFAULT nextval('memory_id_seq'),
    user_id VARCHAR DEFAULT 'default',
    content TEXT NOT NULL,
    embedding FLOAT[{dim}],
    importance FLOAT NOT NULL DEFAULT 0.5,
    category VARCHAR NOT NULL DEFAULT 'fact',
    recall_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT now(),
    last_accessed_at TIMESTAMP NOT NULL DEFAULT now(),
    metadata JSON DEFAULT '{{}}'::JSON
);
"""

FTS_SQL = """
INSTALL fts;
LOAD fts;
"""

SESSION_LOG_SQL = """
CREATE SEQUENCE IF NOT EXISTS session_log_id_seq START 1;

CREATE TABLE IF NOT EXISTS session_memory_log (
    id INTEGER PRIMARY KEY DEFAULT nextval('session_log_id_seq'),
    session_id VARCHAR NOT NULL,
    memory_id INTEGER NOT NULL,
    user_id VARCHAR NOT NULL DEFAULT 'default',
    recalled_at TIMESTAMP NOT NULL DEFAULT now()
);
"""

SESSION_OUTCOME_SQL = """
CREATE SEQUENCE IF NOT EXISTS session_outcome_id_seq START 1;

CREATE TABLE IF NOT EXISTS session_outcome_log (
    id INTEGER PRIMARY KEY DEFAULT nextval('session_outcome_id_seq'),
    session_id VARCHAR NOT NULL,
    user_id VARCHAR NOT NULL DEFAULT 'default',
    outcome VARCHAR NOT NULL,
    recorded_at TIMESTAMP NOT NULL DEFAULT now()
);
"""


SESSION_LIFECYCLE_SQL = """
CREATE TABLE IF NOT EXISTS session_lifecycle (
    session_id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL DEFAULT 'default',
    started_at TIMESTAMP NOT NULL DEFAULT now(),
    last_active_at TIMESTAMP NOT NULL DEFAULT now(),
    ended_at TIMESTAMP,
    end_type VARCHAR,
    interruption_reason VARCHAR,
    interruption_context JSON DEFAULT '{}'::JSON
);
"""

# --- Interruption Taxonomy (v0.12) ---
# 6-value enum for classifying WHY a session was interrupted.
# Used in session_lifecycle.interruption_reason and recovery routing.
INTERRUPTION_OVERFLOW = "overflow"          # Context window full / token limit
INTERRUPTION_USER_AWAY = "user_away"        # User closed / long inactivity
INTERRUPTION_TOOL_FAILURE = "tool_failure"  # Consecutive tool call failures
INTERRUPTION_CRASH = "crash"               # Process died without atexit
INTERRUPTION_RATE_LIMIT = "rate_limit"      # API throttling
INTERRUPTION_UNKNOWN = "unknown"            # Cannot determine (fallback)

VALID_INTERRUPTION_REASONS = {
    INTERRUPTION_OVERFLOW, INTERRUPTION_USER_AWAY, INTERRUPTION_TOOL_FAILURE,
    INTERRUPTION_CRASH, INTERRUPTION_RATE_LIMIT, INTERRUPTION_UNKNOWN,
}

# Recovery strategy per interruption reason — consumed by handlers.
RECOVERY_STRATEGIES: dict[str, dict] = {
    INTERRUPTION_OVERFLOW: {
        "action": "restore_checkpoint",
        "memory_restore_mode": "SELECTIVE",
        "hint": "Context window was full. Restore checkpoint and compress context before continuing.",
    },
    INTERRUPTION_USER_AWAY: {
        "action": "show_summary",
        "memory_restore_mode": "NONE",
        "hint": "User was away. Show a brief handoff summary to re-orient.",
    },
    INTERRUPTION_TOOL_FAILURE: {
        "action": "restore_checkpoint",
        "memory_restore_mode": "SELECTIVE",
        "hint": "Tools failed repeatedly. Restore checkpoint, review failure records, and consider alternative approaches.",
        "include_failures": True,
    },
    INTERRUPTION_CRASH: {
        "action": "restore_checkpoint",
        "memory_restore_mode": "FULL",
        "hint": "Process crashed unexpectedly. Full checkpoint restore recommended; verify data integrity.",
    },
    INTERRUPTION_RATE_LIMIT: {
        "action": "restore_checkpoint",
        "memory_restore_mode": "SELECTIVE",
        "hint": "API rate limit hit. Restore checkpoint; consider switching models or waiting before retrying.",
    },
    INTERRUPTION_UNKNOWN: {
        "action": "show_summary",
        "memory_restore_mode": "NONE",
        "hint": "Previous session ended unexpectedly. Consider recalling its context.",
    },
}

# engram_meta — process/version/runtime contract.
# This table is what MCP clients (Claude Code, Cursor, ...) inspect to
# negotiate compatibility. Keys are app-defined; values are stringly typed.
ENGRAM_META_SQL = """
CREATE TABLE IF NOT EXISTS engram_meta (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT now()
);
"""

TASK_SQL = """
CREATE SEQUENCE IF NOT EXISTS task_id_seq START 1;

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY DEFAULT nextval('task_id_seq'),
    name VARCHAR NOT NULL,
    goal TEXT NOT NULL DEFAULT '',
    status VARCHAR NOT NULL DEFAULT 'planning',
    user_id VARCHAR NOT NULL DEFAULT 'default',
    created_at TIMESTAMP NOT NULL DEFAULT now(),
    updated_at TIMESTAMP NOT NULL DEFAULT now(),
    metadata JSON DEFAULT '{}'::JSON
);
"""


# Checkpoint v2 — Cognitive Continuation Layer
#
# 每个 checkpoint 是一个 task 在某个时刻的认知状态快照。
# - kind: 'handoff'（Agent 主动） / 'auto'（事件驱动自动）
# - checkpoint_reason 枚举：MANUAL_HANDOFF / PLAN_UPDATE / FAILURE /
#                          WORKING_SET_SHIFT / TOOL_FINISHED / AUTO_SAVE
# - must_not_redo: negative memory，结构化对象列表
# - state_diff: 浅 diff（semantic 字段对，不追求 JSON Patch replay correctness）
CHECKPOINT_SQL = """
CREATE SEQUENCE IF NOT EXISTS checkpoint_id_seq START 1;

CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY DEFAULT nextval('checkpoint_id_seq'),
    task_id INTEGER NOT NULL,
    version INTEGER NOT NULL,
    parent_version INTEGER,
    kind VARCHAR NOT NULL DEFAULT 'auto',
    checkpoint_reason VARCHAR NOT NULL DEFAULT 'AUTO_SAVE',
    triggered_by_event VARCHAR,

    goal TEXT,
    completed JSON DEFAULT '[]'::JSON,
    in_progress JSON DEFAULT '[]'::JSON,
    blocked JSON DEFAULT '[]'::JSON,
    preferred_next JSON DEFAULT '[]'::JSON,
    must_not_redo JSON DEFAULT '[]'::JSON,
    must_preserve JSON DEFAULT '[]'::JSON,
    working_set JSON DEFAULT '{}'::JSON,

    state_diff JSON DEFAULT '{}'::JSON,
    source_session_id VARCHAR,
    source_memory_id INTEGER,

    continuation_confidence DOUBLE,
    confidence_breakdown JSON DEFAULT '{}'::JSON,
    failure_signature VARCHAR,

    user_id VARCHAR NOT NULL DEFAULT 'default',
    created_at TIMESTAMP NOT NULL DEFAULT now(),

    UNIQUE (task_id, version)
);
"""

@dataclass
class TaskRow:
    id: int
    name: str
    goal: str
    status: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    metadata: dict | None = field(default=None)

@dataclass
class MemoryRow:
    id: int
    user_id: str
    content: str
    importance: float
    category: str
    recall_count: int
    created_at: datetime
    last_accessed_at: datetime
    metadata: dict | None = field(default=None)
    similarity: float = 0.0
    bm25_score: float = 0.0


def _parse_metadata(raw) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _row_to_memory(row: dict) -> MemoryRow:
    """Map a dict row to MemoryRow. Uses column names for resilience."""
    return MemoryRow(
        id=row["id"],
        user_id=row["user_id"],
        content=row["content"],
        importance=row["importance"],
        category=row["category"],
        recall_count=row["recall_count"],
        created_at=row["created_at"],
        last_accessed_at=row["last_accessed_at"],
        metadata=_parse_metadata(row["metadata"]),
    )


class MemoryDB:
    """DuckDB projection layer for Engram.

    Tier 1 writes (tasks / checkpoints / sessions) MUST also be appended to
    the event log via ``self._event_log``. Tier 2/3 writes (memories / FTS /
    embeddings) are best-effort logged for replay completeness.

    When ``_readonly`` is True, every write method short-circuits with
    ``DegradedModeError``. The HTTP/MCP layer is expected to translate that
    into a 503 / structured error containing ``recover_command``.
    """

    def __init__(
        self,
        db_path: str = DB_PATH,
        dim: int | None = None,
        *,
        log_writes: bool = True,
    ):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._db_path = db_path
        self._fts_available = False
        self._fts_dirty = True  # Force rebuild on first search
        self._readonly = False
        self._readonly_reason: str | None = None
        # Tier 3 quality flag: True when embeddings in DB no longer match the
        # current model. Reads still work (we just can't do vector search until
        # re-embed); writes are NOT blocked because content is the source of truth.
        self._embedding_stale = False
        # Lazy event-log handle. Tests can monkeypatch ``_event_log`` to a Mock.
        self._event_log = None
        self._log_writes = log_writes
        # Residue files surfaced for diagnostics (consumed by daemon/CLI).
        self._residue_files: list[str] = _scan_residue(db_path)
        if self._residue_files:
            log.warning(
                "Residue files detected near %s — may indicate prior corruption: %s",
                db_path, self._residue_files,
            )
        self.conn = _connect_with_retry(db_path)
        self._dim = dim or _dim()
        # Populated by _init_meta(): the duckdb_version recorded by the
        # previous boot, BEFORE this boot overwrote it. Maintenance hook
        # uses this to decide whether to make a pre-upgrade backup.
        self.previous_duckdb_version: str | None = None
        self.current_duckdb_version: str | None = None
        self._init_schema()
        # Best-effort: kick off async maintenance (backup pruning + duckdb
        # upgrade detection). Never blocks boot, never raises.
        self._schedule_maintenance()

    def _schedule_maintenance(self) -> None:
        """Hook P1-3 + P1-5 + P1-1 into the boot path on a daemon thread.

        Order:
          - maintenance (backup pruning + duckdb upgrade detect) is one-shot
            and runs on its own thread.
          - snapshot scheduler is a long-lived daemon thread idempotently
            started once per process.
        Both are best-effort: any failure is logged at debug and cannot
        block boot.
        """
        # Skip background work entirely for short-lived utility opens
        # (recover dry-run, tests, doctor) where ``log_writes=False`` already
        # signals "this is not a primary runtime instance".
        if not self._log_writes:
            return
        try:
            from .maintenance import schedule_startup_maintenance
            schedule_startup_maintenance(
                db_path=self._db_path,
                old_duckdb_version=self.previous_duckdb_version,
                new_duckdb_version=self.current_duckdb_version,
            )
        except Exception as exc:
            log.debug("maintenance scheduling skipped: %s", exc)
        try:
            from .snapshot import ensure_started as ensure_snapshot_started
            ensure_snapshot_started(db_path=self._db_path)
        except Exception as exc:
            log.debug("snapshot scheduler skipped: %s", exc)

    # ---- durability primitives ----

    def _get_event_log(self):
        """Lazily attach to the singleton event log.

        Returns None when logging is disabled (tests) or when the log can't
        be opened — Tier-2/3 writes proceed regardless; Tier-1 helpers MUST
        also call ``_assert_writable()`` first.
        """
        if not self._log_writes:
            return None
        if self._event_log is None:
            try:
                from .event_log import get_event_log
                self._event_log = get_event_log()
            except Exception as exc:  # pragma: no cover — defensive
                log.error("Failed to attach event log: %s", exc)
                self._event_log = None
        return self._event_log

    def _emit_event(self, kind: str, payload: dict) -> None:
        """Append a Tier-1/Tier-2 event. Never raises through to callers
        of Tier-2 paths; Tier-1 helpers should let exceptions propagate
        (they're caught at the handler boundary)."""
        log_ = self._get_event_log()
        if log_ is None:
            return
        log_.append(kind, payload)

    def _assert_writable(self) -> None:
        if self._readonly:
            raise DegradedModeError(
                self._readonly_reason or "MemoryDB is in readonly degraded mode",
                self._db_path,
            )

    def enter_degraded_mode(self, reason: str) -> None:
        """Force the DB into readonly degraded mode (e.g. on partial schema)."""
        self._readonly = True
        self._readonly_reason = reason
        log.error("MemoryDB entering readonly degraded mode: %s", reason)

    @property
    def readonly(self) -> bool:
        return self._readonly

    @property
    def embedding_stale(self) -> bool:
        return self._embedding_stale

    @property
    def residue_files(self) -> list[str]:
        return list(self._residue_files)

    def checkpoint(self) -> None:
        """Flush WAL into the main file. Safe to call on shutdown."""
        if self._readonly or self.conn is None:
            return
        try:
            self.conn.execute("CHECKPOINT")
        except Exception as exc:
            log.warning("CHECKPOINT failed (non-fatal): %s", exc)

    def _checkpoint_after_extension_op(self, label: str) -> None:
        """Flush WAL immediately after an extension-heavy operation.

        DuckDB 1.5.x cannot replay WAL entries that reference extension
        internal tables (FTS ``fts_main_*``, VSS HNSW structures) when
        the extension is not yet loaded at WAL-replay time.  By
        checkpointing right after these operations we ensure the WAL
        never contains extension-specific entries — so even a hard kill
        (SIGKILL / OOM) leaves a replayable WAL.
        """
        try:
            self.conn.execute("CHECKPOINT")
            log.debug("Post-%s CHECKPOINT succeeded", label)
        except Exception as exc:
            log.warning("Post-%s CHECKPOINT failed (non-fatal): %s", label, exc)

    @property
    def fts_available(self) -> bool:
        return self._fts_available

    def _fetchone_dict(self, sql: str, params=None) -> dict | None:
        """Execute query and return first row as dict, or None."""
        result = self.conn.execute(sql, params or [])
        cols = [d[0] for d in result.description]
        row = result.fetchone()
        if row is None:
            return None
        return dict(zip(cols, row))

    def _fetchall_dicts(self, sql: str, params=None) -> list[dict]:
        """Execute query and return all rows as list of dicts."""
        result = self.conn.execute(sql, params or [])
        cols = [d[0] for d in result.description]
        return [dict(zip(cols, row)) for row in result.fetchall()]

    def _detect_embedding_drift(self) -> None:
        """Compare on-disk embedding column dim with current model dim.

        v0.10 policy change: we DO NOT auto-migrate (ALTER TABLE drops every
        existing embedding) and we DO NOT raise. Instead we set
        ``self._embedding_stale = True`` so retrieve.py can fall back to FTS,
        and emit a clear warning telling the user to re-embed explicitly.

        Rationale: embedding is Tier 3 (disposable cache). A model swap must
        not silently null out user-visible content nor crash the runtime.
        """
        try:
            row = self.conn.execute(
                "SELECT array_length(embedding) FROM memories LIMIT 1"
            ).fetchone()
            if row is None or row[0] is None:
                return  # empty table — fresh schema will use current dim
            db_dim = int(row[0])
        except Exception:
            return  # memories table doesn't exist yet

        if db_dim == self._dim:
            return

        self._embedding_stale = True
        log.warning(
            "Embedding dimension drift detected: DB column=FLOAT[%d], "
            "current model=%s wants FLOAT[%d]. Vector search will fall back "
            "to BM25/FTS until you re-embed. Run `engram reembed` (or restart "
            "with the previous ENGRAM_MODEL) to restore vector recall.",
            db_dim, MODEL_NAME, self._dim,
        )
        # Pin the dim we'll keep using for new inserts to whatever the table
        # already has, so writes don't break.
        self._dim = db_dim

    def _init_schema(self):
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS memory_id_seq START 1")
        self._detect_embedding_drift()
        self.conn.execute(_schema_sql(self._dim))
        try:
            self.conn.execute(
                "ALTER TABLE memories ADD COLUMN IF NOT EXISTS metadata JSON DEFAULT '{}'::JSON"
            )
        except Exception as e:
            log.debug("ALTER TABLE metadata column (already exists): %s", e)
        try:
            self.conn.execute(FTS_SQL)
        except Exception as e:
            log.warning("FTS extension load failed: %s", e)
        self._rebuild_fts_index()
        self.conn.execute(SESSION_LOG_SQL)
        self.conn.execute(SESSION_OUTCOME_SQL)
        self.conn.execute(SESSION_LIFECYCLE_SQL)
        self.conn.execute(TASK_SQL)
        self.conn.execute(CHECKPOINT_SQL)
        self.conn.execute(ENGRAM_META_SQL)
        # tasks 表加列：缓存最新 checkpoint + 为未来 execution graph 留位
        # (parent_task_id / retry_of_task_id 暂不实现读写, 仅留位避免破坏性迁移)
        for ddl in (
            "ALTER TABLE tasks ADD COLUMN IF NOT EXISTS latest_checkpoint_version INTEGER DEFAULT 0",
            "ALTER TABLE tasks ADD COLUMN IF NOT EXISTS checkpoint_count INTEGER DEFAULT 0",
            "ALTER TABLE tasks ADD COLUMN IF NOT EXISTS parent_task_id INTEGER",
            "ALTER TABLE tasks ADD COLUMN IF NOT EXISTS retry_of_task_id INTEGER",
        ):
            try:
                self.conn.execute(ddl)
            except Exception as e:
                log.debug("ALTER tasks (extra cols) skipped: %s", e)
        # session_lifecycle 表加列：v0.12 Interruption Taxonomy
        for ddl in (
            "ALTER TABLE session_lifecycle ADD COLUMN IF NOT EXISTS interruption_reason VARCHAR",
            "ALTER TABLE session_lifecycle ADD COLUMN IF NOT EXISTS interruption_context JSON DEFAULT '{}'::JSON",
        ):
            try:
                self.conn.execute(ddl)
            except Exception as exc:
                log.debug("ALTER session_lifecycle (taxonomy cols) skipped: %s", exc)
        # checkpoints 表索引（task 内按 version 倒序查找最高频）
        for ddl in (
            "CREATE INDEX IF NOT EXISTS idx_checkpoints_task_version ON checkpoints(task_id, version DESC)",
            "CREATE INDEX IF NOT EXISTS idx_checkpoints_session ON checkpoints(source_session_id)",
        ):
            try:
                self.conn.execute(ddl)
            except Exception as e:
                log.debug("CREATE INDEX on checkpoints skipped: %s", e)
        # DuckDB sequences can drift from actual max(id) after WAL replay.
        # Realign each sequence to the table's MAX(id) to prevent PK conflicts.
        for tbl, seq in [
            ("memories", "memory_id_seq"),
            ("session_memory_log", "session_log_id_seq"),
            ("session_outcome_log", "session_outcome_id_seq"),
            ("tasks", "task_id_seq"),
            ("checkpoints", "checkpoint_id_seq"),
        ]:
            try:
                max_id = self.conn.execute(
                    f"SELECT COALESCE(MAX(id), 0) FROM {tbl}"
                ).fetchone()[0]
                if max_id > 0:
                    self.conn.execute(
                        f"SELECT setval('{seq}', {max_id})"
                    )
            except Exception as e:
                log.debug("Sequence sync for %s skipped: %s", tbl, e)
        self._init_vss()
        self._init_meta()
        self._backfill_null_embeddings()

    def _backfill_null_embeddings(self) -> None:
        """Re-compute embeddings for memories that have NULL embedding.

        After ``engram recover`` the embedding column is NULL because
        embeddings are Tier 3 (disposable cache) and not stored in the
        event log. We backfill them here so vector search works immediately
        after a recovery, rather than waiting for each memory to be
        individually accessed.
        """
        if self._readonly or self._embedding_stale:
            return
        try:
            rows = self.conn.execute(
                "SELECT id, content FROM memories WHERE embedding IS NULL"
            ).fetchall()
        except Exception:
            return
        if not rows:
            return
        log.info(
            "Found %d memories with NULL embedding — backfilling", len(rows),
        )
        from .embedding import embed
        cast = self._float_cast()
        filled = 0
        for memory_id, content in rows:
            try:
                vec = embed(content)
                self.conn.execute(
                    f"UPDATE memories SET embedding = ?::{cast} WHERE id = ?",
                    [vec, memory_id],
                )
                filled += 1
            except Exception as exc:
                log.warning(
                    "Embedding backfill failed for memory %d: %s",
                    memory_id, exc,
                )
        if filled:
            log.info("Embedding backfill complete: %d/%d filled", filled, len(rows))
            self._checkpoint_after_extension_op("embedding backfill")

    def _init_meta(self) -> None:
        """Seed engram_meta with current process identity.

        Records (idempotent upsert):
          - schema_version        : ENGRAM_SCHEMA_VERSION
          - engram_version        : package version
          - duckdb_version        : runtime duckdb library version
          - embedding_model       : MODEL_NAME at boot time
          - embedding_dim         : effective dim of memories.embedding
          - embedding_stale       : '1' if drift detected, else '0'
          - last_boot_at          : ISO8601

        Side effect: stores the *previous* duckdb_version on ``self`` as
        ``previous_duckdb_version`` so the maintenance hook can detect
        upgrade-driven file format changes before they cause silent issues.
        """
        try:
            from . import __version__ as engram_version
        except Exception:
            engram_version = "0.0.0"
        try:
            duckdb_version = duckdb.__version__
        except Exception:
            duckdb_version = "unknown"
        boot_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        # Capture previous values BEFORE the upsert overwrites them.
        self.previous_duckdb_version = self.get_meta("duckdb_version")
        self.current_duckdb_version = duckdb_version

        rows = [
            ("schema_version", str(ENGRAM_SCHEMA_VERSION)),
            ("engram_version", engram_version),
            ("duckdb_version", duckdb_version),
            ("embedding_model", MODEL_NAME),
            ("embedding_dim", str(self._dim)),
            ("embedding_stale", "1" if self._embedding_stale else "0"),
            ("last_boot_at", boot_ts),
        ]
        for key, value in rows:
            try:
                self.conn.execute(
                    """INSERT INTO engram_meta (key, value)
                       VALUES (?, ?)
                       ON CONFLICT (key) DO UPDATE SET value = excluded.value,
                                                       updated_at = now()""",
                    [key, value],
                )
            except Exception as exc:
                log.debug("engram_meta upsert failed for %s (non-fatal): %s", key, exc)

    def get_meta(self, key: str) -> str | None:
        try:
            row = self.conn.execute(
                "SELECT value FROM engram_meta WHERE key = ?", [key]
            ).fetchone()
        except Exception:
            return None
        return row[0] if row else None

    def all_meta(self) -> dict[str, str]:
        try:
            rows = self.conn.execute(
                "SELECT key, value FROM engram_meta"
            ).fetchall()
        except Exception:
            return {}
        return {r[0]: r[1] for r in rows}

    def _rebuild_fts_index(self):
        """Rebuild the FTS index from scratch to include all current data.

        After rebuilding we immediately CHECKPOINT so the FTS internal
        tables (``fts_main_memories*``) are flushed into the main DB file.
        DuckDB 1.5.x cannot replay FTS WAL entries when the extension is
        not yet loaded, so a stale WAL containing FTS ops causes
        ``INTERNAL Error: GetDefaultDatabase`` on cold start after a crash.
        """
        try:
            self.conn.execute(
                "PRAGMA create_fts_index('memories', 'id', 'content', overwrite=1)"
            )
            self._fts_available = True
            self._fts_dirty = False
            log.info("FTS index rebuilt successfully")
            self._checkpoint_after_extension_op("FTS rebuild")
        except Exception as e:
            self._fts_available = False
            log.warning("FTS index rebuild failed: %s — BM25 search unavailable", e)

    def _ensure_fts_fresh(self):
        """Rebuild FTS index if writes have occurred since last rebuild."""
        if self._fts_available and self._fts_dirty:
            self._rebuild_fts_index()

    def _init_vss(self):
        """Load VSS extension and create HNSW index if beneficial.

        After loading we CHECKPOINT for the same reason as FTS: the VSS
        ``SET hnsw_enable_experimental_persistence`` and any HNSW index
        creation write extension-internal WAL entries that DuckDB 1.5.x
        cannot replay on a cold start.
        """
        self._vss_available = False
        try:
            self.conn.execute("INSTALL vss")
            self.conn.execute("LOAD vss")
            self.conn.execute("SET hnsw_enable_experimental_persistence = true")
            self._vss_available = True
            log.info("VSS extension loaded")
            self._ensure_hnsw_index()
            self._checkpoint_after_extension_op("VSS init")
        except Exception as e:
            log.info("VSS extension unavailable: %s — using brute-force vector search", e)

    def _ensure_hnsw_index(self):
        """Create HNSW index on embedding column when memory count exceeds threshold."""
        if not self._vss_available:
            return
        try:
            total = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            if total < 1000:
                log.debug("Only %d memories, skipping HNSW index (threshold: 1000)", total)
                return

            existing = self.conn.execute(
                "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'memories'"
            ).fetchall()
            hnsw_names = [r[0] for r in existing if "hnsw" in r[0].lower()]
            if hnsw_names:
                log.debug("HNSW index already exists: %s", hnsw_names)
                return

            self.conn.execute(
                "CREATE INDEX memories_hnsw_idx ON memories USING HNSW (embedding) WITH (metric = 'cosine')"
            )
            log.info("HNSW index created on memories.embedding (%d rows)", total)
        except Exception as e:
            log.warning("HNSW index creation failed (non-fatal): %s", e)

    def _float_cast(self) -> str:
        return f"FLOAT[{self._dim}]"

    def _validate_embedding(self, embedding: list[float]) -> None:
        if len(embedding) != self._dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dim}, got {len(embedding)}"
            )

    def insert(
        self,
        content: str,
        embedding: list[float],
        importance: float = 0.5,
        category: str = "fact",
        user_id: str = "default",
        metadata: dict | None = None,
    ) -> int:
        self._assert_writable()
        self._check_conn()
        self._validate_embedding(embedding)
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        cast = self._float_cast()
        result = self.conn.execute(
            f"""
            INSERT INTO memories (user_id, content, embedding, importance, category, metadata)
            VALUES (?, ?, ?::{cast}, ?, ?, ?::JSON)
            RETURNING id
            """,
            [user_id, content, embedding, importance, category, meta_json],
        ).fetchone()
        self._fts_dirty = True
        memory_id = result[0]
        # Tier 2 event — embedding intentionally omitted (Tier 3 cache).
        try:
            self._emit_event("memory.store", {
                "memory_id": memory_id,
                "user_id": user_id,
                "content": content,
                "importance": importance,
                "category": category,
                "metadata": metadata or {},
            })
        except Exception as exc:
            log.warning("memory.store event log append failed: %s", exc)
        return memory_id

    def update(
        self,
        memory_id: int,
        content: str,
        embedding: list[float],
        importance: float | None = None,
        metadata: dict | None = None,
    ):
        self._assert_writable()
        self._validate_embedding(embedding)
        cast = self._float_cast()
        sets = [f"content = ?", f"embedding = ?::{cast}", "last_accessed_at = now()"]
        params: list = [content, embedding]
        if importance is not None:
            sets.append("importance = ?")
            params.append(importance)
        if metadata is not None:
            sets.append("metadata = ?::JSON")
            params.append(json.dumps(metadata, ensure_ascii=False))
        params.append(memory_id)
        self.conn.execute(
            f"UPDATE memories SET {', '.join(sets)} WHERE id = ?",
            params,
        )
        self._fts_dirty = True
        try:
            self._emit_event("memory.update", {
                "memory_id": memory_id,
                "content": content,
                "importance": importance,
                "metadata": metadata,
            })
        except Exception as exc:
            log.warning("memory.update event log append failed: %s", exc)

    def bump_recall(self, memory_id: int):
        self.conn.execute(
            """
            UPDATE memories
            SET recall_count = recall_count + 1, last_accessed_at = now()
            WHERE id = ?
            """,
            [memory_id],
        )

    def log_session_recall(self, session_id: str, memory_ids: list[int], user_id: str = "default"):
        if not memory_ids:
            return
        self._assert_writable()
        rows = [[session_id, mid, user_id] for mid in memory_ids]
        self.conn.executemany(
            "INSERT INTO session_memory_log (session_id, memory_id, user_id) VALUES (?, ?, ?)",
            rows,
        )
        # Tier 1 event — session.memory_recall is part of runtime narrative.
        self._emit_event("session.memory_recall", {
            "session_id": session_id,
            "user_id": user_id,
            "memory_ids": list(memory_ids),
        })

    def get_session_memories(self, session_id: str, user_id: str = "default") -> list[int]:
        rows = self._fetchall_dicts(
            """
            SELECT DISTINCT memory_id FROM session_memory_log
            WHERE session_id = ? AND user_id = ?
            ORDER BY memory_id
            """,
            [session_id, user_id],
        )
        return [r["memory_id"] for r in rows]

    def adjust_importance_batch(self, memory_ids: list[int], delta: float) -> int:
        if not memory_ids:
            return 0
        placeholders = ",".join("?" for _ in memory_ids)
        row = self.conn.execute(
            f"""
            UPDATE memories
            SET importance = CASE
                WHEN importance + ? > 1.0 THEN 1.0
                WHEN importance + ? < 0.0 THEN 0.0
                ELSE importance + ?
            END
            WHERE id IN ({placeholders})
            RETURNING id
            """,
            [delta, delta, delta] + memory_ids,
        ).fetchall()
        return len(row)

    def log_session_outcome(self, session_id: str, outcome: str, user_id: str = "default"):
        self._assert_writable()
        self.conn.execute(
            "INSERT INTO session_outcome_log (session_id, user_id, outcome) VALUES (?, ?, ?)",
            [session_id, user_id, outcome],
        )
        self._emit_event("session.outcome", {
            "session_id": session_id,
            "user_id": user_id,
            "outcome": outcome,
        })

    def get_memory_failure_count(self, memory_ids: list[int], user_id: str = "default") -> dict[int, int]:
        """Count how many failed sessions each memory was involved in."""
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        rows = self._fetchall_dicts(
            f"""
            SELECT sml.memory_id, COUNT(DISTINCT sol.session_id) AS fail_count
            FROM session_memory_log sml
            JOIN session_outcome_log sol
              ON sml.session_id = sol.session_id AND sml.user_id = sol.user_id
            WHERE sol.outcome = 'failure'
              AND sml.user_id = ?
              AND sml.memory_id IN ({placeholders})
            GROUP BY sml.memory_id
            """,
            [user_id] + memory_ids,
        )
        return {r["memory_id"]: r["fail_count"] for r in rows}

    def get_memory_outcome_counts(self, memory_ids: list[int],
                                   user_id: str = "default") -> dict[int, dict[str, int]]:
        """Count success and failure sessions for each memory.

        Returns {memory_id: {"success": N, "failure": M}}.
        """
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        rows = self._fetchall_dicts(
            f"""
            SELECT sml.memory_id, sol.outcome,
                   COUNT(DISTINCT sol.session_id) AS cnt
            FROM session_memory_log sml
            JOIN session_outcome_log sol
              ON sml.session_id = sol.session_id AND sml.user_id = sol.user_id
            WHERE sml.user_id = ?
              AND sml.memory_id IN ({placeholders})
            GROUP BY sml.memory_id, sol.outcome
            """,
            [user_id] + memory_ids,
        )
        result: dict[int, dict[str, int]] = {}
        for r in rows:
            mid = r["memory_id"]
            if mid not in result:
                result[mid] = {"success": 0, "failure": 0}
            outcome = r["outcome"]
            if outcome in ("success", "failure"):
                result[mid][outcome] = r["cnt"]
        return result

    def delete(self, memory_id: int):
        self._assert_writable()
        self.conn.execute("DELETE FROM memories WHERE id = ?", [memory_id])
        self._fts_dirty = True
        try:
            self._emit_event("memory.delete", {"memory_id": memory_id})
        except Exception as exc:
            log.warning("memory.delete event log append failed: %s", exc)

    def get_by_id(self, memory_id: int) -> MemoryRow | None:
        row = self._fetchone_dict(
            """
            SELECT id, user_id, content, importance, category,
                   recall_count, created_at, last_accessed_at, metadata
            FROM memories WHERE id = ?
            """,
            [memory_id],
        )
        if not row:
            return None
        return _row_to_memory(row)

    def get_all(self, user_id: str = "default") -> list[MemoryRow]:
        rows = self._fetchall_dicts(
            """
            SELECT id, user_id, content, importance, category,
                   recall_count, created_at, last_accessed_at, metadata
            FROM memories WHERE user_id = ?
            """,
            [user_id],
        )
        return [_row_to_memory(r) for r in rows]

    def search_vector(
        self,
        query_embedding: list[float],
        user_id: str = "default",
        top_k: int = 20,
        threshold: float = SIMILARITY_LOW,
    ) -> list[MemoryRow]:
        self._validate_embedding(query_embedding)
        threshold = max(0.0, min(1.0, float(threshold)))
        top_k = max(1, min(1000, int(top_k)))
        cast = self._float_cast()
        rows = self._fetchall_dicts(
            f"""
            WITH scored AS (
                SELECT id, user_id, content, importance, category,
                       recall_count, created_at, last_accessed_at, metadata,
                       array_cosine_similarity(embedding, ?::{cast}) AS sim
                FROM memories
                WHERE user_id = ?
            )
            SELECT id, user_id, content, importance, category,
                   recall_count, created_at, last_accessed_at, metadata, sim
            FROM scored
            WHERE sim >= ?
            ORDER BY sim DESC
            LIMIT ?
            """,
            [query_embedding, user_id, threshold, top_k],
        )
        results = []
        for r in rows:
            m = _row_to_memory(r)
            m.similarity = r["sim"]
            results.append(m)
        return results

    def search_fts(
        self,
        query: str,
        user_id: str = "default",
        top_k: int = 20,
    ) -> list[MemoryRow]:
        self._ensure_fts_fresh()
        if not self._fts_available:
            return []
        try:
            rows = self._fetchall_dicts(
                """
                SELECT m.id, m.user_id, m.content, m.importance, m.category,
                       m.recall_count, m.created_at, m.last_accessed_at,
                       m.metadata, fts.score
                FROM (
                    SELECT id, fts_main_memories.match_bm25(id, ?) AS score
                    FROM memories
                ) fts
                JOIN memories m ON m.id = fts.id
                WHERE m.user_id = ? AND fts.score IS NOT NULL
                ORDER BY fts.score DESC
                LIMIT ?
                """,
                [query, user_id, top_k],
            )
            results = []
            for r in rows:
                m = _row_to_memory(r)
                m.bm25_score = r["score"] if r["score"] else 0.0
                results.append(m)
            return results
        except Exception as e:
            log.error("FTS search failed (query=%r): %s", query, e)
            return []

    def search_similar_for_dedup(
        self,
        query_embedding: list[float],
        user_id: str = "default",
        top_k: int = 10,
        threshold: float = DEDUP_SEARCH_THRESHOLD,
    ) -> list[tuple[int, str, list[float]]]:
        cast = self._float_cast()
        rows = self._fetchall_dicts(
            f"""
            WITH scored AS (
                SELECT id, content, embedding,
                       array_cosine_similarity(embedding, ?::{cast}) AS sim
                FROM memories
                WHERE user_id = ?
            )
            SELECT id, content, embedding, sim
            FROM scored
            WHERE sim >= ?
            ORDER BY sim DESC
            LIMIT ?
            """,
            [query_embedding, user_id, threshold, top_k],
        )
        return [(r["id"], r["content"], list(r["embedding"])) for r in rows]

    def get_embedding(self, memory_id: int) -> list[float] | None:
        row = self.conn.execute(
            "SELECT embedding FROM memories WHERE id = ?", [memory_id]
        ).fetchone()
        if not row or not row[0]:
            return None
        return list(row[0])

    def count(self, user_id: str = "default") -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE user_id = ?", [user_id]
        ).fetchone()
        return row[0]

    def _check_conn(self):
        """Raise if connection has been closed."""
        if self.conn is None:
            raise RuntimeError("MemoryDB connection is closed")

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def get_all_user_ids(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT user_id FROM memories"
        ).fetchall()
        return [r[0] for r in rows]

    def get_stats_aggregate(self, user_id: str = "default") -> dict:
        """Return category counts and total count via SQL aggregation."""
        rows = self.conn.execute(
            """
            SELECT category, COUNT(*) as cnt
            FROM memories WHERE user_id = ?
            GROUP BY category
            """,
            [user_id],
        ).fetchall()
        categories = {r[0]: r[1] for r in rows}
        total = sum(categories.values())
        return {"total": total, "categories": categories}

    def get_metadata_for_stats(self, user_id: str = "default",
                               types: tuple[str, ...] | None = None) -> list[dict]:
        """Return only metadata JSON for engineering stats (lightweight).

        When *types* is given, only rows whose metadata->type matches are returned,
        avoiding a full-table scan of all memories.
        """
        if types:
            placeholders = ",".join("?" for _ in types)
            rows = self.conn.execute(
                f"""
                SELECT metadata FROM memories
                WHERE user_id = ?
                  AND json_extract_string(metadata, '$.type') IN ({placeholders})
                """,
                [user_id] + list(types),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT metadata FROM memories WHERE user_id = ?",
                [user_id],
            ).fetchall()
        return [_parse_metadata(r[0]) for r in rows]

    def get_strength_data(self, user_id: str = "default") -> list[tuple]:
        """Return (category, importance, last_accessed_at, recall_count) for strength calc."""
        rows = self._fetchall_dicts(
            """
            SELECT category, importance, last_accessed_at, recall_count
            FROM memories WHERE user_id = ?
            """,
            [user_id],
        )
        return [(r["category"], r["importance"], r["last_accessed_at"], r["recall_count"]) for r in rows]

    def get_by_ids_batch(self, memory_ids: list[int]) -> dict[int, MemoryRow]:
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        rows = self._fetchall_dicts(
            f"""
            SELECT id, user_id, content, importance, category,
                   recall_count, created_at, last_accessed_at, metadata
            FROM memories WHERE id IN ({placeholders})
            """,
            memory_ids,
        )
        return {r["id"]: _row_to_memory(r) for r in rows}

    def get_embeddings_batch(self, memory_ids: list[int]) -> dict[int, list[float]]:
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        rows = self._fetchall_dicts(
            f"SELECT id, embedding FROM memories WHERE id IN ({placeholders})",
            memory_ids,
        )
        return {r["id"]: list(r["embedding"]) for r in rows if r["embedding"]}

    def get_metadata_batch(self, memory_ids: list[int]) -> dict[int, dict]:
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        rows = self._fetchall_dicts(
            f"SELECT id, metadata FROM memories WHERE id IN ({placeholders})",
            memory_ids,
        )
        return {r["id"]: _parse_metadata(r["metadata"]) for r in rows}

    def get_neighbors_batch(
        self, memory_ids: list[int]
    ) -> dict[int, tuple[MemoryRow, list[float]]]:
        """Single query returning both row data and embedding for neighbor expansion."""
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        rows = self._fetchall_dicts(
            f"""
            SELECT id, user_id, content, importance, category,
                   recall_count, created_at, last_accessed_at, metadata,
                   embedding
            FROM memories WHERE id IN ({placeholders})
            """,
            memory_ids,
        )
        result: dict[int, tuple[MemoryRow, list[float]]] = {}
        for r in rows:
            m = _row_to_memory(r)
            emb = list(r["embedding"]) if r["embedding"] else []
            result[r["id"]] = (m, emb)
        return result

    # --- Task CRUD ---

    def _row_to_task(self, row: dict) -> TaskRow:
        return TaskRow(
            id=row["id"],
            name=row["name"],
            goal=row["goal"],
            status=row["status"],
            user_id=row["user_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=_parse_metadata(row.get("metadata")),
        )

    def create_task(self, name: str, goal: str = "", status: str = "planning",
                    user_id: str = "default", metadata: dict | None = None) -> int:
        self._assert_writable()
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        result = self.conn.execute(
            """
            INSERT INTO tasks (name, goal, status, user_id, metadata)
            VALUES (?, ?, ?, ?, ?::JSON)
            RETURNING id
            """,
            [name, goal, status, user_id, meta_json],
        ).fetchone()
        task_id = result[0]
        # Tier 1 — task lifecycle is critical runtime state, MUST be in event log.
        self._emit_event("task.create", {
            "task_id": task_id,
            "name": name,
            "goal": goal,
            "status": status,
            "user_id": user_id,
            "metadata": metadata or {},
        })
        return task_id

    def get_task(self, task_id: int) -> TaskRow | None:
        row = self._fetchone_dict(
            """
            SELECT id, name, goal, status, user_id, created_at, updated_at, metadata
            FROM tasks WHERE id = ?
            """,
            [task_id],
        )
        if not row:
            return None
        return self._row_to_task(row)

    def update_task(self, task_id: int, status: str | None = None,
                    goal: str | None = None, metadata: dict | None = None) -> bool:
        self._assert_writable()
        sets = ["updated_at = now()"]
        params: list = []
        if status is not None:
            sets.append("status = ?")
            params.append(status)
        if goal is not None:
            sets.append("goal = ?")
            params.append(goal)
        if metadata is not None:
            sets.append("metadata = ?::JSON")
            params.append(json.dumps(metadata, ensure_ascii=False))
        params.append(task_id)
        row = self.conn.execute(
            f"UPDATE tasks SET {', '.join(sets)} WHERE id = ? RETURNING id",
            params,
        ).fetchone()
        updated = row is not None
        if updated:
            self._emit_event("task.update", {
                "task_id": task_id,
                "status": status,
                "goal": goal,
                "metadata": metadata,
            })
        return updated

    def list_tasks(self, user_id: str = "default", status: str | None = None) -> list[TaskRow]:
        if status:
            rows = self._fetchall_dicts(
                """
                SELECT id, name, goal, status, user_id, created_at, updated_at, metadata
                FROM tasks WHERE user_id = ? AND status = ?
                ORDER BY updated_at DESC
                """,
                [user_id, status],
            )
        else:
            rows = self._fetchall_dicts(
                """
                SELECT id, name, goal, status, user_id, created_at, updated_at, metadata
                FROM tasks WHERE user_id = ?
                ORDER BY updated_at DESC
                """,
                [user_id],
            )
        return [self._row_to_task(r) for r in rows]

    def get_latest_interrupt_checkpoint(self, user_id: str = "default") -> dict | None:
        """Return the most recent auto-interrupt checkpoint across all tasks.

        Looks for AUTO_SAVE checkpoints triggered by process_exit — these are
        the ones written by trigger_interrupt_checkpoint() on SIGTERM.
        Returns a raw dict (same shape as checkpoint._row_to_checkpoint) or None.
        """
        from .checkpoint import _row_to_checkpoint
        row = self._fetchone_dict(
            """
            SELECT * FROM checkpoints
            WHERE user_id = ?
              AND checkpoint_reason = 'AUTO_SAVE'
              AND triggered_by_event = 'process_exit'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [user_id],
        )
        if not row:
            return None
        return _row_to_checkpoint(row)

    def get_task_memories(self, task_id: int, user_id: str = "default") -> list[MemoryRow]:
        """Get all memories associated with a task via metadata.task_id."""
        rows = self._fetchall_dicts(
            """
            SELECT id, user_id, content, importance, category,
                   recall_count, created_at, last_accessed_at, metadata
            FROM memories
            WHERE user_id = ? AND json_extract_string(metadata, '$.task_id') = ?
            ORDER BY created_at DESC
            """,
            [user_id, str(task_id)],
        )
        return [_row_to_memory(r) for r in rows]

    def get_failures_by_component(self, component: str, user_id: str = "default",
                                  limit: int = 5) -> list[MemoryRow]:
        """Get failure memories for a given component, most recent first."""
        rows = self._fetchall_dicts(
            """
            SELECT id, user_id, content, importance, category,
                   recall_count, created_at, last_accessed_at, metadata
            FROM memories
            WHERE user_id = ?
              AND category = 'failure'
              AND json_extract_string(metadata, '$.component') = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            [user_id, component, limit],
        )
        return [_row_to_memory(r) for r in rows]

    # --- Session Lifecycle ---

    def upsert_session(self, session_id: str, user_id: str = "default"):
        """Register a new session or refresh its heartbeat."""
        self._assert_writable()
        # Distinguish creation vs heartbeat for the event log:
        # only the first call per session_id should emit session.start.
        existing = self.conn.execute(
            "SELECT 1 FROM session_lifecycle WHERE session_id = ?",
            [session_id],
        ).fetchone()
        self.conn.execute(
            """INSERT INTO session_lifecycle (session_id, user_id)
            VALUES (?, ?)
            ON CONFLICT (session_id) DO UPDATE SET last_active_at = now()""",
            [session_id, user_id],
        )
        if existing is None:
            self._emit_event("session.start", {
                "session_id": session_id,
                "user_id": user_id,
            })

    def end_session(self, session_id: str, end_type: str = "handoff",
                    interruption_reason: str | None = None,
                    interruption_context: dict | None = None):
        """Mark a session as ended.

        Args:
            session_id: Session to close.
            end_type: One of handoff / outcome / process_exit / interrupted.
            interruption_reason: One of VALID_INTERRUPTION_REASONS (v0.12).
                Only meaningful when end_type == 'interrupted' or 'process_exit'.
            interruption_context: Optional structured context (e.g. last error,
                token count, failure count).
        """
        self._assert_writable()
        context_json = json.dumps(interruption_context or {}, ensure_ascii=False)
        self.conn.execute(
            """UPDATE session_lifecycle
            SET ended_at = now(), last_active_at = now(), end_type = ?,
                interruption_reason = ?, interruption_context = ?::JSON
            WHERE session_id = ?""",
            [end_type, interruption_reason, context_json, session_id],
        )
        event_payload: dict = {
            "session_id": session_id,
            "end_type": end_type,
        }
        if interruption_reason:
            event_payload["interruption_reason"] = interruption_reason
        if interruption_context:
            event_payload["interruption_context"] = interruption_context
        self._emit_event("session.end", event_payload)

    def get_interrupted_sessions(self, user_id: str = "default",
                                 stale_minutes: int = 30) -> list[dict]:
        """Find sessions that started but never ended, or ended with interruption.

        Returns both:
        - Sessions with ended_at IS NULL and stale heartbeat (not yet classified).
        - Sessions already classified as interrupted (end_type='interrupted')
          from the last 24 hours, so the new Agent can see recent interruptions
          with their taxonomy.
        """
        # Unclosed stale sessions
        unclosed = self._fetchall_dicts(
            """SELECT session_id, started_at, last_active_at,
                      interruption_reason, interruption_context
            FROM session_lifecycle
            WHERE user_id = ? AND ended_at IS NULL
              AND last_active_at < now() - INTERVAL '1 MINUTE' * ?
            ORDER BY last_active_at DESC LIMIT 5""",
            [user_id, stale_minutes],
        )
        # Recently classified interrupted sessions (last 24h)
        recent_interrupted = self._fetchall_dicts(
            """SELECT session_id, started_at, last_active_at,
                      interruption_reason, interruption_context
            FROM session_lifecycle
            WHERE user_id = ? AND end_type = 'interrupted'
              AND ended_at > now() - INTERVAL '24 HOURS'
              AND interruption_reason IS NOT NULL
            ORDER BY ended_at DESC LIMIT 3""",
            [user_id],
        )
        seen = {r["session_id"] for r in unclosed}
        for row in recent_interrupted:
            if row["session_id"] not in seen:
                unclosed.append(row)
        return unclosed

    def get_session_activity_summary(self, session_id: str,
                                     user_id: str = "default") -> dict:
        """Reconstruct what happened in a session from existing data."""
        recalled_ids = self.get_session_memories(session_id, user_id)

        session_row = self._fetchone_dict(
            "SELECT started_at, last_active_at FROM session_lifecycle WHERE session_id = ?",
            [session_id],
        )
        recent_writes: list[dict] = []
        active_tasks: list[dict] = []

        if session_row:
            started = session_row["started_at"]
            last_active = session_row["last_active_at"]
            recent_writes = self._fetchall_dicts(
                """SELECT id, content, category,
                       json_extract_string(metadata, '$.type') AS mem_type
                FROM memories
                WHERE user_id = ? AND created_at BETWEEN ? AND ?
                ORDER BY created_at DESC LIMIT 10""",
                [user_id, started, last_active],
            )
            active_tasks = self._fetchall_dicts(
                """SELECT id, name, status, goal FROM tasks
                WHERE user_id = ? AND status NOT IN ('done', 'cancelled')
                ORDER BY updated_at DESC LIMIT 5""",
                [user_id],
            )

        return {
            "session_id": session_id,
            "recalled_memory_ids": recalled_ids,
            "memories_written": [
                {"id": w["id"], "snippet": w["content"][:100],
                 "type": w.get("mem_type", "")}
                for w in recent_writes
            ],
            "active_tasks": [
                {"id": t["id"], "name": t["name"], "status": t["status"]}
                for t in active_tasks
            ],
        }

    def cleanup_stale_sessions(self, user_id: str = "default",
                               stale_minutes: int = 30):
        """Mark old interrupted sessions as ended and classify interruption reason.

        Heuristic classification for sessions that were never explicitly closed:
        - Sessions with recent track_failure events → tool_failure
        - Very short sessions (< 2 min active) → crash (likely killed before atexit)
        - All others → user_away (most common: user closed the tab/terminal)
        """
        self._assert_writable()
        stale_rows = self._fetchall_dicts(
            """SELECT session_id, started_at, last_active_at
            FROM session_lifecycle
            WHERE user_id = ? AND ended_at IS NULL
              AND last_active_at < now() - INTERVAL '1 MINUTE' * ?""",
            [user_id, stale_minutes],
        )
        for row in stale_rows:
            reason = self._classify_stale_session(row, user_id)
            context = {"classified_by": "cleanup_stale_sessions"}
            context_json = json.dumps(context, ensure_ascii=False)
            self.conn.execute(
                """UPDATE session_lifecycle
                SET ended_at = last_active_at, end_type = 'interrupted',
                    interruption_reason = ?,
                    interruption_context = ?::JSON
                WHERE session_id = ?""",
                [reason, context_json, row["session_id"]],
            )
            self._emit_event("session.end", {
                "session_id": row["session_id"],
                "end_type": "interrupted",
                "interruption_reason": reason,
            })

    def _classify_stale_session(self, session_row: dict,
                                user_id: str) -> str:
        """Heuristic classification of why a stale session was interrupted.

        Signal priority:
        1. Recent failures in event log → tool_failure
        2. Very short session (< 2 min) → crash
        3. Default → user_away
        """
        session_id = session_row["session_id"]
        started = session_row["started_at"]
        last_active = session_row["last_active_at"]

        # Check for recent failure events in the session window
        try:
            failure_count = self.conn.execute(
                """SELECT count(*) FROM memories
                WHERE user_id = ? AND category = 'failure'
                  AND created_at BETWEEN ? AND ?""",
                [user_id, started, last_active],
            ).fetchone()[0]
            if failure_count >= 2:
                return INTERRUPTION_TOOL_FAILURE
        except Exception:
            pass

        # Very short session → likely crash (killed before any meaningful work)
        try:
            if started and last_active:
                duration_seconds = (last_active - started).total_seconds()
                if duration_seconds < 120:
                    return INTERRUPTION_CRASH
        except Exception:
            pass

        return INTERRUPTION_USER_AWAY
