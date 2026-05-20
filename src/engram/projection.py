"""Tier 2: Runtime State Store — SQLite WAL projection layer.

v0.18: Operationally durable state for tasks, checkpoints, executions, sessions.
Uses SQLite in WAL mode for:
  - 1 writer + N concurrent readers (no more DuckDB lock conflicts)
  - Fast restore (no replay needed for normal boot)
  - Operational durability (not disposable, not just a cache)

This module provides the same interface as the Tier2 methods in db.py,
allowing transparent migration from DuckDB to SQLite.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger("engram.projection")

ENGRAM_DIR = os.path.join(os.path.expanduser("~"), ".engram")
DEFAULT_SQLITE_PATH = os.path.join(ENGRAM_DIR, "runtime_state.sqlite")


@dataclass
class TaskRow:
    id: int
    name: str
    goal: str
    status: str
    user_id: str
    created_at: Any
    updated_at: Any
    metadata: dict | None = None
    execution_id: str | None = None
    previous_task_id: int | None = None
    checkpoint_id: str | None = None
    attempt: int = 1
    parent_task_id: int | None = None
    retry_of_task_id: int | None = None


# ---- Schema ----

_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA busy_timeout=5000;

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    goal TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'in_progress',
    user_id TEXT NOT NULL DEFAULT 'default',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    metadata TEXT DEFAULT '{}',
    execution_id TEXT,
    previous_task_id INTEGER,
    checkpoint_id TEXT,
    attempt INTEGER NOT NULL DEFAULT 1,
    parent_task_id INTEGER,
    retry_of_task_id INTEGER,
    latest_checkpoint_version INTEGER DEFAULT 0,
    checkpoint_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL,
    version INTEGER NOT NULL,
    parent_version INTEGER,
    kind TEXT NOT NULL DEFAULT 'auto',
    checkpoint_reason TEXT NOT NULL,
    triggered_by_event TEXT,
    goal TEXT DEFAULT '',
    completed TEXT DEFAULT '[]',
    in_progress TEXT DEFAULT '[]',
    blocked TEXT DEFAULT '[]',
    preferred_next TEXT DEFAULT '[]',
    must_not_redo TEXT DEFAULT '[]',
    must_preserve TEXT DEFAULT '[]',
    working_set TEXT DEFAULT '{}',
    active_constraints TEXT DEFAULT '[]',
    blocked_reasons TEXT DEFAULT '[]',
    state_diff TEXT DEFAULT '{}',
    source_session_id TEXT,
    source_memory_id INTEGER,
    continuation_confidence REAL DEFAULT 0.0,
    confidence_breakdown TEXT DEFAULT '{}',
    failure_signature TEXT,
    user_id TEXT NOT NULL DEFAULT 'default',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_checkpoints_task ON checkpoints(task_id, user_id, version);

CREATE TABLE IF NOT EXISTS execution_sessions (
    execution_id TEXT PRIMARY KEY,
    root_goal TEXT NOT NULL,
    origin_checkpoint TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    user_id TEXT NOT NULL DEFAULT 'default',
    last_active_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    continuity_score REAL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS session_lifecycle (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'default',
    started_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    last_active_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    end_type TEXT,
    interruption_reason TEXT,
    interruption_context TEXT DEFAULT '{}',
    outcome TEXT,
    outcome_notes TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""


def migrate_from_duckdb(duckdb_conn, sqlite_store: "RuntimeStateStore") -> dict:
    """Migrate Tier 2 data from DuckDB to SQLite RuntimeStateStore.

    Returns a summary dict with counts of migrated rows per table.
    Only migrates if SQLite is empty (tasks table has 0 rows).
    """
    # Check if SQLite already has data
    existing = sqlite_store._conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    if existing > 0:
        return {"skipped": True, "reason": "SQLite already has data"}

    counts = {}

    # Migrate tasks
    try:
        rows = duckdb_conn.execute(
            """SELECT id, name, goal, status, user_id, created_at, updated_at,
                      metadata, execution_id, previous_task_id, checkpoint_id,
                      attempt, parent_task_id, retry_of_task_id
               FROM tasks ORDER BY id"""
        ).fetchall()
        for r in rows:
            sqlite_store._conn.execute(
                """INSERT OR IGNORE INTO tasks
                   (id, name, goal, status, user_id, created_at, updated_at,
                    metadata, execution_id, previous_task_id, checkpoint_id,
                    attempt, parent_task_id, retry_of_task_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                list(r),
            )
        counts["tasks"] = len(rows)
    except Exception as exc:
        log.warning("migrate tasks failed: %s", exc)
        counts["tasks"] = 0

    # Migrate execution_sessions
    try:
        rows = duckdb_conn.execute(
            """SELECT execution_id, root_goal, origin_checkpoint, status,
                      user_id, last_active_at, continuity_score, created_at
               FROM execution_sessions ORDER BY created_at"""
        ).fetchall()
        for r in rows:
            sqlite_store._conn.execute(
                """INSERT OR IGNORE INTO execution_sessions
                   (execution_id, root_goal, origin_checkpoint, status,
                    user_id, last_active_at, continuity_score, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                list(r),
            )
        counts["execution_sessions"] = len(rows)
    except Exception as exc:
        log.warning("migrate execution_sessions failed: %s", exc)
        counts["execution_sessions"] = 0

    # Migrate checkpoints
    try:
        rows = duckdb_conn.execute(
            """SELECT id, task_id, version, parent_version, kind,
                      checkpoint_reason, triggered_by_event,
                      goal, completed, in_progress, blocked,
                      preferred_next, must_not_redo, must_preserve, working_set,
                      active_constraints, blocked_reasons,
                      state_diff, source_session_id, source_memory_id,
                      continuation_confidence, confidence_breakdown,
                      failure_signature, user_id, created_at
               FROM checkpoints ORDER BY id"""
        ).fetchall()
        for r in rows:
            row_list = list(r)
            # Convert DuckDB JSON objects to strings for SQLite
            for i in range(len(row_list)):
                if isinstance(row_list[i], (dict, list)):
                    row_list[i] = json.dumps(row_list[i])
            sqlite_store._conn.execute(
                """INSERT OR IGNORE INTO checkpoints
                   (id, task_id, version, parent_version, kind,
                    checkpoint_reason, triggered_by_event,
                    goal, completed, in_progress, blocked,
                    preferred_next, must_not_redo, must_preserve, working_set,
                    active_constraints, blocked_reasons,
                    state_diff, source_session_id, source_memory_id,
                    continuation_confidence, confidence_breakdown,
                    failure_signature, user_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                row_list,
            )
        counts["checkpoints"] = len(rows)
    except Exception as exc:
        log.warning("migrate checkpoints failed: %s", exc)
        counts["checkpoints"] = 0

    # Migrate session_lifecycle
    try:
        rows = duckdb_conn.execute(
            """SELECT session_id, user_id, started_at, last_active_at,
                      end_type, interruption_reason, interruption_context
               FROM session_lifecycle ORDER BY started_at"""
        ).fetchall()
        for r in rows:
            row_list = list(r)
            for i in range(len(row_list)):
                if isinstance(row_list[i], (dict, list)):
                    row_list[i] = json.dumps(row_list[i])
            sqlite_store._conn.execute(
                """INSERT OR IGNORE INTO session_lifecycle
                   (session_id, user_id, started_at, last_active_at,
                    end_type, interruption_reason, interruption_context)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                row_list,
            )
        counts["session_lifecycle"] = len(rows)
    except Exception as exc:
        log.warning("migrate session_lifecycle failed: %s", exc)
        counts["session_lifecycle"] = 0

    sqlite_store._conn.commit()
    log.info("DuckDB → SQLite migration complete: %s", counts)
    return {"skipped": False, "counts": counts}


class RuntimeStateStore:
    """SQLite WAL-based Tier 2 storage for operational runtime state."""

    def __init__(self, db_path: str = DEFAULT_SQLITE_PATH):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA_SQL)

    def close(self) -> None:
        self._conn.close()

    @property
    def path(self) -> str:
        return self._db_path

    def is_empty(self) -> bool:
        """Check if the store has no data (for migration detection)."""
        count = self._conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        return count == 0

    # ---- Task CRUD ----

    def create_task(
        self,
        name: str,
        goal: str = "",
        status: str = "in_progress",
        user_id: str = "default",
        metadata: dict | None = None,
    ) -> int:
        meta_json = json.dumps(metadata or {})
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO tasks (name, goal, status, user_id, metadata)
                   VALUES (?, ?, ?, ?, ?)""",
                [name, goal, status, user_id, meta_json],
            )
            self._conn.commit()
            return cur.lastrowid

    def get_task(self, task_id: int) -> TaskRow | None:
        row = self._conn.execute(
            "SELECT * FROM tasks WHERE id = ?", [task_id]
        ).fetchone()
        if row is None:
            return None
        return self._row_to_task(row)

    def list_tasks(self, user_id: str = "default", status: str | None = None) -> list[TaskRow]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM tasks WHERE user_id = ? AND status = ? ORDER BY id DESC",
                [user_id, status],
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM tasks WHERE user_id = ? ORDER BY id DESC",
                [user_id],
            ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def update_task_checkpoint_cache(self, task_id: int, version: int) -> None:
        """Update the task's checkpoint cache fields after a new checkpoint."""
        with self._lock:
            self._conn.execute(
                """UPDATE tasks
                   SET latest_checkpoint_version = ?,
                       checkpoint_count = checkpoint_count + 1
                   WHERE id = ?""",
                [version, task_id],
            )
            self._conn.commit()

    def get_task_checkpoint_cache(self, task_id: int) -> tuple[int, int]:
        """Return (latest_checkpoint_version, checkpoint_count) for a task."""
        row = self._conn.execute(
            "SELECT latest_checkpoint_version, checkpoint_count FROM tasks WHERE id = ?",
            [task_id],
        ).fetchone()
        if row is None:
            return (0, 0)
        return (row["latest_checkpoint_version"] or 0, row["checkpoint_count"] or 0)

    def update_task(self, task_id: int, **kwargs) -> bool:
        allowed = {"name", "goal", "status", "metadata", "execution_id",
                   "previous_task_id", "checkpoint_id", "attempt",
                   "parent_task_id", "retry_of_task_id"}
        updates = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        if not updates:
            return False
        if "metadata" in updates and isinstance(updates["metadata"], dict):
            updates["metadata"] = json.dumps(updates["metadata"])
        updates["updated_at"] = datetime.now(timezone.utc).isoformat()

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [task_id]
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE tasks SET {set_clause} WHERE id = ?", values
            )
            self._conn.commit()
            return cur.rowcount > 0

    # ---- Execution CRUD ----

    def create_execution(
        self,
        execution_id: str,
        root_goal: str,
        user_id: str = "default",
        origin_checkpoint: str | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO execution_sessions
                   (execution_id, root_goal, user_id, origin_checkpoint)
                   VALUES (?, ?, ?, ?)""",
                [execution_id, root_goal, user_id, origin_checkpoint],
            )
            self._conn.commit()

    def get_execution(self, execution_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM execution_sessions WHERE execution_id = ?",
            [execution_id],
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def end_execution(self, execution_id: str, status: str = "completed") -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE execution_sessions SET status = ? WHERE execution_id = ?",
                [status, execution_id],
            )
            self._conn.commit()

    def get_active_executions(self, user_id: str = "default") -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM execution_sessions WHERE user_id = ? AND status = 'active'",
            [user_id],
        ).fetchall()
        return [dict(r) for r in rows]

    def get_execution_tasks(self, execution_id: str) -> list[TaskRow]:
        rows = self._conn.execute(
            "SELECT * FROM tasks WHERE execution_id = ? ORDER BY id",
            [execution_id],
        ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def create_task_in_execution(
        self,
        name: str,
        goal: str,
        execution_id: str,
        user_id: str = "default",
        attempt: int = 1,
        previous_task_id: int | None = None,
        retry_of_task_id: int | None = None,
        parent_task_id: int | None = None,
        checkpoint_id: str | None = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO tasks
                   (name, goal, status, user_id, execution_id, attempt,
                    previous_task_id, retry_of_task_id, parent_task_id, checkpoint_id)
                   VALUES (?, ?, 'in_progress', ?, ?, ?, ?, ?, ?, ?)""",
                [name, goal, user_id, execution_id, attempt,
                 previous_task_id, retry_of_task_id, parent_task_id, checkpoint_id],
            )
            self._conn.commit()
            return cur.lastrowid

    def get_retry_chain(self, task_id: int) -> list[TaskRow]:
        """Walk the retry chain backwards from task_id."""
        chain = []
        current_id = task_id
        seen = set()
        while current_id and current_id not in seen:
            seen.add(current_id)
            task = self.get_task(current_id)
            if task is None:
                break
            if task.retry_of_task_id:
                chain.append(task)
                current_id = task.retry_of_task_id
            else:
                break
        return chain

    # ---- Checkpoint CRUD ----

    def get_max_version(self, task_id: int, user_id: str = "default") -> int:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM checkpoints WHERE task_id = ? AND user_id = ?",
            [task_id, user_id],
        ).fetchone()
        return row[0] if row else 0

    def insert_checkpoint(self, **kwargs) -> int:
        """Insert a checkpoint row. Returns the row id."""
        fields = [
            "task_id", "version", "parent_version", "kind",
            "checkpoint_reason", "triggered_by_event",
            "goal", "completed", "in_progress", "blocked",
            "preferred_next", "must_not_redo", "must_preserve", "working_set",
            "active_constraints", "blocked_reasons",
            "state_diff", "source_session_id", "source_memory_id",
            "continuation_confidence", "confidence_breakdown",
            "failure_signature", "user_id",
        ]
        values = {}
        for f in fields:
            v = kwargs.get(f)
            if v is not None:
                if isinstance(v, (dict, list)):
                    values[f] = json.dumps(v)
                else:
                    values[f] = v

        cols = ", ".join(values.keys())
        placeholders = ", ".join("?" for _ in values)
        with self._lock:
            cur = self._conn.execute(
                f"INSERT INTO checkpoints ({cols}) VALUES ({placeholders})",
                list(values.values()),
            )
            self._conn.commit()
            return cur.lastrowid

    def get_latest_checkpoint(self, task_id: int, user_id: str = "default") -> dict | None:
        row = self._conn.execute(
            """SELECT * FROM checkpoints
               WHERE task_id = ? AND user_id = ?
               ORDER BY version DESC LIMIT 1""",
            [task_id, user_id],
        ).fetchone()
        if row is None:
            return None
        return self._row_to_checkpoint(row)

    def get_checkpoint_by_version(self, task_id: int, version: int, user_id: str = "default") -> dict | None:
        row = self._conn.execute(
            """SELECT * FROM checkpoints
               WHERE task_id = ? AND user_id = ? AND version = ?""",
            [task_id, user_id, version],
        ).fetchone()
        if row is None:
            return None
        return self._row_to_checkpoint(row)

    def list_checkpoints(self, task_id: int, user_id: str = "default", limit: int = 10) -> list[dict]:
        rows = self._conn.execute(
            """SELECT * FROM checkpoints
               WHERE task_id = ? AND user_id = ?
               ORDER BY version DESC LIMIT ?""",
            [task_id, user_id, limit],
        ).fetchall()
        return [self._row_to_checkpoint(r) for r in rows]

    def get_recent_checkpoint_reasons(self, task_id: int, user_id: str = "default", limit: int = 5) -> list[str]:
        """Get recent checkpoint reasons for drift signal computation."""
        rows = self._conn.execute(
            """SELECT checkpoint_reason FROM checkpoints
               WHERE task_id = ? AND user_id = ?
               ORDER BY version DESC LIMIT ?""",
            [task_id, user_id, limit],
        ).fetchall()
        return [r["checkpoint_reason"] for r in rows]

    def get_latest_checkpoint_ts(self, task_id: int, user_id: str = "default") -> str | None:
        """Get created_at of latest checkpoint for debounce logic."""
        row = self._conn.execute(
            """SELECT created_at FROM checkpoints
               WHERE task_id = ? AND user_id = ?
               ORDER BY version DESC LIMIT 1""",
            [task_id, user_id],
        ).fetchone()
        return row["created_at"] if row else None

    def get_latest_checkpoint_for_reason(self, task_id: int, reason: str, user_id: str = "default") -> str | None:
        """Get created_at of latest checkpoint with given reason (for debounce)."""
        row = self._conn.execute(
            """SELECT created_at FROM checkpoints
               WHERE task_id = ? AND user_id = ? AND checkpoint_reason = ?
               ORDER BY version DESC LIMIT 1""",
            [task_id, user_id, reason],
        ).fetchone()
        return row["created_at"] if row else None

    # ---- Session Lifecycle ----

    def start_session(self, session_id: str, user_id: str = "default") -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR IGNORE INTO session_lifecycle (session_id, user_id)
                   VALUES (?, ?)""",
                [session_id, user_id],
            )
            self._conn.commit()

    def end_session(self, session_id: str, end_type: str = "normal", **kwargs) -> None:
        interruption_reason = kwargs.get("interruption_reason")
        interruption_context = json.dumps(kwargs.get("interruption_context") or {})
        with self._lock:
            self._conn.execute(
                """UPDATE session_lifecycle
                   SET end_type = ?, interruption_reason = ?, interruption_context = ?,
                       last_active_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                   WHERE session_id = ?""",
                [end_type, interruption_reason, interruption_context, session_id],
            )
            self._conn.commit()

    def get_recent_sessions(self, user_id: str = "default", limit: int = 5) -> list[dict]:
        rows = self._conn.execute(
            """SELECT * FROM session_lifecycle
               WHERE user_id = ?
               ORDER BY started_at DESC LIMIT ?""",
            [user_id, limit],
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- Helpers ----

    def _row_to_task(self, row) -> TaskRow:
        meta = row["metadata"]
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}
        return TaskRow(
            id=row["id"],
            name=row["name"],
            goal=row["goal"],
            status=row["status"],
            user_id=row["user_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=meta,
            execution_id=row["execution_id"],
            previous_task_id=row["previous_task_id"],
            checkpoint_id=row["checkpoint_id"],
            attempt=row["attempt"] or 1,
            parent_task_id=row["parent_task_id"],
            retry_of_task_id=row["retry_of_task_id"],
        )

    def _row_to_checkpoint(self, row) -> dict:
        """Convert a wide-table checkpoint row to the dict format expected by checkpoint.py.

        Must match the format returned by checkpoint._row_to_checkpoint:
        - nested "state" dict with goal/completed/in_progress/etc.
        - "reason" key (not "checkpoint_reason")
        """
        def _parse_json(val, default=None):
            if default is None:
                default = []
            if val is None:
                return default
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    return default
            return val

        return {
            "id": row["id"],
            "task_id": row["task_id"],
            "version": row["version"],
            "parent_version": row["parent_version"],
            "kind": row["kind"],
            "reason": row["checkpoint_reason"],
            "checkpoint_reason": row["checkpoint_reason"],
            "triggered_by_event": row["triggered_by_event"],
            "state": {
                "goal": row["goal"] or "",
                "completed": _parse_json(row["completed"], []),
                "in_progress": _parse_json(row["in_progress"], []),
                "blocked": _parse_json(row["blocked"], []),
                "preferred_next": _parse_json(row["preferred_next"], []),
                "must_not_redo": _parse_json(row["must_not_redo"], []),
                "must_preserve": _parse_json(row["must_preserve"], []),
                "working_set": _parse_json(row["working_set"], {}),
                "active_constraints": _parse_json(row["active_constraints"], []),
                "blocked_reasons": _parse_json(row["blocked_reasons"], []),
            },
            "state_diff": _parse_json(row["state_diff"], {}),
            "source_session_id": row["source_session_id"],
            "source_memory_id": row["source_memory_id"],
            "continuation_confidence": row["continuation_confidence"],
            "confidence_breakdown": _parse_json(row["confidence_breakdown"], {}),
            "failure_signature": row["failure_signature"],
            "created_at": row["created_at"],
        }
