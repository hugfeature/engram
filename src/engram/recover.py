"""Recover Tier 1 runtime state from the append-only event log.

Engram's durability contract:
    Event log is the only durability primitive.
    If it cannot be replayed, it is not critical state.

This module replays ~/.engram/events/*.jsonl into a fresh DuckDB database,
reconstructing tasks / checkpoints / session lifecycle / session outcomes /
session memory recall / Tier 2 memories (best-effort).

Tier 3 (embeddings / FTS / vector index) is intentionally NOT recovered —
it is disposable cache and will be rebuilt by normal runtime as needed.

Default mode is **dry-run**: a new DB is built at
``~/.engram/recovered-<ts>/memories.duckdb`` and the user must explicitly
``--promote`` to swap it in (the original DB is moved into ``backups/``).
This protects users from a bad recover overwriting the last good state.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time as _time
from dataclasses import dataclass, field
from typing import Iterable

from .db import (
    MemoryDB,
    DB_PATH,
    ENGRAM_DIR,
    _BACKUP_DIR,
    _ts_suffix,
)
from .event_log import EventLog, DEFAULT_EVENT_DIR

log = logging.getLogger("engram.recover")


@dataclass
class RecoverReport:
    output_db: str
    event_dir: str
    since_date: str | None
    counts: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    promoted: bool = False
    backup_path: str | None = None
    # P1-1: when a snapshot was used as the base, record which one so users
    # can verify they're recovering from the expected point in time.
    snapshot_used: bool = False
    snapshot_seq: int = 0

    def as_dict(self) -> dict:
        return {
            "output_db": self.output_db,
            "event_dir": self.event_dir,
            "since_date": self.since_date,
            "counts": dict(self.counts),
            "skipped": dict(self.skipped),
            "errors": list(self.errors),
            "promoted": self.promoted,
            "backup_path": self.backup_path,
            "snapshot_used": self.snapshot_used,
            "snapshot_seq": self.snapshot_seq,
        }


# Subset of event kinds we know how to replay. Unknown kinds are counted
# under ``skipped`` so the user can see them but don't break recovery.
_REPLAYABLE = {
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
    "memory.store",
    "memory.update",
    "memory.delete",
}


def recover(
    *,
    event_dir: str = DEFAULT_EVENT_DIR,
    output_dir: str | None = None,
    since_date: str | None = None,
    promote: bool = False,
    target_db: str = DB_PATH,
    snapshot_dir: str | None = None,
) -> RecoverReport:
    """Replay events into a fresh DuckDB; optionally promote it to active.

    Args:
        event_dir:   where to read ``events-*.jsonl`` from.
        output_dir:  where to materialize the recovered DB. If None, a
                     timestamped dir under ~/.engram is created.
        since_date:  ``YYYYMMDD`` lower bound (inclusive); None = full replay.
        promote:     if True, move ``target_db`` to backups/ and swap in the
                     recovered DB. If False (default), only build it.
        target_db:   the production DB path that will be replaced on promote.
        snapshot_dir: directory to look for snapshots in. None = use the
                     default (``~/.engram/snapshots/``) ONLY when
                     ``event_dir`` is also the default — otherwise we
                     refuse to mix a custom event log with global
                     snapshots (would yield wrong replay state). Pass
                     an explicit path to force the fast-path; pass an
                     empty string to disable it.
    """
    if output_dir is None:
        output_dir = os.path.join(ENGRAM_DIR, f"recovered-{_ts_suffix()}")
    os.makedirs(output_dir, exist_ok=True)
    output_db = os.path.join(output_dir, "memories.duckdb")

    if os.path.exists(output_db):
        raise FileExistsError(
            f"Refusing to overwrite existing file: {output_db}. "
            "Pick a different --output."
        )

    report = RecoverReport(
        output_db=output_db,
        event_dir=event_dir,
        since_date=since_date,
    )

    # P1-1: snapshot-aware base. If a snapshot exists AND the user is not
    # asking for a partial replay (since_date), we copy that snapshot in as
    # the base DB and only replay events with seq > snapshot.seq.
    # When since_date is set the user is explicitly asking for "events from
    # date X onward" — we honor that and skip the snapshot fast-path so the
    # output reflects exactly that window.
    #
    # Safety: only consult snapshots when event_dir + snapshot_dir are
    # consistent. A custom event_dir paired with the default global
    # snapshot_dir would seed the recovered DB from production state that
    # has nothing to do with the event log being replayed (this happens in
    # tests and in any caller that points at a non-default event_dir).
    effective_snapshot_dir = _resolve_snapshot_dir(event_dir, snapshot_dir)
    base_seq = _try_seed_from_snapshot(
        output_db, since_date, snapshot_dir=effective_snapshot_dir,
    )
    if base_seq > 0:
        report.snapshot_seq = base_seq
        report.snapshot_used = True

    # Build the DB. If snapshot was seeded, MemoryDB just opens the existing
    # file and re-runs idempotent _init_schema (CREATE ... IF NOT EXISTS).
    db = MemoryDB(db_path=output_db, log_writes=False)
    try:
        log_ = EventLog(event_dir=event_dir)
        events = log_.iter_events(since_date=since_date)
        _replay_events(db, events, report, min_seq=base_seq)
        db.checkpoint()
    finally:
        db.close()

    if promote:
        report.backup_path, report.promoted = _promote(output_db, target_db)

    return report


def _resolve_snapshot_dir(event_dir: str, override: str | None) -> str | None:
    """Pick a snapshot dir that is consistent with the event log being replayed.

    Rules:
      - override is "" (empty string)  -> snapshots disabled outright.
      - override is a non-empty path   -> use it verbatim (caller knows best).
      - override is None and event_dir is the default
                                       -> use the global snapshot dir.
      - override is None and event_dir is custom
                                       -> snapshots disabled (returns None)
                                          to avoid mixing data sources.
    """
    if override == "":
        return None
    if override is not None:
        return override
    if os.path.realpath(event_dir) == os.path.realpath(DEFAULT_EVENT_DIR):
        from .snapshot import SNAPSHOT_DIR
        return SNAPSHOT_DIR
    return None


def _try_seed_from_snapshot(
    output_db: str,
    since_date: str | None,
    *,
    snapshot_dir: str | None,
) -> int:
    """If a snapshot is available and the request is a full replay, copy it
    into ``output_db`` and return its seq. Returns 0 to signal "no seed used"."""
    if since_date:
        return 0
    if snapshot_dir is None:
        return 0
    try:
        from .snapshot import latest_snapshot
        snap = latest_snapshot(snapshot_dir)
    except Exception as exc:
        log.debug("snapshot lookup failed: %s", exc)
        return 0
    if snap is None:
        return 0
    try:
        shutil.copy2(snap.path, output_db)
        log.info("recover seeded from snapshot seq=%d path=%s", snap.seq, snap.path)
        return snap.seq
    except OSError as exc:
        log.warning("snapshot seed failed (%s); falling back to full replay", exc)
        # Make sure we don't leave a half-copied file behind.
        try:
            os.remove(output_db)
        except OSError:
            pass
        return 0


def _replay_events(
    db: MemoryDB,
    events: Iterable[dict],
    report: RecoverReport,
    *,
    min_seq: int = 0,
) -> None:
    for event in events:
        seq = event.get("seq", 0)
        if seq <= min_seq:
            continue  # already represented in the snapshot base
        kind = event.get("kind")
        payload = event.get("payload") or {}
        if kind not in _REPLAYABLE:
            report.skipped[kind or "<missing>"] = report.skipped.get(kind or "<missing>", 0) + 1
            continue
        try:
            handler = _REPLAY_HANDLERS[kind]
            handler(db, payload)
        except Exception as exc:
            report.errors.append(f"seq={event.get('seq')} kind={kind}: {exc}")
            log.error("Replay failed for seq=%s kind=%s: %s", event.get("seq"), kind, exc)
            continue
        report.counts[kind] = report.counts.get(kind, 0) + 1


# ---- per-kind replay handlers ----

def _replay_task_create(db: MemoryDB, p: dict) -> None:
    task_id = p["task_id"]
    db.conn.execute(
        """INSERT INTO tasks (id, name, goal, status, user_id, metadata)
           VALUES (?, ?, ?, ?, ?, ?::JSON)
           ON CONFLICT (id) DO UPDATE SET
               name = excluded.name, goal = excluded.goal,
               status = excluded.status, user_id = excluded.user_id,
               metadata = excluded.metadata""",
        [
            task_id,
            p.get("name", ""),
            p.get("goal", "") or "",
            p.get("status", "planning"),
            p.get("user_id", "default"),
            json.dumps(p.get("metadata") or {}, ensure_ascii=False),
        ],
    )
    # Keep the sequence ahead of the highest replayed id so future inserts
    # don't collide.
    _bump_sequence(db, "memory_id_seq", 0)  # no-op for memories
    _bump_sequence_to(db, "task_id_seq", task_id)


def _replay_task_update(db: MemoryDB, p: dict) -> None:
    task_id = p["task_id"]
    sets = ["updated_at = now()"]
    params: list = []
    if p.get("status") is not None:
        sets.append("status = ?")
        params.append(p["status"])
    if p.get("goal") is not None:
        sets.append("goal = ?")
        params.append(p["goal"])
    if p.get("metadata") is not None:
        sets.append("metadata = ?::JSON")
        params.append(json.dumps(p["metadata"], ensure_ascii=False))
    params.append(task_id)
    db.conn.execute(
        f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?",
        params,
    )


def _replay_checkpoint_write(db: MemoryDB, p: dict) -> None:
    state = p.get("state") or {}
    db.conn.execute(
        """INSERT INTO checkpoints (
            id, task_id, version, parent_version, kind,
            checkpoint_reason, triggered_by_event,
            goal, completed, in_progress, blocked, preferred_next,
            must_not_redo, must_preserve, working_set,
            state_diff, source_session_id, source_memory_id,
            continuation_confidence, confidence_breakdown,
            failure_signature, user_id
        ) VALUES (
            ?, ?, ?, ?, ?,
            ?, ?,
            ?, ?::JSON, ?::JSON, ?::JSON, ?::JSON,
            ?::JSON, ?::JSON, ?::JSON,
            ?::JSON, ?, ?,
            ?, ?::JSON,
            ?, ?
        ) ON CONFLICT (id) DO NOTHING""",
        [
            p["checkpoint_id"], p["task_id"], p["version"], p.get("parent_version"),
            p.get("kind", "auto"),
            p.get("checkpoint_reason", "AUTO_SAVE"),
            p.get("triggered_by_event"),
            state.get("goal", "") or "",
            json.dumps(state.get("completed") or [], ensure_ascii=False),
            json.dumps(state.get("in_progress") or [], ensure_ascii=False),
            json.dumps(state.get("blocked") or [], ensure_ascii=False),
            json.dumps(state.get("preferred_next") or [], ensure_ascii=False),
            json.dumps(state.get("must_not_redo") or [], ensure_ascii=False),
            json.dumps(state.get("must_preserve") or [], ensure_ascii=False),
            json.dumps(state.get("working_set") or {}, ensure_ascii=False),
            json.dumps(p.get("state_diff") or {}, ensure_ascii=False),
            p.get("source_session_id"), p.get("source_memory_id"),
            p.get("continuation_confidence"),
            json.dumps(p.get("confidence_breakdown") or {}, ensure_ascii=False),
            p.get("failure_signature"),
            p.get("user_id", "default"),
        ],
    )
    # Refresh tasks summary cache columns.
    db.conn.execute(
        """UPDATE tasks
           SET latest_checkpoint_version = GREATEST(COALESCE(latest_checkpoint_version, 0), ?),
               checkpoint_count = checkpoint_count + 1
           WHERE id = ?""",
        [p["version"], p["task_id"]],
    )
    _bump_sequence_to(db, "checkpoint_id_seq", p["checkpoint_id"])


def _replay_session_start(db: MemoryDB, p: dict) -> None:
    db.conn.execute(
        """INSERT INTO session_lifecycle (session_id, user_id)
           VALUES (?, ?)
           ON CONFLICT (session_id) DO NOTHING""",
        [p["session_id"], p.get("user_id", "default")],
    )


def _replay_session_end(db: MemoryDB, p: dict) -> None:
    interruption_reason = p.get("interruption_reason")
    interruption_context = p.get("interruption_context")
    context_json = json.dumps(interruption_context or {}, ensure_ascii=False)
    db.conn.execute(
        """UPDATE session_lifecycle
           SET ended_at = COALESCE(ended_at, now()),
               last_active_at = now(),
               end_type = ?,
               interruption_reason = COALESCE(?, interruption_reason),
               interruption_context = CASE
                   WHEN ? != '{}' THEN ?::JSON
                   ELSE interruption_context
               END
           WHERE session_id = ?""",
        [p.get("end_type", "handoff"), interruption_reason,
         context_json, context_json, p["session_id"]],
    )


def _replay_session_outcome(db: MemoryDB, p: dict) -> None:
    db.conn.execute(
        "INSERT INTO session_outcome_log (session_id, user_id, outcome) VALUES (?, ?, ?)",
        [p["session_id"], p.get("user_id", "default"), p.get("outcome", "")],
    )


def _replay_session_memory_recall(db: MemoryDB, p: dict) -> None:
    rows = [
        [p["session_id"], mid, p.get("user_id", "default")]
        for mid in p.get("memory_ids") or []
    ]
    if rows:
        db.conn.executemany(
            "INSERT INTO session_memory_log (session_id, memory_id, user_id) VALUES (?, ?, ?)",
            rows,
        )


def _replay_memory_store(db: MemoryDB, p: dict) -> None:
    """Tier 2 replay: rebuild content + metadata. Embeddings are NOT replayed
    (Tier 3); they will be lazily regenerated by the runtime."""
    memory_id = p["memory_id"]
    # Insert with a NULL embedding placeholder. Schema requires FLOAT[dim]
    # but accepts NULL for missing values.
    db.conn.execute(
        """INSERT INTO memories (id, user_id, content, importance, category, metadata)
           VALUES (?, ?, ?, ?, ?, ?::JSON)
           ON CONFLICT (id) DO UPDATE SET
               content = excluded.content,
               importance = excluded.importance,
               category = excluded.category,
               metadata = excluded.metadata""",
        [
            memory_id,
            p.get("user_id", "default"),
            p.get("content", ""),
            p.get("importance", 0.5),
            p.get("category", "fact"),
            json.dumps(p.get("metadata") or {}, ensure_ascii=False),
        ],
    )
    _bump_sequence_to(db, "memory_id_seq", memory_id)


def _replay_memory_update(db: MemoryDB, p: dict) -> None:
    sets = ["last_accessed_at = now()"]
    params: list = []
    if p.get("content") is not None:
        sets.append("content = ?")
        params.append(p["content"])
    if p.get("importance") is not None:
        sets.append("importance = ?")
        params.append(p["importance"])
    if p.get("metadata") is not None:
        sets.append("metadata = ?::JSON")
        params.append(json.dumps(p["metadata"], ensure_ascii=False))
    params.append(p["memory_id"])
    db.conn.execute(
        f"UPDATE memories SET {', '.join(sets)} WHERE id = ?",
        params,
    )


def _replay_memory_delete(db: MemoryDB, p: dict) -> None:
    db.conn.execute("DELETE FROM memories WHERE id = ?", [p["memory_id"]])


# ---- Execution Lineage replay handlers (v0.16) ----

def _replay_execution_start(db: MemoryDB, p: dict) -> None:
    db.conn.execute(
        """INSERT INTO execution_sessions
               (execution_id, root_goal, origin_checkpoint, status, user_id)
           VALUES (?, ?, ?, 'active', ?)
           ON CONFLICT (execution_id) DO NOTHING""",
        [
            p["execution_id"],
            p.get("root_goal", ""),
            p.get("origin_checkpoint"),
            p.get("user_id", "default"),
        ],
    )


def _replay_execution_end(db: MemoryDB, p: dict) -> None:
    sets = ["status = ?", "last_active_at = now()"]
    params: list = [p.get("status", "completed")]
    if p.get("continuity_score") is not None:
        sets.append("continuity_score = ?")
        params.append(p["continuity_score"])
    params.append(p["execution_id"])
    db.conn.execute(
        f"UPDATE execution_sessions SET {', '.join(sets)} WHERE execution_id = ?",
        params,
    )


def _replay_task_retry(db: MemoryDB, p: dict) -> None:
    task_id = p["task_id"]
    execution_id = p.get("execution_id")
    retry_of = p.get("retry_of_task_id")
    attempt = p.get("attempt", 1)
    user_id = p.get("user_id", "default")
    # Update existing task row with lineage fields (task was already created via task.create)
    db.conn.execute(
        """UPDATE tasks
           SET execution_id = ?, retry_of_task_id = ?, previous_task_id = ?, attempt = ?
           WHERE id = ?""",
        [execution_id, retry_of, retry_of, attempt, task_id],
    )
    # Mark the retried task as cancelled
    if retry_of:
        db.conn.execute(
            "UPDATE tasks SET status = 'cancelled' WHERE id = ? AND status NOT IN ('done', 'cancelled')",
            [retry_of],
        )


def _replay_task_spawn(db: MemoryDB, p: dict) -> None:
    task_id = p["task_id"]
    execution_id = p.get("execution_id")
    parent_task_id = p.get("parent_task_id")
    checkpoint_id = p.get("checkpoint_id")
    # Update existing task row with lineage fields
    db.conn.execute(
        """UPDATE tasks
           SET execution_id = ?, parent_task_id = ?, checkpoint_id = ?
           WHERE id = ?""",
        [execution_id, parent_task_id, checkpoint_id, task_id],
    )


_REPLAY_HANDLERS = {
    "task.create": _replay_task_create,
    "task.update": _replay_task_update,
    "task.retry": _replay_task_retry,
    "task.spawn": _replay_task_spawn,
    "checkpoint.write": _replay_checkpoint_write,
    "execution.start": _replay_execution_start,
    "execution.end": _replay_execution_end,
    "session.start": _replay_session_start,
    "session.end": _replay_session_end,
    "session.outcome": _replay_session_outcome,
    "session.memory_recall": _replay_session_memory_recall,
    "memory.store": _replay_memory_store,
    "memory.update": _replay_memory_update,
    "memory.delete": _replay_memory_delete,
}


def _bump_sequence_to(db: MemoryDB, seq_name: str, target_value: int | None) -> None:
    if target_value is None or target_value <= 0:
        return
    try:
        db.conn.execute(f"SELECT setval('{seq_name}', ?, true)", [target_value])
    except Exception as exc:
        log.debug("setval(%s, %s) failed: %s", seq_name, target_value, exc)


def _bump_sequence(db: MemoryDB, seq_name: str, target_value: int) -> None:
    # Compatibility helper kept for callers that pre-compute target=0.
    if target_value > 0:
        _bump_sequence_to(db, seq_name, target_value)


def _promote(recovered_db: str, target_db: str) -> tuple[str | None, bool]:
    """Atomically swap the recovered DB into place; original goes to backups/."""
    os.makedirs(_BACKUP_DIR, exist_ok=True)
    backup_path: str | None = None
    if os.path.exists(target_db):
        backup_path = os.path.join(
            _BACKUP_DIR, f"memories-pre-recover-{_ts_suffix()}.duckdb"
        )
        shutil.move(target_db, backup_path)
        log.warning("Original DB moved to backup: %s", backup_path)
        # Move sidecar files too if present.
        for sidecar in (target_db + ".wal",):
            if os.path.exists(sidecar):
                shutil.move(sidecar, backup_path + os.path.splitext(sidecar)[1])
    shutil.move(recovered_db, target_db)
    log.warning("Recovered DB promoted: %s -> %s", recovered_db, target_db)
    return backup_path, True


# ---- doctor: report on residue / health without recovering ----

def doctor(db_path: str = DB_PATH, event_dir: str = DEFAULT_EVENT_DIR) -> dict:
    """Quick read-only health report. Safe to run in degraded mode."""
    from .db import _scan_residue
    from .maintenance import (
        BACKUP_DIR,
        ARCHIVE_DIR,
        _list_managed_backups,
        _read_retain_count,
    )
    live_backups = _list_managed_backups(BACKUP_DIR)
    archived_backups = _list_managed_backups(ARCHIVE_DIR)
    # P1-1 snapshot inventory.
    try:
        from .snapshot import list_snapshots, SNAPSHOT_DIR, latest_snapshot
        all_snaps = list_snapshots(SNAPSHOT_DIR)
        latest = latest_snapshot(SNAPSHOT_DIR)
        snapshot_info = {
            "dir": SNAPSHOT_DIR,
            "count": len(all_snaps),
            "latest_seq": latest.seq if latest else 0,
            "latest_path": latest.path if latest else None,
            "latest_size_bytes": latest.size_bytes if latest else 0,
        }
    except Exception as exc:
        snapshot_info = {"error": str(exc)}
    info: dict = {
        "db_path": db_path,
        "db_exists": os.path.exists(db_path),
        "event_dir": event_dir,
        "residue_files": _scan_residue(db_path),
        "backups": {
            "dir": BACKUP_DIR,
            "live_count": len(live_backups),
            "archive_count": len(archived_backups),
            "retain": _read_retain_count(),
            # Show only the newest 3 to keep doctor output skimmable.
            "live_recent": [os.path.basename(p) for p in live_backups[-3:]],
        },
        "snapshots": snapshot_info,
    }
    try:
        log_ = EventLog(event_dir=event_dir)
        kinds: dict[str, int] = {}
        max_seq = 0
        for event in log_.iter_events():
            k = event.get("kind") or "<missing>"
            kinds[k] = kinds.get(k, 0) + 1
            if event.get("seq", 0) > max_seq:
                max_seq = event["seq"]
        info["event_kinds"] = kinds
        info["event_max_seq"] = max_seq
    except Exception as exc:
        info["event_log_error"] = str(exc)

    if info["db_exists"]:
        try:
            db = MemoryDB(db_path=db_path, log_writes=False)
            info["meta"] = db.all_meta()
            info["counts"] = {
                "memories": db.count(),
                "tasks": db.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0],
                "checkpoints": db.conn.execute(
                    "SELECT COUNT(*) FROM checkpoints"
                ).fetchone()[0],
            }
            info["readonly"] = db.readonly
            info["embedding_stale"] = db.embedding_stale
            db.close()
        except Exception as exc:
            info["db_error"] = str(exc)
    return info
