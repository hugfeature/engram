"""Shared singletons and dispatch logic for MCP stdio and HTTP servers."""

from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import uuid
from datetime import datetime, timezone

from mcp.types import TextContent

from .db import MemoryDB
from .graph import MemoryGraph
from .handlers import TOOL_HANDLERS, ARG_MAPPING

log = logging.getLogger("engram.shared")

_db: MemoryDB | None = None
_graph: MemoryGraph | None = None
_current_session_id: str | None = None

# Tools that participate in session lifecycle tracking
_SESSION_AWARE_TOOLS = {"recall_memory", "store_memory", "track_failure", "track_progress"}

# Guard: ensure interrupt checkpoint is created at most once per process lifetime.
# trigger_interrupt_checkpoint() is idempotent — safe to call from both
# server.py finally blocks (before DB close) and _on_exit (atexit fallback).
_interrupt_checkpoint_done = False


def get_db() -> MemoryDB:
    global _db
    if _db is None:
        _db = MemoryDB()
    return _db


def get_graph() -> MemoryGraph:
    global _graph
    if _graph is None:
        _graph = MemoryGraph()
    return _graph


TOOL_REST_MAP: dict[str, str] = {
    "recall_memory": "/v1/recall",
    "store_memory": "/v1/store",
    "update_memory": "/v1/update",
    "session_handoff": "/v1/handoff",
    "consolidate_memory": "/v1/consolidate",
    "memory_stats": "/v1/stats",
    "track_failure": "/v1/failure",
    "track_progress": "/v1/progress",
    "session_outcome": "/v1/session-outcome",
    "create_task": "/v1/tasks",
    "update_task": "/v1/tasks/update",
    "get_task": "/v1/tasks/get",
    "list_tasks": "/v1/tasks/list",
    "restore_checkpoint": "/v1/checkpoints/restore",
    "list_checkpoints": "/v1/checkpoints/list",
    "get_runtime_health": "/v1/runtime-health",
    "report_interruption": "/v1/report-interruption",
    "evaluate_continuity": "/v1/continuity",
}


def _ensure_session_id() -> str:
    """Lazily create a process-level session ID and register the atexit hook."""
    global _current_session_id
    if _current_session_id is None:
        _current_session_id = f"auto-{uuid.uuid4().hex[:12]}"
        atexit.register(_on_exit)
        log.info("Session auto-created: %s", _current_session_id)
    return _current_session_id


# ---------------------------------------------------------------------------
# Interrupt Checkpoint — deterministic, no LLM, called on process exit
# ---------------------------------------------------------------------------

def _get_git_modified_files() -> list[str]:
    """Return unstaged + staged modified file paths via git diff.

    Purely code-driven — no LLM involvement.
    Returns an empty list if git is unavailable or repo not found.
    """
    import shutil
    
    # Security: Use absolute path to prevent PATH hijacking
    git_path = shutil.which("git")
    if not git_path:
        return []
    
    files: list[str] = []
    for cmd in (
        [git_path, "diff", "--name-only", "HEAD"],
        [git_path, "diff", "--cached", "--name-only"],
    ):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3,
                shell=False,  # Explicitly disable shell
            )
            files.extend(line for line in result.stdout.strip().splitlines() if line)
        except Exception:
            pass
    # Deduplicate, preserve order
    return list(dict.fromkeys(files))


def _get_recent_events(limit: int = 50) -> list[dict]:
    """Read the last *limit* events from today's event log.

    Only reads today's file (since_date=today) to keep shutdown fast.
    Falls back to an empty list on any error — never raises.
    """
    try:
        from .event_log import get_event_log
        el = get_event_log()
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        events: list[dict] = []
        for event in el.iter_events(since_date=today):
            events.append(event)
            # Rolling buffer — avoid unbounded memory on large logs
            if len(events) > limit * 3:
                events = events[-limit:]
        return events[-limit:]
    except Exception:
        return []


def _summarise_recent_events(events: list[dict]) -> dict:
    """Extract deterministic runtime signals from the event stream.

    All fields are code-derived — no LLM. Walks events in reverse
    chronological order and stops as soon as each field is found.
    """
    last_tool_called: str = ""
    last_success_action: str = ""
    last_failure: str = ""

    for ev in reversed(events):
        kind = ev.get("kind", "")
        payload = ev.get("payload", {})

        # last_tool_called: most recent checkpoint's triggered_by_event
        if not last_tool_called and kind == "checkpoint.write":
            triggered = payload.get("triggered_by_event", "")
            if triggered:
                last_tool_called = triggered

        # last_success_action: most recent completed item in any checkpoint
        if not last_success_action and kind == "checkpoint.write":
            completed = payload.get("state", {}).get("completed", [])
            if isinstance(completed, list) and completed:
                last_success_action = completed[-1]

        # last_failure: most recent task.update that moved status to failed/blocked
        if not last_failure and kind == "task.update":
            status = payload.get("status", "")
            if status in ("failed", "blocked"):
                task_id = payload.get("task_id", "?")
                last_failure = f"task {task_id} → {status}"

        if last_tool_called and last_success_action and last_failure:
            break  # All fields found, stop early

    return {
        "last_tool_called": last_tool_called,
        "last_success_action": last_success_action,
        "last_failure": last_failure,
    }


def _infer_goal_from_events(events: list[dict]) -> str:
    """Infer what this session was working on from the event stream.

    Used as a fallback when no explicit task exists but we still want
    to save an interrupt checkpoint.  Purely deterministic — no LLM.
    """
    # 1. Most recent task.create event carries the best goal description.
    for ev in reversed(events):
        if ev.get("kind") == "task.create":
            payload = ev.get("payload", {})
            return payload.get("goal", "") or payload.get("name", "")
    # 2. Fall back to the most recent memory.store content (first 80 chars).
    for ev in reversed(events):
        if ev.get("kind") == "memory.store":
            content = ev.get("payload", {}).get("content", "")
            if content:
                return content[:80]
    return "Session interrupted without explicit task"


def trigger_interrupt_checkpoint() -> bool:
    """Build and persist a structured interrupt checkpoint.

    Designed to be called BEFORE the DB is closed — e.g. from the
    finally blocks in server.py and http_server.py.  Also called as a
    best-effort fallback from _on_exit() (atexit), where the DB may
    already be closed; in that case it returns False silently.

    Idempotent: the second call is always a no-op (returns False).

    What gets saved (all deterministic, no LLM):
    - goal / continuity state inherited from the latest existing checkpoint
    - modified_files from ``git diff HEAD`` + ``git diff --cached``
    - last_tool_called / last_success_action / last_failure from event log
    - resume_hint left empty (placeholder for future MCP Sampling)

    Returns:
        True  — checkpoint written successfully.
        False — skipped (already done, no active task, DB unavailable, error).
    """
    global _interrupt_checkpoint_done
    if _interrupt_checkpoint_done:
        return False
    _interrupt_checkpoint_done = True

    session_id = _current_session_id  # can be None; source_session_id is Optional

    try:
        db = get_db()

        # Bail out if DB connection is already closed by a prior finally block
        if db.conn is None or db.readonly:
            log.debug("Interrupt checkpoint skipped: DB unavailable (conn=%s readonly=%s)",
                      db.conn, db.readonly)
            return False

        # Serialize against request handlers and other schedulers — this runs
        # on the APScheduler thread (every 2min) and from atexit. Concurrent
        # conn access with a request would SIGBUS the daemon.
        with db.global_lock:
            # Find the most recently updated in-progress task.
            # list_tasks returns ORDER BY updated_at DESC, so active_tasks[0]
            # is the task the agent was most recently working on.
            active_tasks = db.list_tasks(status="in_progress")
            if not active_tasks:
                active_tasks = db.list_tasks(status="planning")

            if not active_tasks:
                # Fallback: infer a task from recent events so we don't lose
                # the interrupt checkpoint when no explicit task was created.
                recent_events = _get_recent_events(limit=30)
                if not recent_events:
                    log.debug("Interrupt checkpoint skipped: no tasks and no events")
                    return False

                inferred_goal = _infer_goal_from_events(recent_events)
                task_id = db.create_task(
                    name=f"auto-session-{(_current_session_id or 'unknown')[:8]}",
                    goal=inferred_goal,
                    status="in_progress",
                )
                task = db.get_task(task_id)
                log.info(
                    "No active task found; auto-created session task %d for interrupt checkpoint",
                    task_id,
                )
            else:
                task = active_tasks[0]

            from .checkpoint import (
                create_checkpoint,
                get_checkpoint,
                REASON_AUTO_SAVE,
                _as_dict,
                _as_list,
            )

            # Inherit existing checkpoint state so we don't regress continuity.
            # If there is no prior checkpoint, base_state is empty and we build
            # from task metadata alone.
            existing = get_checkpoint(db, task.id, user_id=task.user_id)
            base_state: dict = existing["state"] if existing else {}
            existing_working_set: dict = _as_dict(base_state.get("working_set"))

            # Deterministic runtime signals — code only, no LLM
            modified_files = _get_git_modified_files()
            recent_events = _get_recent_events(limit=50)
            event_summary = _summarise_recent_events(recent_events)

            # Merge git-diff files with any files already tracked in working_set.
            # Use dict.fromkeys to deduplicate while preserving order.
            merged_files = list(dict.fromkeys(
                existing_working_set.get("files", []) + modified_files
            ))

            state = {
                # Continuity fields: inherit from last checkpoint, fall back to task
                "goal": task.goal or base_state.get("goal", ""),
                "completed":      _as_list(base_state.get("completed")),
                "in_progress":    _as_list(base_state.get("in_progress")),
                "blocked":        _as_list(base_state.get("blocked")),
                "preferred_next": _as_list(base_state.get("preferred_next")),
                "must_not_redo":  _as_list(base_state.get("must_not_redo")),
                "must_preserve":  _as_list(base_state.get("must_preserve")),
                # working_set: preserve existing keys, overlay interrupt signals
                "working_set": {
                    **existing_working_set,
                    "files": merged_files,
                    # _interrupt namespace keeps interrupt metadata separate from
                    # the files/tools/artifacts fields used by Jaccard comparison
                    "_interrupt": {
                        "last_tool_called":   event_summary["last_tool_called"],
                        "last_success_action": event_summary["last_success_action"],
                        "last_failure":        event_summary["last_failure"],
                        # resume_hint left empty — placeholder for MCP Sampling
                        # (Phase 2: LLM fills this via sampling/createMessage)
                        "resume_hint":    "",
                        "interrupt_reason": _reported_interruption_reason or "process_exit",
                    },
                },
            }

            create_checkpoint(
                db=db,
                task_id=task.id,
                reason=REASON_AUTO_SAVE,
                state=state,
                source_session_id=session_id,
                user_id=task.user_id,
                triggered_by_event="process_exit",
            )

        log.info(
            "Interrupt checkpoint saved: task_id=%d session=%s "
            "modified_files=%d last_tool=%r",
            task.id,
            _current_session_id,
            len(modified_files),
            event_summary["last_tool_called"],
        )
        return True

    except Exception as exc:
        log.warning("Interrupt checkpoint failed (non-fatal): %s", exc)
        return False


# ---------------------------------------------------------------------------
# Session exit handler
# ---------------------------------------------------------------------------

def _on_exit():
    """Mark the current session as ended when the process exits.

    Because atexit *did* fire, we know this is NOT a crash (SIGKILL/OOM).
    We classify the exit reason based on session signals:
    - If an LLM explicitly reported an interruption reason earlier
      (via report_interruption), we honour that.
    - Otherwise we mark it as process_exit with no interruption_reason
      (normal shutdown).

    Interrupt checkpoint is attempted first.  When the server finally blocks
    (server.py / http_server.py) have already called trigger_interrupt_checkpoint()
    before closing the DB, the call here is a fast no-op (idempotency guard).
    When those blocks did NOT call it (e.g. stdio server), this is the last
    chance to save state — so we try anyway, accepting that the DB might
    already be closed.
    """
    if _current_session_id is None:
        return

    # Best-effort interrupt checkpoint (no-op if already done or DB closed)
    trigger_interrupt_checkpoint()

    try:
        db = get_db()
        # If the LLM already reported an interruption reason during this
        # session (via report_interruption tool), honour it instead of
        # overwriting with a generic process_exit.
        reason = _reported_interruption_reason
        context = _reported_interruption_context or {}
        if reason:
            context.setdefault("exit_source", "atexit_with_report")
        with db.global_lock:
            db.end_session(
                _current_session_id,
                end_type="process_exit",
                interruption_reason=reason,
                interruption_context=context if context else None,
            )
        log.info(
            "Session closed on exit: %s (reason=%s)",
            _current_session_id,
            reason or "normal",
        )
    except Exception as exc:
        log.debug("Session close on exit failed (non-fatal): %s", exc)


# --- LLM-reported interruption state (set via report_interruption tool) ---
_reported_interruption_reason: str | None = None
_reported_interruption_context: dict | None = None


def set_interruption_report(reason: str, context: dict | None = None):
    """Called by report_interruption handler to record LLM-reported reason.

    This is consumed by _on_exit() so the final session.end event
    carries the correct interruption taxonomy.
    """
    global _reported_interruption_reason, _reported_interruption_context
    _reported_interruption_reason = reason
    _reported_interruption_context = context


def _dispatch(name: str, arguments: dict) -> dict:
    """Core dispatch — look up handler, map args, execute. Returns raw dict.

    Translates ``DegradedModeError`` into a structured response so MCP
    clients can branch on ``code='degraded_mode'`` and surface the
    ``recover_command`` to the user.
    """
    from .db import DegradedModeError
    from .handlers import _degraded_error

    db = get_db()
    graph = get_graph()

    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return {"ok": False, "error": f"Unknown tool: {name}"}

    # Auto-inject session_id for session-aware tools when caller omits it
    if name in _SESSION_AWARE_TOOLS and "session_id" not in arguments:
        try:
            arguments = {**arguments, "session_id": _ensure_session_id()}
        except DegradedModeError as exc:
            return _degraded_error(exc)

    arg_map = ARG_MAPPING.get(name, {})
    kwargs = {}
    for mcp_key, handler_key in arg_map.items():
        if mcp_key in arguments:
            kwargs[handler_key] = arguments[mcp_key]

    try:
        # Serialize all conn access against background jobs (pruner, snapshot,
        # checkpoint). DuckDBPyConnection is not thread-safe; without this the
        # daemon SIGBUSes when a request overlaps a scheduler tick.
        with db.global_lock:
            return handler(db, graph, **kwargs)
    except DegradedModeError as exc:
        return _degraded_error(exc)


def dispatch_tool(name: str, arguments: dict) -> list[TextContent]:
    """MCP dispatch — wraps result in TextContent."""
    result = _dispatch(name, arguments)
    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


def dispatch_rest(name: str, body: dict) -> dict:
    """REST dispatch — returns raw dict result."""
    return _dispatch(name, body)
