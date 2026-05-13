"""Shared singletons and dispatch logic for MCP stdio and HTTP servers."""

from __future__ import annotations

import atexit
import json
import logging
import uuid

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


def _on_exit():
    """Mark the current session as ended when the process exits.

    Because atexit *did* fire, we know this is NOT a crash (SIGKILL/OOM).
    We classify the exit reason based on session signals:
    - If an LLM explicitly reported an interruption reason earlier
      (via report_interruption), we honour that.
    - Otherwise we mark it as process_exit with no interruption_reason
      (normal shutdown).
    """
    if _current_session_id is None:
        return
    try:
        db = get_db()
        # If the LLM already reported an interruption reason during this
        # session (via report_interruption tool), honour it instead of
        # overwriting with a generic process_exit.
        reason = _reported_interruption_reason
        context = _reported_interruption_context or {}
        if reason:
            context.setdefault("exit_source", "atexit_with_report")
        db.end_session(
            _current_session_id,
            end_type="process_exit",
            interruption_reason=reason,
            interruption_context=context if context else None,
        )
        log.info("Session closed on exit: %s (reason=%s)", _current_session_id, reason or "normal")
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
