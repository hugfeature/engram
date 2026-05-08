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
    """Mark the current session as ended when the process exits."""
    if _current_session_id is None:
        return
    try:
        db = get_db()
        db.conn.execute(
            """UPDATE session_lifecycle
               SET ended_at = now(), end_type = 'process_exit'
             WHERE session_id = ? AND ended_at IS NULL""",
            [_current_session_id],
        )
        log.info("Session closed on exit: %s", _current_session_id)
    except Exception as exc:
        log.debug("Session close on exit failed (non-fatal): %s", exc)


def _dispatch(name: str, arguments: dict) -> dict:
    """Core dispatch — look up handler, map args, execute. Returns raw dict."""
    db = get_db()
    graph = get_graph()

    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return {"ok": False, "error": f"Unknown tool: {name}"}

    # Auto-inject session_id for session-aware tools when caller omits it
    if name in _SESSION_AWARE_TOOLS and "session_id" not in arguments:
        arguments = {**arguments, "session_id": _ensure_session_id()}

    arg_map = ARG_MAPPING.get(name, {})
    kwargs = {}
    for mcp_key, handler_key in arg_map.items():
        if mcp_key in arguments:
            kwargs[handler_key] = arguments[mcp_key]

    return handler(db, graph, **kwargs)


def dispatch_tool(name: str, arguments: dict) -> list[TextContent]:
    """MCP dispatch — wraps result in TextContent."""
    result = _dispatch(name, arguments)
    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


def dispatch_rest(name: str, body: dict) -> dict:
    """REST dispatch — returns raw dict result."""
    return _dispatch(name, body)