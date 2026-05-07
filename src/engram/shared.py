"""Shared singletons and dispatch logic for MCP stdio and HTTP servers."""

from __future__ import annotations

import json
import logging

from mcp.types import TextContent

from .db import MemoryDB
from .graph import MemoryGraph
from .handlers import TOOL_HANDLERS, ARG_MAPPING

log = logging.getLogger("engram.shared")

_db: MemoryDB | None = None
_graph: MemoryGraph | None = None


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
}


def _dispatch(name: str, arguments: dict) -> dict:
    """Core dispatch — look up handler, map args, execute. Returns raw dict."""
    db = get_db()
    graph = get_graph()

    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return {"ok": False, "error": f"Unknown tool: {name}"}

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