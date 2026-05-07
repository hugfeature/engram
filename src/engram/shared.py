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


def dispatch_tool(name: str, arguments: dict) -> list[TextContent]:
    """Look up handler by tool name, map args, and execute."""
    db = get_db()
    graph = get_graph()

    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

    arg_map = ARG_MAPPING.get(name, {})
    kwargs = {}
    for mcp_key, handler_key in arg_map.items():
        if mcp_key in arguments:
            kwargs[handler_key] = arguments[mcp_key]

    result = handler(db, graph, **kwargs)
    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]