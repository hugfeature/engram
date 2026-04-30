"""Engram MCP Server — stdio mode, 6 tools."""

from __future__ import annotations

import json
import logging
import os
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent

from .db import MemoryDB
from .graph import MemoryGraph
from .pruner import start_scheduler
from .handlers import TOOL_HANDLERS, ARG_MAPPING
from .tools import TOOL_SCHEMAS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("engram")

server = Server("engram")

_db: MemoryDB | None = None
_graph: MemoryGraph | None = None
_scheduler = None


def _get_db() -> MemoryDB:
    global _db
    if _db is None:
        _db = MemoryDB()
    return _db


def _get_graph() -> MemoryGraph:
    global _graph
    if _graph is None:
        _graph = MemoryGraph()
    return _graph


@server.list_tools()
async def list_tools() -> list:
    return TOOL_SCHEMAS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    db = _get_db()
    graph = _get_graph()

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


def main():
    import asyncio

    global _scheduler
    _scheduler = start_scheduler(_get_db(), _get_graph())
    log.info("Engram MCP server starting (stdio mode)")

    async def _run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    try:
        asyncio.run(_run())
    finally:
        if _scheduler:
            _scheduler.shutdown(wait=False)
            log.info("Scheduler shut down")
        if _graph:
            _graph.flush()
            log.info("Graph flushed")


if __name__ == "__main__":
    main()
