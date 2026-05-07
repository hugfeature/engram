"""Engram MCP Server — stdio mode, 9 MCP tools."""

from __future__ import annotations

import asyncio
import logging
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent

from .pruner import start_scheduler
from .tools import TOOL_SCHEMAS
from .shared import get_db, get_graph, dispatch_tool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("engram")

server = Server("engram")

_scheduler = None


@server.list_tools()
async def list_tools() -> list:
    return TOOL_SCHEMAS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    return dispatch_tool(name, arguments)


def main():
    global _scheduler
    _scheduler = start_scheduler(get_db(), get_graph())
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
        from .shared import _db, _graph
        if _graph:
            _graph.flush()
            log.info("Graph flushed")
        if _db:
            _db.close()
            log.info("DB closed")


if __name__ == "__main__":
    main()