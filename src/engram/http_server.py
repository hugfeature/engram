"""Engram Unified Server — FastAPI REST + MCP StreamableHTTP."""

import argparse
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, create_model
from mcp.server import Server as MCPServer
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import TextContent

from .pruner import start_scheduler
from .embedding import is_degraded
from .tools import TOOL_SCHEMAS
from . import shared
from .shared import (
    get_db, get_graph, dispatch_tool, dispatch_rest,
    TOOL_REST_MAP,
)
from . import __version__

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("engram.http")


# --- MCP Server ---

mcp_server = MCPServer("engram")


@mcp_server.list_tools()
async def list_tools():
    return TOOL_SCHEMAS


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    return dispatch_tool(name, arguments)


session_mgr = StreamableHTTPSessionManager(app=mcp_server, stateless=True)


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = None
    try:
        get_db()
    except Exception as e:
        log.error("DB init failed (degraded mode): %s", e)
    try:
        get_graph()
    except Exception as e:
        log.error("Graph init failed (degraded mode): %s", e)
    try:
        if shared._db is not None and shared._graph is not None:
            scheduler = start_scheduler(shared._db, shared._graph)
    except Exception as e:
        log.warning("Scheduler init failed: %s", e)
    log.info("Engram server started (REST + MCP)")
    async with session_mgr.run():
        yield
    if scheduler:
        try:
            scheduler.shutdown(wait=False)
        except Exception:
            pass
        log.info("Scheduler shut down")
    if shared._graph is not None:
        try:
            shared._graph.flush()
        except Exception as e:
            log.error("Graph flush on shutdown failed: %s", e)
    if shared._db is not None:
        try:
            shared._db.close()
        except Exception as e:
            log.error("DB close on shutdown failed: %s", e)
    log.info("Engram server shutting down")


# --- FastAPI App ---

app = FastAPI(title="Engram", version=__version__, lifespan=lifespan)

# Mount MCP at /mcp
app.mount("/mcp", session_mgr.handle_request)


# --- Auto-generate REST routes from tool schemas ---

_JSON_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}

_ARRAY_ITEM_MAP = {
    "string": str,
    "integer": int,
    "number": float,
}


def _build_request_model(tool_name: str, schema: dict) -> type[BaseModel]:
    """Build a Pydantic model from a tool's inputSchema for REST validation + OpenAPI docs."""
    fields = {}
    props = schema.get("properties", {})
    required = set(schema.get("required", []))

    for name, prop in props.items():
        json_type = prop.get("type", "string")
        if json_type == "array":
            items_type = prop.get("items", {}).get("type", "string")
            item_py = _ARRAY_ITEM_MAP.get(items_type, str)
            py_type = list[item_py]
        elif json_type == "object":
            py_type = dict
        else:
            py_type = _JSON_TYPE_MAP.get(json_type, str)

        desc = prop.get("description", "")
        if name in required:
            fields[name] = (py_type, Field(description=desc))
        else:
            default = prop.get("default")
            if default is not None:
                fields[name] = (py_type, Field(default=default, description=desc))
            else:
                fields[name] = (py_type | None, Field(default=None, description=desc))

    model_name = "".join(w.capitalize() for w in tool_name.split("_")) + "Request"
    return create_model(model_name, **fields)


def _respond(result: dict) -> dict | JSONResponse:
    """Return 400 JSON if handler returned an error (ok=False), otherwise 200."""
    if result.get("ok") is False:
        return JSONResponse(content=result, status_code=400)
    return result


# Register REST routes from TOOL_REST_MAP + TOOL_SCHEMAS
for _tool in TOOL_SCHEMAS:
    _path = TOOL_REST_MAP.get(_tool.name)
    if _path is None:
        continue
    _model = _build_request_model(_tool.name, _tool.inputSchema)

    def _make_route(name: str, mdl: type[BaseModel]):
        def route(req: mdl):
            body = req.model_dump(exclude_unset=True)
            return _respond(dispatch_rest(name, body))
        route.__name__ = f"{name}_endpoint"
        return route

    app.post(_path, name=_tool.name)(_make_route(_tool.name, _model))


# --- Auto-generated tools catalog ---

@app.get("/v1/tools")
def tools_list():
    tools = []
    for tool in TOOL_SCHEMAS:
        path = TOOL_REST_MAP.get(tool.name)
        if path is None:
            continue
        params = {}
        for pname, pdef in tool.inputSchema.get("properties", {}).items():
            ptype = pdef.get("type", "string")
            if pname in tool.inputSchema.get("required", []):
                params[pname] = f"{ptype} (required)"
            else:
                params[pname] = ptype
        tools.append({"name": tool.name, "method": "POST", "path": path, "params": params})
    return {"tools": tools}


# --- Health ---

@app.get("/v1/health")
@app.get("/health")
def health():
    fts_ok = False
    db_ok = False
    try:
        if shared._db is not None:
            shared._db.conn.execute("SELECT 1").fetchone()
            db_ok = True
            fts_ok = shared._db.fts_available
    except Exception:
        pass
    graph_ok = shared._graph is not None
    degraded = is_degraded()
    status = "ok" if (db_ok and graph_ok and not degraded) else "degraded"
    return {
        "status": status,
        "version": __version__,
        "db": db_ok,
        "graph": graph_ok,
        "fts": fts_ok,
        "embedding_degraded": degraded,
        "transports": ["rest", "mcp-streamable-http"],
    }


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Engram unified server (REST + MCP)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8900)
    args = parser.parse_args()

    log.info(f"Engram server starting on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()