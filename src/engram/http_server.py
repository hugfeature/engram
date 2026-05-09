"""Engram Unified Server — FastAPI REST + MCP StreamableHTTP."""

import argparse
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
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
        # DatabaseLockedError gets surfaced specifically: another engram is
        # already running and the DB file is healthy. Refuse to serve, do
        # NOT proceed in degraded mode (that would imply data loss when
        # there is none).
        from .db import DatabaseLockedError
        if isinstance(e, DatabaseLockedError):
            log.error(
                "DB is locked by another process — refusing to start. %s", e
            )
            raise SystemExit(2) from e
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
            # Flush WAL so next start has nothing to replay/move aside.
            shared._db.checkpoint()
        except Exception as e:
            log.warning("CHECKPOINT on shutdown failed (non-fatal): %s", e)
        try:
            shared._db.close()
        except Exception as e:
            log.error("DB close on shutdown failed: %s", e)
    log.info("Engram server shutting down")


# --- FastAPI App ---

app = FastAPI(title="Engram", version=__version__, lifespan=lifespan)


@app.exception_handler(Exception)
async def _degraded_mode_handler(request: Request, exc: Exception):
    """Translate runtime durability errors to clean HTTP responses.

    DegradedModeError -> HTTP 503 (clients should call `engram recover`).
    DatabaseCorruptionError -> HTTP 503 with backup path.
    Other unexpected exceptions fall through to FastAPI default handling.
    """
    from .db import DegradedModeError, DatabaseCorruptionError
    if isinstance(exc, DegradedModeError):
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "code": "degraded_mode",
                "error": str(exc),
                "recover_command": exc.recover_command,
                "db_path": exc.db_path,
            },
        )
    if isinstance(exc, DatabaseCorruptionError):
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "code": "database_corrupt",
                "error": str(exc),
                "recover_command": exc.recover_command,
                "db_path": exc.db_path,
                "backup_path": exc.backup_path,
            },
        )
    raise exc


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
    db_readonly = False
    embedding_stale = False
    residue: list[str] = []
    meta: dict[str, str] = {}
    try:
        if shared._db is not None:
            shared._db.conn.execute("SELECT 1").fetchone()
            db_ok = True
            fts_ok = shared._db.fts_available
            db_readonly = shared._db.readonly
            embedding_stale = shared._db.embedding_stale
            residue = shared._db.residue_files
            meta = shared._db.all_meta()
    except Exception:
        pass
    graph_ok = shared._graph is not None
    degraded = is_degraded()
    status = (
        "ok"
        if (db_ok and graph_ok and not degraded and not db_readonly)
        else "degraded"
    )
    return {
        "status": status,
        "version": __version__,
        "db": db_ok,
        "db_readonly": db_readonly,
        "graph": graph_ok,
        "fts": fts_ok,
        "embedding_degraded": degraded,
        "embedding_stale": embedding_stale,
        "residue_files": residue,
        "engram_meta": meta,
        "transports": ["rest", "mcp-streamable-http"],
    }


def _port_in_use(host: str, port: int) -> bool:
    """Return True if another process is already bound to ``host:port``.

    We probe with a short connection attempt rather than listing sockets so
    the check works on macOS / Linux without root or extra deps. False on
    any error (we'd rather try and fail loudly than refuse to start because
    the probe itself broke).
    """
    import socket
    probe_host = host if host not in ("0.0.0.0", "::") else "127.0.0.1"
    try:
        with socket.create_connection((probe_host, port), timeout=0.5):
            return True
    except (OSError, socket.timeout):
        return False


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Engram unified server (REST + MCP)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8900)
    args = parser.parse_args()

    # Pre-flight: refuse to start if the port is already taken. Without this
    # check, two engram processes race for the DuckDB lock and the loser
    # used to mis-classify the lock conflict as DB corruption (v0.10/v0.11
    # bug, fixed in v0.11.1). Catching it here is cheaper and clearer.
    if _port_in_use(args.host, args.port):
        log.error(
            "Port %s:%d is already in use — another engram server is "
            "probably running. Stop it first (`engram-server stop` or "
            "`launchctl unload ~/Library/LaunchAgents/com.engram.server.plist`).",
            args.host, args.port,
        )
        sys.exit(2)

    log.info(f"Engram server starting on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()