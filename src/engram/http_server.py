"""Engram Unified Server — FastAPI REST + MCP StreamableHTTP."""

from __future__ import annotations

import argparse
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from mcp.server import Server as MCPServer
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import TextContent

from .pruner import start_scheduler
from .embedding import is_degraded
from .tools import TOOL_SCHEMAS
from .handlers import (
    handle_recall, handle_store, handle_update,
    handle_session_handoff, handle_consolidate, handle_stats,
    handle_track_failure, handle_track_progress,
    handle_session_outcome,
)
from . import shared
from .shared import get_db, get_graph, dispatch_tool
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


# --- Request models ---

class RecallRequest(BaseModel):
    query: str
    user_id: str = "default"
    top_k: int = 5
    session_id: str | None = None


class StoreRequest(BaseModel):
    content: str
    importance: float
    category: str = "fact"
    user_id: str = "default"
    metadata: dict | None = None


class UpdateRequest(BaseModel):
    memory_id: int
    new_content: str
    importance: float | None = None


class HandoffRequest(BaseModel):
    summary: str
    completed: list[str] | None = None
    in_progress: list[str] | None = None
    blocked: list[str] | None = None
    next_steps: list[str] | None = None
    user_id: str = "default"


class ConsolidateRequest(BaseModel):
    user_id: str = "default"


class StatsRequest(BaseModel):
    user_id: str = "default"


class TrackFailureRequest(BaseModel):
    error: str
    component: str
    root_cause: str | None = None
    severity: str = "major"
    fix: str | None = None
    related_test_ids: list[str] | None = None
    user_id: str = "default"


class TrackProgressRequest(BaseModel):
    feature: str
    status: str
    completion: float = 0
    blockers: list[str] | None = None
    quality_score: float | None = None
    notes: str | None = None
    user_id: str = "default"


class SessionOutcomeRequest(BaseModel):
    session_id: str
    outcome: str
    notes: str | None = None
    user_id: str = "default"


# --- REST Routes ---


def _respond(result: dict) -> dict | JSONResponse:
    """Map handler outputs to HTTP status codes with optional explicit override."""
    explicit_status = result.get("status_code")
    if isinstance(explicit_status, int) and 100 <= explicit_status <= 599:
        payload = {k: v for k, v in result.items() if k != "status_code"}
        return JSONResponse(content=payload, status_code=explicit_status)

    if "error" in result:
        err = str(result.get("error", "")).lower()
        if "not found" in err:
            code = 404
        else:
            code = 400
        return JSONResponse(content=result, status_code=code)
    return result


@app.post("/v1/recall")
def recall_endpoint(req: RecallRequest):
    return _respond(handle_recall(get_db(), get_graph(),
                         query=req.query, user_id=req.user_id,
                         top_k=req.top_k, session_id=req.session_id))


@app.post("/v1/store")
def store_endpoint(req: StoreRequest):
    return _respond(handle_store(get_db(), get_graph(),
                        content=req.content, importance=req.importance,
                        category=req.category, user_id=req.user_id,
                        metadata=req.metadata))


@app.post("/v1/update")
def update_endpoint(req: UpdateRequest):
    return _respond(handle_update(get_db(), get_graph(),
                         memory_id=req.memory_id, new_content=req.new_content,
                         importance=req.importance))


@app.post("/v1/handoff")
def handoff_endpoint(req: HandoffRequest):
    return _respond(handle_session_handoff(
        get_db(), get_graph(),
        summary=req.summary, completed=req.completed,
        in_progress=req.in_progress, blocked=req.blocked,
        next_steps=req.next_steps, user_id=req.user_id,
    ))


@app.post("/v1/consolidate")
def consolidate_endpoint(req: ConsolidateRequest):
    return _respond(handle_consolidate(get_db(), get_graph(), user_id=req.user_id))


@app.post("/v1/stats")
def stats_endpoint(req: StatsRequest):
    return _respond(handle_stats(get_db(), user_id=req.user_id))


@app.post("/v1/failure")
def failure_endpoint(req: TrackFailureRequest):
    return _respond(handle_track_failure(
        get_db(), get_graph(),
        error=req.error, component=req.component,
        root_cause=req.root_cause, severity=req.severity,
        fix=req.fix, related_test_ids=req.related_test_ids,
        user_id=req.user_id,
    ))


@app.post("/v1/progress")
def progress_endpoint(req: TrackProgressRequest):
    return _respond(handle_track_progress(
        get_db(), get_graph(),
        feature=req.feature, status=req.status,
        completion=req.completion, blockers=req.blockers,
        quality_score=req.quality_score, notes=req.notes,
        user_id=req.user_id,
    ))


@app.post("/v1/session-outcome")
def session_outcome_endpoint(req: SessionOutcomeRequest):
    return _respond(handle_session_outcome(
        get_db(), get_graph(),
        session_id=req.session_id, outcome=req.outcome,
        notes=req.notes, user_id=req.user_id,
    ))


@app.get("/v1/health")
@app.get("/health")
def health():
    fts_ok = False
    db_ok = False
    try:
        if _db is not None:
            _db.conn.execute("SELECT 1").fetchone()
            db_ok = True
            fts_ok = _db.fts_available
    except Exception:
        pass
    graph_ok = _graph is not None
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


@app.get("/v1/tools")
def tools_list():
    return {
        "tools": [
            {"name": "recall", "method": "POST", "path": "/v1/recall",
             "params": {"query": "str (required)", "user_id": "str", "top_k": "int"}},
            {"name": "store", "method": "POST", "path": "/v1/store",
             "params": {"content": "str (required)", "importance": "float (required)",
                        "category": "str", "user_id": "str", "metadata": "object"}},
            {"name": "update", "method": "POST", "path": "/v1/update",
             "params": {"memory_id": "int (required)", "new_content": "str (required)",
                        "importance": "float"}},
            {"name": "handoff", "method": "POST", "path": "/v1/handoff",
             "params": {"summary": "str (required)", "completed": "list[str]",
                        "in_progress": "list[str]", "blocked": "list[str]",
                        "next_steps": "list[str]", "user_id": "str"}},
            {"name": "consolidate", "method": "POST", "path": "/v1/consolidate",
             "params": {"user_id": "str"}},
            {"name": "stats", "method": "POST", "path": "/v1/stats",
             "params": {"user_id": "str"}},
            {"name": "failure", "method": "POST", "path": "/v1/failure",
             "params": {"error": "str (required)", "component": "str (required)",
                        "root_cause": "str", "severity": "str", "fix": "str",
                        "related_test_ids": "list[str]", "user_id": "str"}},
            {"name": "progress", "method": "POST", "path": "/v1/progress",
             "params": {"feature": "str (required)", "status": "str (required)",
                        "completion": "float", "blockers": "list[str]",
                        "quality_score": "float", "notes": "str", "user_id": "str"}},
            {"name": "session_outcome", "method": "POST", "path": "/v1/session-outcome",
             "params": {"session_id": "str (required)", "outcome": "str (required)",
                        "notes": "str", "user_id": "str"}},
        ]
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
