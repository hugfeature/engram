"""Pure business logic handlers — no transport dependency (MCP or HTTP)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from .db import MemoryDB
from .graph import MemoryGraph
from .embedding import embed
from .resolve import resolve, Action
from .retrieve import recall
from .consolidator import run_consolidate
from .decay import compute_strength
from .pruner import maintenance

log = logging.getLogger("engram.handlers")

MAX_CONTENT_LENGTH = 100_000  # 100KB


def _validate_user_id(user_id: str) -> str:
    if not user_id or not isinstance(user_id, str):
        return "default"
    user_id = user_id.strip()[:100]
    return user_id or "default"


def _safe_embed(content: str) -> list[float] | None:
    try:
        return embed(content)
    except Exception as e:
        log.error("Embedding generation failed: %s", e)
        return None


def handle_recall(db: MemoryDB, graph: MemoryGraph, query: str,
                  user_id: str = "default", top_k: int = 5,
                  session_id: str | None = None) -> dict:
    if not query or not query.strip():
        return {"error": "query must be non-empty"}
    user_id = _validate_user_id(user_id)
    top_k = min(max(int(top_k), 1), 100)

    results = recall(query, db, graph, user_id, top_k)

    if session_id and results:
        db.log_session_recall(session_id, [r.id for r in results], user_id)

    meta_batch = db.get_metadata_batch([r.id for r in results]) if results else {}

    memories_out = []
    for r in results:
        entry = {
            "id": r.id,
            "content": r.content,
            "category": r.category,
            "importance": r.importance,
            "strength": round(r.strength, 4),
            "similarity": round(r.similarity, 4),
            "score": round(r.score, 4),
        }
        meta = meta_batch.get(r.id, {})
        if meta:
            entry["metadata"] = meta
        memories_out.append(entry)

    return {"memoriesFound": len(results), "memories": memories_out}


def handle_store(db: MemoryDB, graph: MemoryGraph, content: str,
                 importance: float, category: str = "fact",
                 user_id: str = "default", metadata: dict | None = None) -> dict:
    if not content or not content.strip():
        return {"error": "content must be non-empty"}
    if len(content) > MAX_CONTENT_LENGTH:
        return {"error": f"content too large (max {MAX_CONTENT_LENGTH // 1000}KB)"}
    try:
        importance = min(max(float(importance), 0.0), 1.0)
    except (TypeError, ValueError):
        importance = 0.5
    if category not in ("fact", "assumption", "failure", "strategy"):
        category = "fact"
    user_id = _validate_user_id(user_id)

    if metadata is not None:
        import json as _json
        try:
            if len(_json.dumps(metadata, ensure_ascii=False)) > 10_000:
                return {"error": "metadata too large (max 10KB)"}
        except (TypeError, ValueError):
            return {"error": "metadata must be JSON-serializable"}

    new_embedding = _safe_embed(content)
    if new_embedding is None:
        return {"error": "Embedding generation failed", "error_code": "internal"}
    existing = db.search_similar_for_dedup(new_embedding, user_id, top_k=10, threshold=0.60)
    resolution = resolve(content, new_embedding, existing)

    result_id = None
    if resolution.action == Action.NEW:
        mid = db.insert(content, new_embedding, importance, category, user_id, metadata=metadata)
        graph.index_memory_incremental(mid, new_embedding, db, user_id, importance, category)
        msg = f"Stored new memory (id={mid})"
        result_id = mid
    elif resolution.action == Action.REINFORCE:
        db.bump_recall(resolution.existing_id)
        msg = f"Reinforced existing memory (id={resolution.existing_id})"
        result_id = resolution.existing_id
    elif resolution.action == Action.REPLACE:
        db.update(resolution.existing_id, content, new_embedding, importance)
        msg = f"Replaced contradicting memory (id={resolution.existing_id})"
        result_id = resolution.existing_id
    elif resolution.action == Action.MERGE:
        merged = resolution.merged_content or content
        merged_emb = _safe_embed(merged)
        if merged_emb is None:
            return {"error": "Embedding generation failed"}
        db.update(resolution.existing_id, merged, merged_emb, importance)
        msg = f"Merged into existing memory (id={resolution.existing_id})"
        result_id = resolution.existing_id
    else:
        msg = "No action taken"

    out = {"result": msg}
    if result_id is not None:
        out["memory_id"] = result_id
    return out


def handle_update(db: MemoryDB, graph: MemoryGraph,
                  memory_id: int, new_content: str,
                  importance: float | None = None) -> dict:
    if not new_content or not new_content.strip():
        return {"error": "new_content must be non-empty", "error_code": "invalid_argument"}
    if len(new_content) > MAX_CONTENT_LENGTH:
        return {"error": f"new_content too large (max {MAX_CONTENT_LENGTH // 1000}KB)", "error_code": "unprocessable"}
        return {"error": "new_content must be non-empty"}
    if len(new_content) > MAX_CONTENT_LENGTH:
        return {"error": f"new_content too large (max {MAX_CONTENT_LENGTH // 1000}KB)"}
    try:
        memory_id = int(memory_id)
    except (TypeError, ValueError):
        return {"error": "memory_id must be an integer", "error_code": "invalid_argument"}
    existing = db.get_by_id(memory_id)
    if not existing:
        return {"error": f"Memory {memory_id} not found", "error_code": "not_found"}

    new_embedding = _safe_embed(new_content)
    if new_embedding is None:
        return {"error": "Embedding generation failed", "error_code": "internal"}
    db.update(memory_id, new_content, new_embedding, importance)

    graph.index_memory_incremental(
        memory_id, new_embedding, db,
        existing.user_id, importance or existing.importance, existing.category,
    )
    return {"result": f"Updated memory (id={memory_id})"}


def handle_session_handoff(db: MemoryDB, graph: MemoryGraph, summary: str,
                           completed: list[str] | None = None,
                           in_progress: list[str] | None = None,
                           blocked: list[str] | None = None,
                           next_steps: list[str] | None = None,
                           user_id: str = "default") -> dict:
    if not summary or not summary.strip():
        return {"error": "summary must be non-empty"}
    user_id = _validate_user_id(user_id)

    completed = completed or []
    in_progress = in_progress or []
    blocked = blocked or []
    next_steps = next_steps or []

    parts = [f"Session Handoff: {summary}"]
    if completed:
        parts.append("Completed: " + "; ".join(completed))
    if in_progress:
        parts.append("In progress: " + "; ".join(in_progress))
    if blocked:
        parts.append("Blocked: " + "; ".join(blocked))
    if next_steps:
        parts.append("Next steps: " + "; ".join(next_steps))
    content = "\n".join(parts)

    now_iso = datetime.now(timezone.utc).isoformat()
    meta = {
        "type": "handoff",
        "summary": summary,
        "completed": completed,
        "in_progress": in_progress,
        "blocked": blocked,
        "next_steps": next_steps,
        "timestamp": now_iso,
    }

    handoff_embedding = _safe_embed(content)
    if handoff_embedding is None:
        return {"error": "Embedding generation failed"}
    mid = db.insert(content, handoff_embedding, 0.9, "strategy", user_id, metadata=meta)
    graph.index_memory_incremental(mid, handoff_embedding, db, user_id, 0.9, "strategy")

    return {"result": f"Session handoff recorded (id={mid})", "memory_id": mid}


def handle_consolidate(db: MemoryDB, graph: MemoryGraph,
                       user_id: str = "default") -> dict:
    try:
        results = run_consolidate(db, graph, user_id)
    except Exception as e:
        log.error("Consolidation failed: %s", e)
        return {"error": f"consolidation failed: {e}", "error_code": "internal", "details": [], "status_code": 500}
        return {"error": f"consolidation failed: {e}", "details": [], "status_code": 500}
    if not results:
        msg = "No similar memories found to consolidate"
    else:
        merged_count = sum(len(r["removed"]) for r in results)
        msg = f"Consolidated {len(results)} cluster(s), merged {merged_count} duplicate(s)"
    return {"result": msg, "details": results}


def handle_stats(db: MemoryDB, user_id: str = "default") -> dict:
    user_id = _validate_user_id(user_id)

    # SQL aggregation — no full row loading
    agg = db.get_stats_aggregate(user_id)
    total = agg["total"]
    categories = agg["categories"]

    # Strength calculation — only load lightweight columns
    strength_rows = db.get_strength_data(user_id)
    now = datetime.now(timezone.utc)
    strengths: list[float] = []
    for cat, imp, last_accessed, recall_count in strength_rows:
        days = (now - last_accessed.replace(tzinfo=timezone.utc)).total_seconds() / 86400
        strengths.append(compute_strength(cat, imp, days, recall_count))

    result = {
        "total": total,
        "categories": categories,
        "avg_strength": round(sum(strengths) / len(strengths), 4) if strengths else 0,
        "last_maintenance": maintenance.last_time.isoformat() if maintenance.last_time else None,
        "fts_available": db.fts_available,
    }

    # Engineering stats — only load metadata column
    all_meta = db.get_metadata_for_stats(user_id)
    failures = [m for m in all_meta if m.get("type") == "failure"]
    progress_items = [m for m in all_meta if m.get("type") == "progress"]

    eng_stats: dict = {}
    if failures:
        components: dict[str, int] = {}
        severity_dist: dict[str, int] = {}
        for f in failures:
            comp = f.get("component", "unknown")
            components[comp] = components.get(comp, 0) + 1
            sev = f.get("severity", "unknown")
            severity_dist[sev] = severity_dist.get(sev, 0) + 1
        eng_stats["failures"] = {
            "total": len(failures),
            "by_component": components,
            "by_severity": severity_dist,
        }

    if progress_items:
        latest: dict[str, dict] = {}
        for p in progress_items:
            feat = p.get("feature", "unknown")
            ts = p.get("timestamp", "")
            if feat not in latest or ts > latest[feat].get("timestamp", ""):
                latest[feat] = p
        active = {k: {"status": v["status"], "completion": v.get("completion", 0)}
                  for k, v in latest.items() if v.get("status") != "done"}
        eng_stats["features"] = {
            "total_tracked": len(latest),
            "active": active,
        }

    if eng_stats:
        result["engineering"] = eng_stats

    return result


def handle_track_failure(db: MemoryDB, graph: MemoryGraph, error: str,
                         component: str, root_cause: str | None = None,
                         severity: str = "major", fix: str | None = None,
                         related_test_ids: list[str] | None = None,
                         user_id: str = "default") -> dict:
    if not error or not error.strip():
        return {"error": "error must be non-empty"}
    if not component or not component.strip():
        return {"error": "component must be non-empty"}
    if severity not in ("critical", "major", "minor"):
        severity = "major"
    user_id = _validate_user_id(user_id)

    parts = [f"Failure in {component}: {error}"]
    if root_cause:
        parts.append(f"Root cause: {root_cause}")
    if fix:
        parts.append(f"Fix: {fix}")
    if related_test_ids:
        parts.append(f"Related tests: {', '.join(related_test_ids)}")
    content = "\n".join(parts)

    now_iso = datetime.now(timezone.utc).isoformat()
    meta = {
        "type": "failure",
        "error": error,
        "component": component,
        "severity": severity,
        "root_cause": root_cause,
        "fix": fix,
        "related_test_ids": related_test_ids or [],
        "timestamp": now_iso,
    }

    importance = {"critical": 0.9, "major": 0.7, "minor": 0.5}[severity]
    emb = _safe_embed(content)
    if emb is None:
        return {"error": "Embedding generation failed"}
    mid = db.insert(content, emb, importance, "failure", user_id, metadata=meta)
    graph.index_memory_incremental(mid, emb, db, user_id, importance, "failure")

    return {"result": f"Failure tracked (id={mid})", "memory_id": mid}


def handle_track_progress(db: MemoryDB, graph: MemoryGraph, feature: str,
                          status: str, completion: float = 0,
                          blockers: list[str] | None = None,
                          quality_score: float | None = None,
                          notes: str | None = None,
                          user_id: str = "default") -> dict:
    if not feature or not feature.strip():
        return {"error": "feature must be non-empty"}
    valid_statuses = ("planning", "in_progress", "blocked", "review", "done")
    if status not in valid_statuses:
        return {"error": f"status must be one of {valid_statuses}"}
    try:
        completion = min(max(float(completion), 0), 100)
    except (TypeError, ValueError):
        completion = 0
    user_id = _validate_user_id(user_id)

    parts = [f"Progress: {feature} [{status}] {completion:.0f}%"]
    if blockers:
        parts.append(f"Blockers: {'; '.join(blockers)}")
    if quality_score is not None:
        parts.append(f"Quality: {quality_score:.2f}")
    if notes:
        parts.append(f"Notes: {notes}")
    content = "\n".join(parts)

    now_iso = datetime.now(timezone.utc).isoformat()
    meta = {
        "type": "progress",
        "feature": feature,
        "status": status,
        "completion": completion,
        "blockers": blockers or [],
        "quality_score": quality_score,
        "notes": notes,
        "timestamp": now_iso,
    }

    importance_map = {
        "planning": 0.6, "in_progress": 0.8,
        "blocked": 0.9, "review": 0.7, "done": 0.5,
    }
    importance = importance_map[status]
    emb = _safe_embed(content)
    if emb is None:
        return {"error": "Embedding generation failed"}
    mid = db.insert(content, emb, importance, "strategy", user_id, metadata=meta)
    graph.index_memory_incremental(mid, emb, db, user_id, importance, "strategy")

    return {"result": f"Progress tracked (id={mid})", "memory_id": mid}


def handle_session_outcome(db: MemoryDB, graph: MemoryGraph, session_id: str,
                           outcome: str, notes: str | None = None,
                           user_id: str = "default") -> dict:
    if not session_id or not session_id.strip():
        return {"error": "session_id must be non-empty"}
    if outcome not in ("success", "failure"):
        return {"error": "outcome must be 'success' or 'failure'"}
    user_id = _validate_user_id(user_id)

    memory_ids = db.get_session_memories(session_id, user_id)
    if not memory_ids:
        return {
            "result": "No memories recalled in this session",
            "session_id": session_id,
            "outcome": outcome,
            "memories_adjusted": 0,
        }

    # Importance feedback
    if outcome == "success":
        adjusted = db.adjust_importance_batch(memory_ids, +0.05)
    else:
        adjusted = db.adjust_importance_batch(memory_ids, -0.02)

        # Store failure lesson as a new memory
        lesson = notes or f"Session {session_id} failed (no notes provided)"
        lesson_content = f"Session failure lesson ({session_id}): {lesson}"
        lesson_embedding = _safe_embed(lesson_content)
        if lesson_embedding is not None:
            meta = {
                "type": "session_failure",
                "session_id": session_id,
                "original_notes": notes,
            }
            mid = db.insert(lesson_content, lesson_embedding, 0.7, "failure",
                            user_id, metadata=meta)
            graph.index_memory_incremental(mid, lesson_embedding, db, user_id, 0.7, "failure")

    return {
        "result": f"Session outcome recorded ({outcome})",
        "session_id": session_id,
        "outcome": outcome,
        "memories_adjusted": adjusted,
    }


TOOL_HANDLERS = {
    "recall_memory": lambda db, graph, **kw: handle_recall(db, graph, **kw),
    "store_memory": lambda db, graph, **kw: handle_store(db, graph, **kw),
    "update_memory": lambda db, graph, **kw: handle_update(db, graph, **kw),
    "session_handoff": lambda db, graph, **kw: handle_session_handoff(db, graph, **kw),
    "consolidate_memory": lambda db, graph, **kw: handle_consolidate(db, graph, **kw),
    "memory_stats": lambda db, graph, **kw: handle_stats(db, **kw),
    "track_failure": lambda db, graph, **kw: handle_track_failure(db, graph, **kw),
    "track_progress": lambda db, graph, **kw: handle_track_progress(db, graph, **kw),
    "session_outcome": lambda db, graph, **kw: handle_session_outcome(db, graph, **kw),
}

ARG_MAPPING = {
    "recall_memory": {"query": "query", "user_id": "user_id", "top_k": "top_k", "session_id": "session_id"},
    "store_memory": {"content": "content", "importance": "importance", "category": "category",
                     "user_id": "user_id", "metadata": "metadata"},
    "update_memory": {"memory_id": "memory_id", "new_content": "new_content", "importance": "importance"},
    "session_handoff": {"summary": "summary", "completed": "completed", "in_progress": "in_progress",
                        "blocked": "blocked", "next_steps": "next_steps", "user_id": "user_id"},
    "consolidate_memory": {"user_id": "user_id"},
    "memory_stats": {"user_id": "user_id"},
    "track_failure": {"error": "error", "component": "component", "root_cause": "root_cause",
                      "severity": "severity", "fix": "fix", "related_test_ids": "related_test_ids",
                      "user_id": "user_id"},
    "track_progress": {"feature": "feature", "status": "status", "completion": "completion",
                       "blockers": "blockers", "quality_score": "quality_score", "notes": "notes",
                       "user_id": "user_id"},
    "session_outcome": {"session_id": "session_id", "outcome": "outcome",
                        "notes": "notes", "user_id": "user_id"},
}
