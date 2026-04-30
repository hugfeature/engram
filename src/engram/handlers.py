"""Pure business logic handlers — no transport dependency (MCP or HTTP)."""

from __future__ import annotations

from datetime import datetime, timezone

from .db import MemoryDB
from .graph import MemoryGraph
from .embedding import embed
from .resolve import resolve, Action
from .retrieve import recall
from .consolidator import run_consolidate
from .decay import compute_strength
from .pruner import last_maintenance_time


def handle_recall(db: MemoryDB, graph: MemoryGraph, query: str,
                  user_id: str = "default", top_k: int = 5) -> dict:
    if not query or not query.strip():
        return {"error": "query must be non-empty"}
    top_k = min(max(int(top_k), 1), 100)

    results = recall(query, db, graph, user_id, top_k)
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
    importance = min(max(float(importance), 0.0), 1.0)
    if category not in ("fact", "assumption", "failure", "strategy"):
        category = "fact"

    new_embedding = embed(content)
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
        merged_emb = embed(merged)
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
    existing = db.get_by_id(memory_id)
    if not existing:
        return {"error": f"Memory {memory_id} not found"}

    new_embedding = embed(new_content)
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

    handoff_embedding = embed(content)
    mid = db.insert(content, handoff_embedding, 0.9, "strategy", user_id, metadata=meta)
    graph.index_memory_incremental(mid, handoff_embedding, db, user_id, 0.9, "strategy")

    return {"result": f"Session handoff recorded (id={mid})", "memory_id": mid}


def handle_consolidate(db: MemoryDB, graph: MemoryGraph,
                       user_id: str = "default") -> dict:
    results = run_consolidate(db, graph, user_id)
    if not results:
        msg = "No similar memories found to consolidate"
    else:
        merged_count = sum(len(r["removed"]) for r in results)
        msg = f"Consolidated {len(results)} cluster(s), merged {merged_count} duplicate(s)"
    return {"result": msg, "details": results}


def handle_stats(db: MemoryDB, user_id: str = "default") -> dict:
    memories = db.get_all(user_id)
    now = datetime.now(timezone.utc)

    categories: dict[str, int] = {}
    strengths: list[float] = []
    for m in memories:
        categories[m.category] = categories.get(m.category, 0) + 1
        days = (now - m.last_accessed_at.replace(tzinfo=timezone.utc)).total_seconds() / 86400
        strengths.append(compute_strength(m.category, m.importance, days, m.recall_count))

    return {
        "total": len(memories),
        "categories": categories,
        "avg_strength": round(sum(strengths) / len(strengths), 4) if strengths else 0,
        "last_maintenance": last_maintenance_time.isoformat() if last_maintenance_time else None,
    }


TOOL_HANDLERS = {
    "recall_memory": lambda db, graph, **kw: handle_recall(db, graph, **kw),
    "store_memory": lambda db, graph, **kw: handle_store(db, graph, **kw),
    "update_memory": lambda db, graph, **kw: handle_update(db, graph, **kw),
    "session_handoff": lambda db, graph, **kw: handle_session_handoff(db, graph, **kw),
    "consolidate_memory": lambda db, graph, **kw: handle_consolidate(db, graph, **kw),
    "memory_stats": lambda db, graph, **kw: handle_stats(db, **kw),
}

ARG_MAPPING = {
    "recall_memory": {"query": "query", "user_id": "user_id", "top_k": "top_k"},
    "store_memory": {"content": "content", "importance": "importance", "category": "category",
                     "user_id": "user_id", "metadata": "metadata"},
    "update_memory": {"memory_id": "memory_id", "new_content": "new_content", "importance": "importance"},
    "session_handoff": {"summary": "summary", "completed": "completed", "in_progress": "in_progress",
                        "blocked": "blocked", "next_steps": "next_steps", "user_id": "user_id"},
    "consolidate_memory": {"user_id": "user_id"},
    "memory_stats": {"user_id": "user_id"},
}
