"""Pure business logic handlers — no transport dependency (MCP or HTTP)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from .db import MemoryDB
from .graph import MemoryGraph
from .embedding import embed
from .config import DEDUP_SEARCH_THRESHOLD
from .resolve import resolve, Action
from .retrieve import recall
from .consolidator import run_consolidate
from .decay import compute_strength, compute_quality_score
from .pruner import maintenance

log = logging.getLogger("engram.handlers")

MAX_CONTENT_LENGTH = 100_000  # 100KB


def _error(msg: str) -> dict:
    """Standardized error response — all handlers use this."""
    return {"ok": False, "error": msg}


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


_VALID_MEMORY_TYPES = {"all", "handoff", "failure", "progress"}

_HANDOFF_STEP_FOLLOW_THROUGH_THRESHOLD = 0.82


def _validate_handoff_next_steps(db: MemoryDB, handoff_meta: dict,
                                  user_id: str) -> list[dict] | None:
    """Check if handoff next_steps were followed through in later memories.

    Returns a list of step validation results, or None if no next_steps.
    """
    next_steps = handoff_meta.get("next_steps")
    if not next_steps:
        return None

    handoff_ts = handoff_meta.get("timestamp")
    validations = []
    for step in next_steps:
        step_embedding = _safe_embed(step)
        if step_embedding is None:
            validations.append({"step": step, "status": "unknown", "reason": "embed_failed"})
            continue

        similar = db.search_vector(step_embedding, user_id, top_k=3)
        followed = False
        evidence_content = None
        for row in similar:
            if row.similarity < _HANDOFF_STEP_FOLLOW_THROUGH_THRESHOLD:
                continue
            row_meta = db.get_metadata_batch([row.id]).get(row.id, {})
            row_type = row_meta.get("type", "")
            # Only count progress / handoff / general memories as follow-through
            if row_type in ("progress", "handoff", ""):
                row_ts = row_meta.get("timestamp")
                if handoff_ts and row_ts and row_ts > handoff_ts:
                    followed = True
                    evidence_content = row.content[:120]
                    break
                elif not handoff_ts:
                    followed = True
                    evidence_content = row.content[:120]
                    break

        if followed:
            validations.append({"step": step, "status": "done", "evidence": evidence_content})
        else:
            validations.append({"step": step, "status": "pending"})

    return validations

def handle_recall(db: MemoryDB, graph: MemoryGraph, query: str,
                  user_id: str = "default", top_k: int = 5,
                  session_id: str | None = None,
                  memory_type: str = "all") -> dict:
    if not query or not query.strip():
        return _error("query must be non-empty")
    user_id = _validate_user_id(user_id)
    top_k = min(max(int(top_k), 1), 100)
    if memory_type not in _VALID_MEMORY_TYPES:
        memory_type = "all"

    # Fetch more results when filtering, to compensate for post-filter reduction
    fetch_k = top_k * 3 if memory_type != "all" else top_k
    results = recall(query, db, graph, user_id, fetch_k)

    if session_id and results:
        db.log_session_recall(session_id, [r.id for r in results], user_id)

    meta_batch = db.get_metadata_batch([r.id for r in results]) if results else {}

    # Filter by memory_type if specified
    if memory_type != "all":
        filtered = []
        for r in results:
            meta = meta_batch.get(r.id, {})
            if meta.get("type") == memory_type:
                filtered.append(r)
        results = filtered[:top_k]
    else:
        # Auto-pin: find the latest handoff and move it to top
        handoff_idx = None
        for i, r in enumerate(results):
            meta = meta_batch.get(r.id, {})
            if meta.get("type") == "handoff":
                handoff_idx = i
                break  # meta_batch order matches results (sorted by score desc)

        if handoff_idx is not None and handoff_idx > 0:
            handoff = results.pop(handoff_idx)
            results.insert(0, handoff)

    # --- Recall enhancements (all wrapped in try/except to never break core recall) ---

    failure_context: dict[str, list[dict]] = {}
    outcome_counts: dict[int, dict[str, int]] = {}
    memory_rows: dict = {}
    result_ids = [r.id for r in results]

    try:
        # Collect components referenced in results for batch failure lookup
        component_set: set[str] = set()
        for r in results:
            meta = meta_batch.get(r.id, {})
            component = meta.get("component") or meta.get("feature")
            if component and meta.get("type") != "failure":
                component_set.add(component)

        # Pre-fetch failure context for all referenced components
        for component in component_set:
            failures = db.get_failures_by_component(component, user_id, limit=3)
            if failures:
                failure_context[component] = [
                    {
                        "memory_id": f.id,
                        "error": (f.metadata or {}).get("error", ""),
                        "severity": (f.metadata or {}).get("severity", ""),
                        "fix": (f.metadata or {}).get("fix", ""),
                        "timestamp": (f.metadata or {}).get("timestamp", ""),
                    }
                    for f in failures
                ]
    except Exception as exc:
        log.warning("Recall enhancement (failure context) failed: %s", exc)

    try:
        # Batch-fetch outcome counts and memory rows for quality scoring
        outcome_counts = db.get_memory_outcome_counts(result_ids, user_id) if result_ids else {}
        memory_rows = db.get_by_ids_batch(result_ids) if result_ids else {}
    except Exception as exc:
        log.warning("Recall enhancement (quality scoring data) failed: %s", exc)

    memories_out = []
    for r in results:
        # Compute dynamic quality score (safe — falls back to importance)
        try:
            counts = outcome_counts.get(r.id, {"success": 0, "failure": 0})
            mem_row = memory_rows.get(r.id)
            recall_count = mem_row.recall_count if mem_row else 0
            quality = compute_quality_score(
                importance=r.importance,
                recall_count=recall_count,
                success_count=counts["success"],
                failure_count=counts["failure"],
            )
        except Exception:
            quality = r.importance

        entry = {
            "id": r.id,
            "content": r.content,
            "category": r.category,
            "importance": r.importance,
            "quality_score": round(quality, 4),
            "strength": round(r.strength, 4),
            "similarity": round(r.similarity, 4),
            "score": round(r.score, 4),
        }
        meta = meta_batch.get(r.id, {})
        if meta:
            entry["metadata"] = meta
            # Handoff validation (safe — never breaks recall)
            if meta.get("type") == "handoff":
                try:
                    step_validations = _validate_handoff_next_steps(db, meta, user_id)
                    if step_validations:
                        entry["handoff_validation"] = step_validations
                except Exception as exc:
                    log.warning("Handoff validation failed for memory %d: %s", r.id, exc)
            # Attach related failure context for non-failure memories
            if meta.get("type") != "failure":
                component = meta.get("component") or meta.get("feature")
                if component and component in failure_context:
                    entry["related_failures"] = failure_context[component]
        memories_out.append(entry)

    return {"memoriesFound": len(results), "memories": memories_out}


def handle_store(db: MemoryDB, graph: MemoryGraph, content: str,
                 importance: float, category: str = "fact",
                 user_id: str = "default", metadata: dict | None = None) -> dict:
    if not content or not content.strip():
        return _error("content must be non-empty")
    if len(content) > MAX_CONTENT_LENGTH:
        return _error(f"content too large (max {MAX_CONTENT_LENGTH // 1000}KB)")
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
                return _error("metadata too large (max 10KB)")
        except (TypeError, ValueError):
            return _error("metadata must be JSON-serializable")

    new_embedding = _safe_embed(content)
    if new_embedding is None:
        return _error("Embedding generation failed")
    existing = db.search_similar_for_dedup(new_embedding, user_id, top_k=10, threshold=DEDUP_SEARCH_THRESHOLD)
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
            return _error("Embedding generation failed")
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
        return _error("new_content must be non-empty")
    if len(new_content) > MAX_CONTENT_LENGTH:
        return _error(f"new_content too large (max {MAX_CONTENT_LENGTH // 1000}KB)")
    try:
        memory_id = int(memory_id)
    except (TypeError, ValueError):
        return _error("memory_id must be an integer")
    existing = db.get_by_id(memory_id)
    if not existing:
        return _error(f"Memory {memory_id} not found")

    new_embedding = _safe_embed(new_content)
    if new_embedding is None:
        return _error("Embedding generation failed")
    db.update(memory_id, new_content, new_embedding, importance)

    effective_importance = importance if importance is not None else existing.importance
    graph.index_memory_incremental(
        memory_id, new_embedding, db,
        existing.user_id, effective_importance, existing.category,
    )
    return {"result": f"Updated memory (id={memory_id})"}


def handle_session_handoff(db: MemoryDB, graph: MemoryGraph, summary: str,
                           completed: list[str] | None = None,
                           in_progress: list[str] | None = None,
                           blocked: list[str] | None = None,
                           next_steps: list[str] | None = None,
                           user_id: str = "default",
                           task_id: int | None = None) -> dict:
    if not summary or not summary.strip():
        return _error("summary must be non-empty")
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
    if task_id is not None:
        meta["task_id"] = task_id

    handoff_embedding = _safe_embed(content)
    if handoff_embedding is None:
        return _error("Embedding generation failed")
    mid = db.insert(content, handoff_embedding, 0.9, "strategy", user_id, metadata=meta)
    graph.index_memory_incremental(mid, handoff_embedding, db, user_id, 0.9, "strategy")

    result = {"result": f"Session handoff recorded (id={mid})", "memory_id": mid}
    if task_id is not None:
        result["task_id"] = task_id
    return result


def handle_consolidate(db: MemoryDB, graph: MemoryGraph,
                       user_id: str = "default") -> dict:
    try:
        results = run_consolidate(db, graph, user_id)
    except Exception as e:
        log.error("Consolidation failed: %s", e)
        return _error(f"consolidation failed: {e}")
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

    # Engineering stats — pre-filter structured metadata via SQL
    all_meta = db.get_metadata_for_stats(user_id, types=("failure", "progress"))
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
                         user_id: str = "default",
                         task_id: int | None = None) -> dict:
    if not error or not error.strip():
        return _error("error must be non-empty")
    if not component or not component.strip():
        return _error("component must be non-empty")
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
    if task_id is not None:
        meta["task_id"] = task_id

    importance = {"critical": 0.9, "major": 0.7, "minor": 0.5}[severity]
    emb = _safe_embed(content)
    if emb is None:
        return _error("Embedding generation failed")
    mid = db.insert(content, emb, importance, "failure", user_id, metadata=meta)
    graph.index_memory_incremental(mid, emb, db, user_id, importance, "failure")

    result = {"result": f"Failure tracked (id={mid})", "memory_id": mid}
    if task_id is not None:
        result["task_id"] = task_id
    return result


def handle_track_progress(db: MemoryDB, graph: MemoryGraph, feature: str,
                          status: str, completion: float = 0,
                          blockers: list[str] | None = None,
                          quality_score: float | None = None,
                          notes: str | None = None,
                          user_id: str = "default",
                          task_id: int | None = None) -> dict:
    if not feature or not feature.strip():
        return _error("feature must be non-empty")
    valid_statuses = ("planning", "in_progress", "blocked", "review", "done")
    if status not in valid_statuses:
        return _error(f"status must be one of {valid_statuses}")
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
    if task_id is not None:
        meta["task_id"] = task_id

    importance_map = {
        "planning": 0.6, "in_progress": 0.8,
        "blocked": 0.9, "review": 0.7, "done": 0.5,
    }
    importance = importance_map[status]
    emb = _safe_embed(content)
    if emb is None:
        return _error("Embedding generation failed")
    mid = db.insert(content, emb, importance, "strategy", user_id, metadata=meta)
    graph.index_memory_incremental(mid, emb, db, user_id, importance, "strategy")

    result = {"result": f"Progress tracked (id={mid})", "memory_id": mid}
    if task_id is not None:
        result["task_id"] = task_id
    return result


def handle_session_outcome(db: MemoryDB, graph: MemoryGraph, session_id: str,
                           outcome: str, notes: str | None = None,
                           user_id: str = "default") -> dict:
    if not session_id or not session_id.strip():
        return _error("session_id must be non-empty")
    if outcome not in ("success", "failure"):
        return _error("outcome must be 'success' or 'failure'")
    user_id = _validate_user_id(user_id)

    memory_ids = db.get_session_memories(session_id, user_id)
    if not memory_ids:
        return {
            "result": "No memories recalled in this session",
            "session_id": session_id,
            "outcome": outcome,
            "memories_adjusted": 0,
        }

    # Record outcome for historical tracking
    db.log_session_outcome(session_id, outcome, user_id)

    extra_demoted = 0
    if outcome == "success":
        adjusted = db.adjust_importance_batch(memory_ids, +0.10)
    else:
        adjusted = db.adjust_importance_batch(memory_ids, -0.05)

        # Extra demotion for memories involved in multiple failed sessions
        failure_counts = db.get_memory_failure_count(memory_ids, user_id)
        repeat_offenders = [mid for mid, count in failure_counts.items() if count >= 2]
        if repeat_offenders:
            extra_penalty = -0.05  # Additional penalty per repeated failure
            extra_demoted = db.adjust_importance_batch(repeat_offenders, extra_penalty)
            log.info(
                "Extra demotion applied to %d memories with %s repeated failures",
                extra_demoted, {mid: failure_counts[mid] for mid in repeat_offenders},
            )

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

    result = {
        "result": f"Session outcome recorded ({outcome})",
        "session_id": session_id,
        "outcome": outcome,
        "memories_adjusted": adjusted,
    }
    if extra_demoted:
        result["extra_demoted"] = extra_demoted
    return result


def handle_create_task(db: MemoryDB, graph: MemoryGraph, name: str,
                       goal: str = "", status: str = "planning",
                       user_id: str = "default",
                       metadata: dict | None = None) -> dict:
    if not name or not name.strip():
        return _error("name must be non-empty")
    valid_statuses = ("planning", "in_progress", "blocked", "review", "done", "cancelled")
    if status not in valid_statuses:
        return _error(f"status must be one of {valid_statuses}")
    user_id = _validate_user_id(user_id)
    task_id = db.create_task(name=name.strip(), goal=goal, status=status,
                             user_id=user_id, metadata=metadata)
    return {"result": f"Task created (id={task_id})", "task_id": task_id}


def handle_update_task(db: MemoryDB, graph: MemoryGraph, task_id: int,
                       status: str | None = None, goal: str | None = None,
                       user_id: str = "default",
                       metadata: dict | None = None) -> dict:
    try:
        task_id = int(task_id)
    except (TypeError, ValueError):
        return _error("task_id must be an integer")
    if status is not None:
        valid_statuses = ("planning", "in_progress", "blocked", "review", "done", "cancelled")
        if status not in valid_statuses:
            return _error(f"status must be one of {valid_statuses}")
    user_id = _validate_user_id(user_id)
    existing = db.get_task(task_id)
    if not existing:
        return _error(f"Task {task_id} not found")
    if existing.user_id != user_id:
        return _error(f"Task {task_id} not found for user '{user_id}'")
    updated = db.update_task(task_id, status=status, goal=goal, metadata=metadata)
    if not updated:
        return _error(f"Failed to update task {task_id}")
    return {"result": f"Task updated (id={task_id})", "task_id": task_id}


def handle_get_task(db: MemoryDB, graph: MemoryGraph, task_id: int,
                    user_id: str = "default") -> dict:
    try:
        task_id = int(task_id)
    except (TypeError, ValueError):
        return _error("task_id must be an integer")
    user_id = _validate_user_id(user_id)
    task = db.get_task(task_id)
    if not task:
        return _error(f"Task {task_id} not found")
    if task.user_id != user_id:
        return _error(f"Task {task_id} not found for user '{user_id}'")

    memories = db.get_task_memories(task_id, user_id)
    handoffs = []
    failures = []
    progress_snapshots = []
    other_memories = []
    for m in memories:
        meta = m.metadata or {}
        memory_type = meta.get("type", "")
        entry = {
            "memory_id": m.id,
            "content": m.content,
            "importance": m.importance,
            "created_at": m.created_at.isoformat() if m.created_at else None,
            "metadata": meta,
        }
        if memory_type == "handoff":
            handoffs.append(entry)
        elif memory_type == "failure":
            failures.append(entry)
        elif memory_type == "progress":
            progress_snapshots.append(entry)
        else:
            other_memories.append(entry)

    return {
        "task": {
            "id": task.id,
            "name": task.name,
            "goal": task.goal,
            "status": task.status,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "updated_at": task.updated_at.isoformat() if task.updated_at else None,
            "metadata": task.metadata,
        },
        "handoffs": handoffs,
        "failures": failures,
        "progress": progress_snapshots,
        "other_memories": other_memories,
        "total_memories": len(memories),
    }


def handle_list_tasks(db: MemoryDB, graph: MemoryGraph,
                      user_id: str = "default",
                      status: str | None = None) -> dict:
    user_id = _validate_user_id(user_id)
    tasks = db.list_tasks(user_id, status=status)
    return {
        "tasks": [
            {
                "id": t.id,
                "name": t.name,
                "goal": t.goal,
                "status": t.status,
                "created_at": t.created_at.isoformat() if t.created_at else None,
                "updated_at": t.updated_at.isoformat() if t.updated_at else None,
                "metadata": t.metadata,
            }
            for t in tasks
        ],
        "total": len(tasks),
    }

TOOL_HANDLERS = {
    "recall_memory": handle_recall,
    "store_memory": handle_store,
    "update_memory": handle_update,
    "session_handoff": handle_session_handoff,
    "consolidate_memory": handle_consolidate,
    "memory_stats": lambda db, graph, **kw: handle_stats(db, user_id=kw.get("user_id", "default")),
    "track_failure": handle_track_failure,
    "track_progress": handle_track_progress,
    "session_outcome": handle_session_outcome,
    "create_task": handle_create_task,
    "update_task": handle_update_task,
    "get_task": handle_get_task,
    "list_tasks": handle_list_tasks,
}

ARG_MAPPING = {
    "recall_memory": {"query": "query", "user_id": "user_id", "top_k": "top_k", "session_id": "session_id", "memory_type": "memory_type"},
    "store_memory": {"content": "content", "importance": "importance", "category": "category",
                     "user_id": "user_id", "metadata": "metadata"},
    "update_memory": {"memory_id": "memory_id", "new_content": "new_content", "importance": "importance"},
    "session_handoff": {"summary": "summary", "completed": "completed", "in_progress": "in_progress",
                        "blocked": "blocked", "next_steps": "next_steps", "user_id": "user_id",
                        "task_id": "task_id"},
    "consolidate_memory": {"user_id": "user_id"},
    "memory_stats": {"user_id": "user_id"},
    "track_failure": {"error": "error", "component": "component", "root_cause": "root_cause",
                      "severity": "severity", "fix": "fix", "related_test_ids": "related_test_ids",
                      "user_id": "user_id", "task_id": "task_id"},
    "track_progress": {"feature": "feature", "status": "status", "completion": "completion",
                       "blockers": "blockers", "quality_score": "quality_score", "notes": "notes",
                       "user_id": "user_id", "task_id": "task_id"},
    "session_outcome": {"session_id": "session_id", "outcome": "outcome",
                        "notes": "notes", "user_id": "user_id"},
    "create_task": {"name": "name", "goal": "goal", "status": "status",
                    "user_id": "user_id", "metadata": "metadata"},
    "update_task": {"task_id": "task_id", "status": "status", "goal": "goal",
                    "user_id": "user_id", "metadata": "metadata"},
    "get_task": {"task_id": "task_id", "user_id": "user_id"},
    "list_tasks": {"user_id": "user_id", "status": "status"},
}
