"""Pure business logic handlers — no transport dependency (MCP or HTTP)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from .db import MemoryDB, DegradedModeError
from .graph import MemoryGraph
from .embedding import embed
from .config import DEDUP_SEARCH_THRESHOLD
from .resolve import resolve, Action
from .retrieve import recall
from .consolidator import run_consolidate
from .decay import compute_strength, compute_quality_score
from .pruner import maintenance
from . import checkpoint

log = logging.getLogger("engram.handlers")

MAX_CONTENT_LENGTH = 100_000  # 100KB


def _error(msg: str) -> dict:
    """Standardized error response — all handlers use this."""
    return {"ok": False, "error": msg}


def _degraded_error(exc: DegradedModeError) -> dict:
    """Structured response for readonly degraded mode.

    MCP clients (Claude Code / Cursor) can branch on ``code='degraded_mode'``
    and surface ``recover_command`` to the user.
    """
    return {
        "ok": False,
        "code": "degraded_mode",
        "error": str(exc),
        "recover_command": exc.recover_command,
        "db_path": exc.db_path,
    }


import re


def _validate_user_id(user_id: str) -> str:
    """Validate and sanitize user_id to prevent injection attacks.
    
    Only allows alphanumeric characters, underscores, and hyphens.
    Maximum length: 100 characters.
    """
    if not user_id or not isinstance(user_id, str):
        return "default"
    
    # Only allow letters, numbers, underscores, and hyphens
    cleaned = re.sub(r'[^a-zA-Z0-9_-]', '', user_id.strip()[:100])
    return cleaned or "default"


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

    # Session lifecycle: register/refresh heartbeat
    if session_id:
        try:
            db.upsert_session(session_id, user_id)
        except Exception as exc:
            log.debug("Session upsert failed (non-fatal): %s", exc)

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

    # Detect interrupted sessions and attach reminder to response
    interrupted_sessions = []
    if session_id:
        try:
            interrupted_sessions = db.get_interrupted_sessions(user_id, stale_minutes=30)
        except Exception as exc:
            log.debug("Interrupted sessions check failed (non-fatal): %s", exc)

    result = {"memoriesFound": len(results), "memories": memories_out}
    if interrupted_sessions:
        result["interrupted_sessions"] = [
            _build_recovery_hint(s) for s in interrupted_sessions
        ]

    # Auto-inject latest interrupt checkpoint so the agent sees it on first
    # recall without needing to call restore_checkpoint manually.
    # Wrapped in try/except — never breaks core recall.
    try:
        interrupt_ckpt = db.get_latest_interrupt_checkpoint(user_id)
        if interrupt_ckpt:
            ws = interrupt_ckpt["state"].get("working_set") or {}
            interrupt_meta = ws.get("_interrupt", {})
            result["interrupt_recovery"] = {
                "task_id": interrupt_ckpt["task_id"],
                "checkpoint_version": interrupt_ckpt["version"],
                "goal": interrupt_ckpt["state"].get("goal", ""),
                "modified_files": ws.get("files", []),
                "last_tool_called": interrupt_meta.get("last_tool_called", ""),
                "last_success_action": interrupt_meta.get("last_success_action", ""),
                "last_failure": interrupt_meta.get("last_failure", ""),
                "interrupt_reason": interrupt_meta.get("interrupt_reason", "process_exit"),
                "hint": (
                    f"Previous session was interrupted. "
                    f"Call restore_checkpoint(task_id={interrupt_ckpt['task_id']}) to resume."
                ),
            }
    except Exception as exc:
        log.debug("Interrupt checkpoint auto-inject failed (non-fatal): %s", exc)

    # Context pressure warning: if agent is mid-task and has interrupt state,
    # remind it to save proactively before context fills up.
    if result.get("interrupt_recovery") or result.get("interrupted_sessions"):
        result["context_pressure_hint"] = (
            "⚠️ Interrupted state detected. If context window is filling up, "
            "call report_interruption(reason='overflow') then "
            "session_handoff(..., task_id=X) before context is lost."
        )

    return result


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
                           must_not_redo: list[dict] | None = None,
                           must_preserve: list[str] | None = None,
                           working_set: dict | None = None,
                           user_id: str = "default",
                           session_id: str | None = None,
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

    # Mark any active sessions for this user as ended via handoff
    try:
        db.cleanup_stale_sessions(user_id, stale_minutes=0)
    except Exception as exc:
        log.debug("Session cleanup on handoff failed (non-fatal): %s", exc)

    result = {"result": f"Session handoff recorded (id={mid})", "memory_id": mid}
    if task_id is not None:
        result["task_id"] = task_id

    # Cognitive Continuation：写入 MANUAL_HANDOFF checkpoint（仅在 task_id 给定时）
    if task_id is not None:
        try:
            task_row = db.get_task(task_id)
            goal = task_row.goal if task_row else ""
            ckpt_state = {
                "goal": goal,
                "completed": completed,
                "in_progress": in_progress,
                "blocked": blocked,
                "preferred_next": next_steps,
                "must_not_redo": must_not_redo or [],
                "must_preserve": must_preserve or [],
                "working_set": working_set or {},
            }
            ckpt = checkpoint.create_checkpoint(
                db,
                task_id=task_id,
                reason=checkpoint.REASON_MANUAL_HANDOFF,
                state=ckpt_state,
                source_session_id=session_id,
                source_memory_id=mid,
                user_id=user_id,
                triggered_by_event="session_handoff",
            )
            result["checkpoint_id"] = ckpt["checkpoint_id"]
            result["checkpoint_version"] = ckpt["version"]
            result["continuation_confidence"] = ckpt["continuation_confidence"]
        except Exception as exc:
            # checkpoint 失败不影响 handoff 主流程（向后兼容）
            log.warning("Checkpoint creation on handoff failed (non-fatal): %s", exc)

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

    # Cognitive Continuation：FAILURE 强触发 checkpoint（不受 debounce 限制）
    if task_id is not None:
        try:
            task_row = db.get_task(task_id)
            goal = task_row.goal if task_row else ""
            # 失败的 component 加入 working_set.tools，root_cause 进 blocked
            ckpt_state = {
                "goal": goal,
                "blocked": [error] + ([root_cause] if root_cause else []),
                "working_set": {"tools": [component]},
            }
            failure_sig = checkpoint.derive_failure_signature(component, severity)
            ckpt_meta = checkpoint.create_checkpoint(
                db,
                task_id=task_id,
                reason=checkpoint.REASON_FAILURE,
                state=ckpt_state,
                source_memory_id=mid,
                user_id=user_id,
                failure_signature=failure_sig,
                triggered_by_event="track_failure",
            )
            result["checkpoint_version"] = ckpt_meta["version"]
            result["checkpoint_reason"] = ckpt_meta["reason"]
        except Exception as exc:
            log.warning("Checkpoint creation on failure failed (non-fatal): %s", exc)

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

    # Cognitive Continuation：event-first 决定是否写 auto checkpoint
    # 仅在 task_id 给定时触发；done 状态进 completed，其他进 in_progress
    if task_id is not None:
        try:
            task_row = db.get_task(task_id)
            goal = task_row.goal if task_row else ""
            feature_label = f"{feature} [{status}]"
            payload_state = {
                "goal": goal,
                "completed": [feature_label] if status == "done" else [],
                "in_progress": [feature_label] if status != "done" else [],
                "blocked": list(blockers) if blockers else [],
                "working_set": {"artifacts": [feature]},
            }
            should, reason = checkpoint.should_create_auto_checkpoint(
                db, task_id, "progress", payload_state, user_id=user_id,
            )
            if should and reason:
                ckpt_meta = checkpoint.create_checkpoint(
                    db,
                    task_id=task_id,
                    reason=reason,
                    state=payload_state,
                    source_memory_id=mid,
                    user_id=user_id,
                    triggered_by_event="track_progress",
                )
                result["checkpoint_version"] = ckpt_meta["version"]
                result["checkpoint_reason"] = ckpt_meta["reason"]
        except Exception as exc:
            log.warning("Checkpoint creation on progress failed (non-fatal): %s", exc)

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

    result = {
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

    # Cognitive Continuation：附带 latest_checkpoint（精简版，不含 related_memories
    # 以避免响应膨胀；需要完整恢复请用 restore_checkpoint）
    try:
        latest = checkpoint.get_checkpoint(db, task_id, user_id=user_id)
        if latest is not None:
            result["latest_checkpoint"] = {
                "version": latest["version"],
                "kind": latest["kind"],
                "checkpoint_reason": latest["reason"],
                "created_at": latest["created_at"].isoformat() if latest.get("created_at") else None,
                "failure_signature": latest.get("failure_signature"),
                "continuation": checkpoint.build_continuation(latest, db=db),
            }
    except Exception as exc:
        log.debug("get_task latest_checkpoint injection failed (non-fatal): %s", exc)

    return result


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


def _validate_task_id(task_id) -> int | None:
    """Coerce task_id to int; return None if invalid."""
    if task_id is None:
        return None
    try:
        return int(task_id)
    except (TypeError, ValueError):
        return None


def handle_restore_checkpoint(db: MemoryDB, graph: MemoryGraph,
                              task_id, version: int | None = None,
                              memory_restore_mode: str = "SELECTIVE",
                              user_id: str = "default") -> dict:
    """Restore a constrained continuation package from a task checkpoint.

    Args:
        task_id: target task
        version: specific checkpoint version, or None for latest
        memory_restore_mode: FULL / SELECTIVE (default) / NONE — controls
            related memory recall to mitigate context pollution
        user_id: tenant isolation

    Returns:
        {task_id, version, kind, reason, created_at, continuation, ...}
        On no-checkpoint: {error: "no_checkpoint", fallback: {...}}
    """
    tid = _validate_task_id(task_id)
    if tid is None:
        return _error("task_id must be a valid integer")
    user_id = _validate_user_id(user_id)

    mode = (memory_restore_mode or "SELECTIVE").upper()
    if mode not in checkpoint.VALID_RESTORE_MODES:
        mode = "SELECTIVE"

    ckpt_row = checkpoint.get_checkpoint(db, tid, version=version, user_id=user_id)
    if ckpt_row is None:
        # Fallback：老 task 无 checkpoint 时，引导调用 get_task + recall_memory
        task_row = db.get_task(tid)
        if task_row is None or task_row.user_id != user_id:
            return _error(f"task {tid} not found")
        memories = db.get_task_memories(tid, user_id)
        return {
            "ok": False,
            "error": "no_checkpoint",
            "task_id": tid,
            "fallback": {
                "task": {
                    "id": task_row.id,
                    "name": task_row.name,
                    "goal": task_row.goal,
                    "status": task_row.status,
                },
                "associated_memories_count": len(memories),
                "hint": "No checkpoint exists for this task. Call recall_memory(query=task_name) and get_task(task_id) instead.",
            },
        }

    continuation = checkpoint.build_continuation(ckpt_row, db=db)

    result = {
        "task_id": tid,
        "version": ckpt_row["version"],
        "kind": ckpt_row["kind"],
        "checkpoint_reason": ckpt_row["reason"],
        "created_at": ckpt_row["created_at"].isoformat() if ckpt_row.get("created_at") else None,
        "source_session_id": ckpt_row.get("source_session_id"),
        "failure_signature": ckpt_row.get("failure_signature"),
        "continuation": continuation,
        "memory_restore_mode": mode,
    }

    # Memory recall — controlled by mode to mitigate context pollution
    if mode != "NONE":
        try:
            related, related_failures = _collect_continuation_memories(
                db, graph, ckpt_row, mode, user_id,
            )
            result["related_memories"] = related
            result["related_failures"] = related_failures
            result["memory_filter_applied"] = (
                "all_task_memories" if mode == "FULL"
                else "quality>=0.5 + failure_forced"
            )
        except Exception as exc:
            log.warning("Restore memory recall failed (non-fatal): %s", exc)
            result["related_memories"] = []
            result["related_failures"] = []

    # Continuity Metrics — compare restored checkpoint against its parent
    try:
        from . import continuity as _cont
        parent_version = ckpt_row.get("parent_version")
        if parent_version is not None:
            score = _cont.evaluate_from_checkpoints(
                db, tid,
                before_version=parent_version,
                after_version=ckpt_row["version"],
                user_id=user_id,
            )
            if score is not None:
                result["continuity_score"] = score.to_dict()
    except Exception as exc:
        log.debug("Continuity score computation failed (non-fatal): %s", exc)

    return result


def _collect_continuation_memories(db: MemoryDB, graph: MemoryGraph,
                                    ckpt_row: dict, mode: str,
                                    user_id: str) -> tuple[list[dict], list[dict]]:
    """Collect related memories for a checkpoint restore.

    FULL: all task memories (cap 20).
    SELECTIVE: importance >= 0.5 OR category=failure (cap 10).
    """
    task_id = ckpt_row["task_id"]
    memories = db.get_task_memories(task_id, user_id)

    if mode == "SELECTIVE":
        memories = [m for m in memories if m.importance >= 0.5 or m.category == "failure"]
        memories = memories[:10]
    else:  # FULL
        memories = memories[:20]

    related = [
        {
            "memory_id": m.id,
            "category": m.category,
            "importance": round(m.importance, 3),
            "content": m.content[:300],
            "metadata_type": (m.metadata or {}).get("type", ""),
        }
        for m in memories
    ]

    # Always include failure context for components/tools referenced in working_set
    working_set = ckpt_row["state"].get("working_set", {}) or {}
    components: set[str] = set()
    for key in ("tools", "artifacts"):
        for v in working_set.get(key, []) or []:
            if isinstance(v, str):
                components.add(v)

    related_failures: list[dict] = []
    for comp in components:
        failures = db.get_failures_by_component(comp, user_id, limit=3)
        for f in failures:
            meta = f.metadata or {}
            related_failures.append({
                "memory_id": f.id,
                "component": comp,
                "error": meta.get("error", ""),
                "severity": meta.get("severity", ""),
                "fix": meta.get("fix", ""),
                "timestamp": meta.get("timestamp", ""),
            })

    return related, related_failures


def handle_list_checkpoints(db: MemoryDB, graph: MemoryGraph,
                            task_id, limit: int = 10,
                            user_id: str = "default") -> dict:
    """List checkpoint history for a task (latest first, no full state)."""
    tid = _validate_task_id(task_id)
    if tid is None:
        return _error("task_id must be a valid integer")
    user_id = _validate_user_id(user_id)
    try:
        limit = min(max(int(limit), 1), 100)
    except (TypeError, ValueError):
        limit = 10

    rows = checkpoint.list_checkpoints(db, tid, limit=limit, user_id=user_id)
    return {
        "task_id": tid,
        "checkpoints": rows,
        "total": len(rows),
    }


def handle_get_runtime_health(db: MemoryDB, graph: MemoryGraph, **_kw) -> dict:
    """Expose ``recover.doctor()`` as an MCP tool result.

    Notes for callers:
      - Always returns ``{"ok": True, ...}`` because this is a *read-only*
        diagnostic; even DB-corrupt scenarios should still produce data.
      - Includes a top-level ``advice`` field summarising what the operator
        should do (recover / clean residue / nothing). LLMs should read it
        first instead of re-deriving from the raw fields.
    """
    from .recover import doctor
    info = doctor()

    advice: list[str] = []
    if info.get("readonly"):
        advice.append(
            "DB is in readonly degraded mode. Run `engram-setup recover` "
            "to rebuild from the event log."
        )
    if info.get("residue_files"):
        advice.append(
            f"{len(info['residue_files'])} residue file(s) present "
            "(prior corruption). Inspect, then delete manually if no longer "
            "needed."
        )
    if info.get("embedding_stale"):
        advice.append(
            "Embedding column dim drifted from the active model. Vector "
            "recall is falling back to BM25 until you re-embed."
        )
    backups = info.get("backups") or {}
    if backups.get("live_count", 0) > backups.get("retain", 0):
        advice.append(
            f"{backups['live_count']} live backups exceed retention "
            f"({backups['retain']}); surplus will archive on next boot."
        )

    return {"ok": True, "advice": advice, **info}


# --- Interruption Taxonomy (v0.12) ---

def handle_report_interruption(db: MemoryDB, graph: MemoryGraph,
                               reason: str,
                               context: dict | None = None,
                               session_id: str | None = None,
                               user_id: str = "default") -> dict:
    """LLM-reported interruption reason.

    Called by the LLM when it detects an imminent interruption (e.g. context
    window overflow, rate limit). The reason is stored in process-level state
    and flushed to session_lifecycle on process exit via atexit.

    If session_id is provided, the session is also updated immediately so the
    taxonomy is visible even if atexit doesn't fire.
    """
    from .db import VALID_INTERRUPTION_REASONS, INTERRUPTION_UNKNOWN, RECOVERY_STRATEGIES
    from .shared import set_interruption_report

    user_id = _validate_user_id(user_id)
    if reason not in VALID_INTERRUPTION_REASONS:
        return _error(f"Invalid interruption reason '{reason}'. "
                      f"Valid: {sorted(VALID_INTERRUPTION_REASONS)}")

    set_interruption_report(reason, context)

    if session_id:
        try:
            db.end_session(
                session_id,
                end_type="interrupted",
                interruption_reason=reason,
                interruption_context=context,
            )
        except Exception as exc:
            log.warning("Immediate session close failed (will retry on exit): %s", exc)

    # v0.17: Interruption Intelligence — enrich with severity/recoverability
    strategy = RECOVERY_STRATEGIES.get(reason, RECOVERY_STRATEGIES[INTERRUPTION_UNKNOWN])

    return {
        "ok": True,
        "reason": reason,
        "result": f"Interruption reported: {reason}. Session will be classified on exit.",
        "intelligence": {
            "severity": strategy.get("severity", "medium"),
            "recoverability": strategy.get("recoverability", "low"),
            "data_loss_risk": strategy.get("data_loss_risk", "partial"),
            "recommended_action": strategy["action"],
            "memory_restore_mode": strategy.get("memory_restore_mode", "NONE"),
        },
    }


def _build_recovery_hint(session: dict) -> dict:
    """Build a taxonomy-aware recovery hint for an interrupted session.

    Returns a dict with session_id, timing info, interruption_reason,
    recovery_strategy, and a human-readable hint.
    """
    from .db import RECOVERY_STRATEGIES, INTERRUPTION_UNKNOWN

    reason = session.get("interruption_reason") or INTERRUPTION_UNKNOWN
    strategy = RECOVERY_STRATEGIES.get(reason, RECOVERY_STRATEGIES[INTERRUPTION_UNKNOWN])

    result = {
        "session_id": session["session_id"],
        "started_at": str(session["started_at"]),
        "last_active_at": str(session["last_active_at"]),
        "interruption_reason": reason,
        "recovery_strategy": strategy["action"],
        "memory_restore_mode": strategy.get("memory_restore_mode", "NONE"),
        "hint": strategy["hint"],
    }
    context = session.get("interruption_context")
    if context and context != {}:
        result["interruption_context"] = context
    return result


def handle_evaluate_continuity(db: MemoryDB, graph: MemoryGraph,
                               task_id, before_version: int | None = None,
                               after_version: int | None = None,
                               actions_taken_after_restore: list[str] | None = None,
                               user_id: str = "default") -> dict:
    """Evaluate continuity metrics between two checkpoint versions."""
    from . import continuity as _cont

    tid = _validate_task_id(task_id)
    if tid is None:
        return _error("task_id must be a valid integer")
    user_id = _validate_user_id(user_id)

    score = _cont.evaluate_from_checkpoints(
        db, tid,
        before_version=before_version,
        after_version=after_version,
        user_id=user_id,
        actions_taken_after_restore=actions_taken_after_restore,
    )
    if score is None:
        return _error("Need at least 2 checkpoints to evaluate continuity")

    return {
        "ok": True,
        "task_id": tid,
        "before_version": before_version,
        "after_version": after_version,
        "continuity_score": score.to_dict(),
    }


# ============================================================
# Execution Lineage — v0.16 Durable Runtime Continuity
# ============================================================

import uuid as _uuid


def handle_start_execution(db: MemoryDB, graph: MemoryGraph,
                           goal: str, user_id: str = "default",
                           origin_checkpoint: str | None = None) -> dict:
    """Start a new execution lineage — a continuous runtime intent."""
    if not goal or not goal.strip():
        return _error("goal must be non-empty")
    user_id = _validate_user_id(user_id)
    execution_id = str(_uuid.uuid4())
    try:
        db.create_execution(
            execution_id=execution_id,
            root_goal=goal.strip(),
            user_id=user_id,
            origin_checkpoint=origin_checkpoint,
        )
    except DegradedModeError as exc:
        return _degraded_error(exc)
    # Create the first task (attempt #1) within this execution
    task_id = db.create_task_in_execution(
        name=goal.strip()[:100],
        goal=goal.strip(),
        execution_id=execution_id,
        user_id=user_id,
        checkpoint_id=origin_checkpoint,
        attempt=1,
    )
    db._emit_event("task.create", {
        "task_id": task_id,
        "name": goal.strip()[:100],
        "goal": goal.strip(),
        "execution_id": execution_id,
        "user_id": user_id,
        "checkpoint_id": origin_checkpoint,
        "attempt": 1,
    })
    return {
        "ok": True,
        "execution_id": execution_id,
        "task_id": task_id,
        "attempt": 1,
        "message": f"Execution started. task_id={task_id} is attempt #1.",
    }


def handle_retry_task(db: MemoryDB, graph: MemoryGraph,
                      task_id: int, reason: str = "",
                      user_id: str = "default") -> dict:
    """Retry a failed task within the same execution lineage."""
    tid = _validate_task_id(task_id)
    if tid is None:
        return _error("task_id must be a valid integer")
    user_id = _validate_user_id(user_id)
    original = db.get_task(tid)
    if not original:
        return _error(f"Task {tid} not found")
    if original.user_id != user_id:
        return _error(f"Task {tid} not found for user '{user_id}'")
    if not original.execution_id:
        return _error(f"Task {tid} is not part of an execution lineage")

    # Determine attempt number from retry chain
    chain = db.get_retry_chain(tid)
    new_attempt = len(chain) + 1

    try:
        new_task_id = db.create_task_in_execution(
            name=original.name,
            goal=original.goal,
            execution_id=original.execution_id,
            user_id=user_id,
            previous_task_id=tid,
            retry_of_task_id=tid,
            attempt=new_attempt,
        )
    except DegradedModeError as exc:
        return _degraded_error(exc)

    # Mark original task as failed if not already
    if original.status not in ("done", "cancelled"):
        db.update_task(tid, status="cancelled")

    db._emit_event("task.retry", {
        "task_id": new_task_id,
        "retry_of_task_id": tid,
        "execution_id": original.execution_id,
        "attempt": new_attempt,
        "reason": reason,
        "user_id": user_id,
    })

    return {
        "ok": True,
        "execution_id": original.execution_id,
        "new_task_id": new_task_id,
        "retry_of_task_id": tid,
        "attempt": new_attempt,
        "message": f"Retry created. task_id={new_task_id} is attempt #{new_attempt} of execution {original.execution_id}.",
    }


def handle_spawn_subtask(db: MemoryDB, graph: MemoryGraph,
                         parent_task_id: int, name: str, goal: str = "",
                         user_id: str = "default",
                         checkpoint_id: str | None = None) -> dict:
    """Spawn a subtask within the same execution lineage."""
    ptid = _validate_task_id(parent_task_id)
    if ptid is None:
        return _error("parent_task_id must be a valid integer")
    if not name or not name.strip():
        return _error("name must be non-empty")
    user_id = _validate_user_id(user_id)
    parent = db.get_task(ptid)
    if not parent:
        return _error(f"Parent task {ptid} not found")
    if parent.user_id != user_id:
        return _error(f"Parent task {ptid} not found for user '{user_id}'")
    if not parent.execution_id:
        return _error(f"Parent task {ptid} is not part of an execution lineage")

    try:
        new_task_id = db.create_task_in_execution(
            name=name.strip(),
            goal=goal or name.strip(),
            execution_id=parent.execution_id,
            user_id=user_id,
            parent_task_id=ptid,
            checkpoint_id=checkpoint_id,
            attempt=1,
        )
    except DegradedModeError as exc:
        return _degraded_error(exc)

    db._emit_event("task.spawn", {
        "task_id": new_task_id,
        "parent_task_id": ptid,
        "execution_id": parent.execution_id,
        "name": name.strip(),
        "goal": goal or name.strip(),
        "checkpoint_id": checkpoint_id,
        "user_id": user_id,
    })

    return {
        "ok": True,
        "execution_id": parent.execution_id,
        "new_task_id": new_task_id,
        "parent_task_id": ptid,
        "message": f"Subtask spawned (id={new_task_id}) under parent task {ptid}.",
    }


def handle_trace_execution(db: MemoryDB, graph: MemoryGraph,
                           execution_id: str, user_id: str = "default") -> dict:
    """Trace the full execution lineage — all attempts, retries, spawns."""
    if not execution_id or not execution_id.strip():
        return _error("execution_id must be non-empty")
    user_id = _validate_user_id(user_id)
    execution = db.get_execution(execution_id)
    if not execution:
        return _error(f"Execution {execution_id} not found")
    if execution.get("user_id") != user_id:
        return _error(f"Execution {execution_id} not found for user '{user_id}'")

    tasks = db.get_execution_tasks(execution_id)

    # Build lineage structure
    task_nodes = []
    for t in tasks:
        node = {
            "task_id": t.id,
            "name": t.name,
            "goal": t.goal,
            "status": t.status,
            "attempt": t.attempt,
            "parent_task_id": t.parent_task_id,
            "previous_task_id": t.previous_task_id,
            "retry_of_task_id": t.retry_of_task_id,
            "checkpoint_id": t.checkpoint_id,
            "created_at": t.created_at.isoformat() if t.created_at else None,
        }
        task_nodes.append(node)

    # Identify current active task
    active_tasks = [t for t in tasks if t.status in ("planning", "in_progress", "blocked")]
    current_task = active_tasks[-1] if active_tasks else (tasks[-1] if tasks else None)

    return {
        "ok": True,
        "execution": {
            "execution_id": execution_id,
            "root_goal": execution.get("root_goal"),
            "status": execution.get("status"),
            "origin_checkpoint": execution.get("origin_checkpoint"),
            "continuity_score": execution.get("continuity_score"),
            "created_at": execution["created_at"].isoformat() if execution.get("created_at") else None,
            "last_active_at": execution["last_active_at"].isoformat() if execution.get("last_active_at") else None,
        },
        "tasks": task_nodes,
        "total_attempts": len(task_nodes),
        "current_task_id": current_task.id if current_task else None,
    }


def handle_end_execution(db: MemoryDB, graph: MemoryGraph,
                         execution_id: str, status: str = "completed",
                         user_id: str = "default") -> dict:
    """Mark an execution as completed or abandoned."""
    if not execution_id or not execution_id.strip():
        return _error("execution_id must be non-empty")
    valid_statuses = ("completed", "abandoned", "paused")
    if status not in valid_statuses:
        return _error(f"status must be one of {valid_statuses}")
    user_id = _validate_user_id(user_id)
    execution = db.get_execution(execution_id)
    if not execution:
        return _error(f"Execution {execution_id} not found")
    if execution.get("user_id") != user_id:
        return _error(f"Execution {execution_id} not found for user '{user_id}'")

    try:
        db.end_execution(execution_id, status=status)
    except DegradedModeError as exc:
        return _degraded_error(exc)

    return {
        "ok": True,
        "execution_id": execution_id,
        "status": status,
        "message": f"Execution {execution_id} marked as {status}.",
    }


# ---- v0.17: Runtime Reliability Signal handlers ----


def handle_detect_drift(db: MemoryDB, graph: MemoryGraph,
                        task_id: int,
                        current_goal: str = "",
                        tools_used: list[str] | None = None,
                        actions_taken: list[str] | None = None,
                        in_progress: list[str] | None = None,
                        violated_constraints: list[str] | None = None,
                        user_id: str = "default") -> dict:
    """Detect execution drift after a checkpoint restore or retry.

    Compares the Agent's current reported state against the checkpoint baseline
    to identify goal drift, tool drift, planning drift, and constraint drift.
    """
    from .drift import detect_drift

    user_id = _validate_user_id(user_id)
    task_id = int(task_id)

    current_state = {
        "goal": current_goal,
        "tools_used": tools_used or [],
        "actions_taken": actions_taken or [],
        "in_progress": in_progress or [],
        "violated_constraints": violated_constraints or [],
    }

    try:
        signal = detect_drift(db, task_id, current_state, user_id=user_id)
    except Exception as exc:
        return _error(f"Drift detection failed: {exc}")

    result = signal.to_dict()
    # Classify drift severity
    if signal.composite >= 0.7:
        result["severity"] = "critical"
        result["recommendation"] = "Execution has drifted significantly. Consider abandoning this retry and restoring from checkpoint."
    elif signal.composite >= 0.4:
        result["severity"] = "high"
        result["recommendation"] = "Notable drift detected. Re-read checkpoint constraints and re-align with original goal."
    elif signal.composite >= 0.2:
        result["severity"] = "medium"
        result["recommendation"] = "Minor drift detected. Stay aware of original constraints."
    else:
        result["severity"] = "low"
        result["recommendation"] = "Execution is on track."

    return {"ok": True, "task_id": task_id, "drift": result}


def handle_score_recovery(db: MemoryDB, graph: MemoryGraph,
                          task_id: int,
                          goal: str = "",
                          completed: list[str] | None = None,
                          in_progress: list[str] | None = None,
                          must_not_redo: list[str] | None = None,
                          active_constraints: list[str] | None = None,
                          tools_used: list[str] | None = None,
                          user_id: str = "default") -> dict:
    """Score the semantic continuity of a recovery.

    Called after a checkpoint restore to measure how well the Agent has
    re-oriented. Returns goal_alignment, constraint_retention,
    task_position_alignment, tool_behavior_stability, retry_degradation,
    and recovery_confidence.
    """
    from .reliability import score_recovery

    user_id = _validate_user_id(user_id)
    task_id = int(task_id)

    post_restore_state = {
        "goal": goal,
        "completed": completed or [],
        "in_progress": in_progress or [],
        "must_not_redo": must_not_redo or [],
        "active_constraints": active_constraints or [],
        "tools_used": tools_used or [],
    }

    try:
        score = score_recovery(db, task_id, post_restore_state, user_id=user_id)
    except Exception as exc:
        return _error(f"Recovery scoring failed: {exc}")

    result = score.to_dict()

    # Classify confidence level
    if score.recovery_confidence >= 0.8:
        result["confidence_level"] = "high"
        result["recommendation"] = "Recovery looks solid. Continue execution."
    elif score.recovery_confidence >= 0.5:
        result["confidence_level"] = "medium"
        result["recommendation"] = "Some continuity gaps. Re-read checkpoint state carefully."
    else:
        result["confidence_level"] = "low"
        result["recommendation"] = "Poor continuity. Consider full checkpoint restore or starting fresh."

    return {"ok": True, "task_id": task_id, "score": result}


def handle_recommend_recovery(db: MemoryDB, graph: MemoryGraph,
                              task_id: int,
                              interruption_reason: str | None = None,
                              retry_count: int = 0,
                              user_id: str = "default") -> dict:
    """Lightweight recovery heuristics — recommend a recovery action.

    NOT a policy engine. Just hardcoded heuristics based on:
    - interruption reason
    - retry depth
    - checkpoint freshness
    - execution state
    """
    from .db import (
        RECOVERY_STRATEGIES, INTERRUPTION_UNKNOWN,
        VALID_INTERRUPTION_REASONS,
    )
    from . import checkpoint as _ckpt

    user_id = _validate_user_id(user_id)
    task_id = int(task_id)

    # Get context
    task = db.get_task(task_id)
    if task is None:
        return _error(f"Task {task_id} not found")

    ckpt = _ckpt.get_checkpoint(db, task_id, user_id=user_id)
    retry_chain = db.get_retry_chain(task_id)
    retry_depth = len(retry_chain)

    # Heuristic 1: Too many retries → abandon
    if retry_depth >= 3:
        return {
            "ok": True,
            "task_id": task_id,
            "action": "abandon",
            "reason": f"Retry depth {retry_depth} exceeds threshold. Retrying further is unlikely to succeed.",
            "confidence": 0.8,
        }

    # Heuristic 2: No checkpoint → can only start fresh
    if ckpt is None:
        return {
            "ok": True,
            "task_id": task_id,
            "action": "start_fresh",
            "reason": "No checkpoint available. Must start execution from scratch.",
            "confidence": 0.6,
        }

    # Heuristic 3: Interruption-specific strategy
    reason = interruption_reason or INTERRUPTION_UNKNOWN
    if reason not in VALID_INTERRUPTION_REASONS:
        reason = INTERRUPTION_UNKNOWN

    strategy = RECOVERY_STRATEGIES.get(reason, RECOVERY_STRATEGIES[INTERRUPTION_UNKNOWN])

    # Heuristic 4: Checkpoint too old (> 2 hours) → start fresh
    ckpt_age_seconds = 0
    if ckpt.get("created_at"):
        from datetime import datetime, timezone
        try:
            created = ckpt["created_at"]
            if isinstance(created, str):
                created = datetime.fromisoformat(created.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            ckpt_age_seconds = (now - created).total_seconds()
        except Exception:
            pass

    if ckpt_age_seconds > 7200:  # > 2 hours old
        return {
            "ok": True,
            "task_id": task_id,
            "action": "start_fresh",
            "reason": f"Checkpoint is {ckpt_age_seconds/3600:.1f}h old. State may be too stale for reliable restore.",
            "confidence": 0.5,
            "alternative": strategy["action"],
        }

    # Heuristic 5: Repeated tool failure → try different approach
    if reason == "tool_failure" and retry_count > 1:
        return {
            "ok": True,
            "task_id": task_id,
            "action": "restore_checkpoint",
            "reason": "Repeated tool failures. Restore checkpoint and try alternative tools/approach.",
            "confidence": 0.7,
            "hint": "Consider using different tools or breaking the task into smaller steps.",
        }

    # Default: follow interruption-specific strategy
    return {
        "ok": True,
        "task_id": task_id,
        "action": strategy["action"],
        "reason": strategy["hint"],
        "confidence": 0.7,
        "memory_restore_mode": strategy.get("memory_restore_mode", "NONE"),
        "severity": strategy.get("severity", "medium"),
        "recoverability": strategy.get("recoverability", "low"),
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
    "restore_checkpoint": handle_restore_checkpoint,
    "list_checkpoints": handle_list_checkpoints,
    "get_runtime_health": handle_get_runtime_health,
    "report_interruption": handle_report_interruption,
    "evaluate_continuity": handle_evaluate_continuity,
    "start_execution": handle_start_execution,
    "retry_task": handle_retry_task,
    "spawn_subtask": handle_spawn_subtask,
    "trace_execution": handle_trace_execution,
    "end_execution": handle_end_execution,
    "detect_drift": handle_detect_drift,
    "score_recovery": handle_score_recovery,
    "recommend_recovery": handle_recommend_recovery,
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
    "restore_checkpoint": {"task_id": "task_id", "version": "version",
                           "memory_restore_mode": "memory_restore_mode",
                           "user_id": "user_id"},
    "list_checkpoints": {"task_id": "task_id", "limit": "limit", "user_id": "user_id"},
    "get_runtime_health": {},  # No input args; doctor reads everything from disk.
    "report_interruption": {"reason": "reason", "context": "context",
                            "session_id": "session_id", "user_id": "user_id"},
    "evaluate_continuity": {"task_id": "task_id", "before_version": "before_version",
                            "after_version": "after_version",
                            "actions_taken_after_restore": "actions_taken_after_restore",
                            "user_id": "user_id"},
    "start_execution": {"goal": "goal", "user_id": "user_id",
                        "origin_checkpoint": "origin_checkpoint"},
    "retry_task": {"task_id": "task_id", "reason": "reason", "user_id": "user_id"},
    "spawn_subtask": {"parent_task_id": "parent_task_id", "name": "name",
                      "goal": "goal", "user_id": "user_id",
                      "checkpoint_id": "checkpoint_id"},
    "trace_execution": {"execution_id": "execution_id", "user_id": "user_id"},
    "end_execution": {"execution_id": "execution_id", "status": "status",
                      "user_id": "user_id"},
    "detect_drift": {"task_id": "task_id", "current_goal": "current_goal",
                     "tools_used": "tools_used", "actions_taken": "actions_taken",
                     "in_progress": "in_progress",
                     "violated_constraints": "violated_constraints",
                     "user_id": "user_id"},
    "score_recovery": {"task_id": "task_id", "goal": "goal",
                       "completed": "completed", "in_progress": "in_progress",
                       "must_not_redo": "must_not_redo",
                       "active_constraints": "active_constraints",
                       "tools_used": "tools_used", "user_id": "user_id"},
    "recommend_recovery": {"task_id": "task_id",
                           "interruption_reason": "interruption_reason",
                           "retry_count": "retry_count", "user_id": "user_id"},
}
