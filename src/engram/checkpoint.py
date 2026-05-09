"""Cognitive Continuation — Checkpoint v2.

Engram 恢复的是 agent 的 cognition，不是 machine 的 execution。
Checkpoint 是某个 task 在某个时刻的认知状态快照，支持 constrained continuation：
不强求 LLM 选同一个 action（物理限制做不到），而是给它一组约束收窄行动空间。

设计取舍：
- semantic diff > structural diff：浅 diff 关心"目标变了没/working_set 漂了没"，
  不追求 RFC 6902 JSON Patch 的 replay correctness（probabilistic cognitive state 无法 replay）
- event-first, time-second：按认知事件触发（FAILURE / PLAN_UPDATE / WORKING_SET_SHIFT），
  时间只作为 5min 兜底
- must_not_redo 是 negative memory，结构化对象列表
- 跨 task checkpoint 禁止：MVP 强制线性 continuity
"""

from __future__ import annotations

import logging
from typing import Any, Optional

log = logging.getLogger("engram.checkpoint")


# ============================================================
# 枚举常量
# ============================================================

# checkpoint_reason 6 值枚举
REASON_MANUAL_HANDOFF = "MANUAL_HANDOFF"        # Agent 调用 session_handoff
REASON_PLAN_UPDATE = "PLAN_UPDATE"              # in_progress 列表显著变化
REASON_FAILURE = "FAILURE"                      # track_failure 触发
REASON_WORKING_SET_SHIFT = "WORKING_SET_SHIFT"  # 工作集 Jaccard < 0.5
REASON_TOOL_FINISHED = "TOOL_FINISHED"          # 预留，第 4 层接入后启用
REASON_AUTO_SAVE = "AUTO_SAVE"                  # 5min 兜底

VALID_REASONS = {
    REASON_MANUAL_HANDOFF, REASON_PLAN_UPDATE, REASON_FAILURE,
    REASON_WORKING_SET_SHIFT, REASON_TOOL_FINISHED, REASON_AUTO_SAVE,
}

# kind 枚举
KIND_HANDOFF = "handoff"   # Agent 主动产生（MANUAL_HANDOFF 专用）
KIND_AUTO = "auto"         # 自动产生（其他 reason）

# must_not_redo[].reason 6 值枚举
NEGATIVE_REASON_ALREADY_COMPLETED = "already_completed"
NEGATIVE_REASON_SIDE_EFFECT_EMITTED = "side_effect_emitted"
NEGATIVE_REASON_USER_FORBIDDEN = "user_forbidden"
NEGATIVE_REASON_TOOL_UNSAFE = "tool_unsafe"
NEGATIVE_REASON_SUPERSEDED = "superseded"
NEGATIVE_REASON_FAILED_DONT_RETRY = "failed_dont_retry"

VALID_NEGATIVE_REASONS = {
    NEGATIVE_REASON_ALREADY_COMPLETED, NEGATIVE_REASON_SIDE_EFFECT_EMITTED,
    NEGATIVE_REASON_USER_FORBIDDEN, NEGATIVE_REASON_TOOL_UNSAFE,
    NEGATIVE_REASON_SUPERSEDED, NEGATIVE_REASON_FAILED_DONT_RETRY,
}

# memory_restore_mode
RESTORE_MODE_FULL = "FULL"
RESTORE_MODE_SELECTIVE = "SELECTIVE"   # 默认
RESTORE_MODE_NONE = "NONE"
VALID_RESTORE_MODES = {RESTORE_MODE_FULL, RESTORE_MODE_SELECTIVE, RESTORE_MODE_NONE}

# 触发参数（v1.1 终版）
DEBOUNCE_SAME_REASON_SECONDS = 60        # 同 reason 60s 内最多 1 个 auto checkpoint
AUTO_SAVE_FALLBACK_SECONDS = 300         # 5 分钟无 checkpoint 强制 AUTO_SAVE
PLAN_CHANGED_JACCARD_THRESHOLD = 0.7     # in_progress Jaccard < 0.7 视为 plan 变化
WORKING_SET_SHIFT_JACCARD_THRESHOLD = 0.5  # working_set 集合 Jaccard < 0.5 视为漂移

# continuation_confidence 权重（v1.1 终版）
CONFIDENCE_WEIGHT_STATE_COMPLETENESS = 0.25
CONFIDENCE_WEIGHT_RECENCY = 0.25
CONFIDENCE_WEIGHT_VERIFICATION_HISTORY = 0.30
CONFIDENCE_WEIGHT_DRIFT_SIGNALS = 0.20
RECENCY_HALF_LIFE_HOURS = 24


# ============================================================
# Helpers
# ============================================================

def jaccard_similarity(a: set, b: set) -> float:
    """两个集合的 Jaccard 相似度。均空 → 1.0；其一空 → 0.0。"""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 1.0


def derive_failure_signature(component: str, error_class: str) -> str:
    """failure_signature = component:error_class 的稳定指纹。"""
    comp = (component or "unknown").strip().lower()
    err = (error_class or "unknown").strip().lower()
    return f"{comp}:{err}"


def normalize_must_not_redo(items: Any) -> list[dict]:
    """规范化 must_not_redo 列表。

    宽松策略：未知 reason → 'already_completed'；缺失 action → 跳过该项。
    """
    if not items or not isinstance(items, list):
        return []
    normalized: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        action = item.get("action")
        if not action or not isinstance(action, str):
            continue
        reason = item.get("reason", NEGATIVE_REASON_ALREADY_COMPLETED)
        if reason not in VALID_NEGATIVE_REASONS:
            reason = NEGATIVE_REASON_ALREADY_COMPLETED
        normalized.append({
            "action": action,
            "reason": reason,
            "scope": item.get("scope", "current_task"),
            "expires_at": item.get("expires_at"),
            "idempotency_key": item.get("idempotency_key"),
        })
    return normalized


def _as_list(value: Any) -> list:
    """兼容 DuckDB JSON 返回：可能是 list / JSON 字符串 / None。"""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        import json as _json
        try:
            v = _json.loads(value)
            return v if isinstance(v, list) else []
        except (ValueError, TypeError):
            return []
    return []


def _as_dict(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        import json as _json
        try:
            v = _json.loads(value)
            return v if isinstance(v, dict) else {}
        except (ValueError, TypeError):
            return {}
    return {}


def _extract_working_set_signature(working_set: dict) -> set:
    """把 working_set 折叠成单个集合用于 Jaccard 比较。"""
    items: set = set()
    for key in ("files", "tools", "artifacts"):
        for v in working_set.get(key, []) or []:
            if isinstance(v, str):
                items.add(f"{key}:{v}")
    return items


def _now_utc():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)


def _to_utc(dt):
    """DuckDB 返回的 timestamp 可能是 naive，统一为 aware UTC。"""
    from datetime import timezone
    if dt is None:
        return _now_utc()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ============================================================
# 事件检测
# ============================================================

def detect_plan_changed(prev_state: Optional[dict], new_state: dict) -> bool:
    """in_progress 列表 Jaccard < PLAN_CHANGED_JACCARD_THRESHOLD 视为显著变化。

    首个 checkpoint（prev_state=None）且 new_state 有 in_progress 时也视为变化。
    """
    new_set = {x for x in _as_list(new_state.get("in_progress")) if isinstance(x, str)}
    if prev_state is None:
        return bool(new_set)
    prev_set = {x for x in _as_list(prev_state.get("in_progress")) if isinstance(x, str)}
    return jaccard_similarity(prev_set, new_set) < PLAN_CHANGED_JACCARD_THRESHOLD


def detect_working_set_shifted(prev_state: Optional[dict], new_state: dict) -> bool:
    """working_set 集合 Jaccard < WORKING_SET_SHIFT_JACCARD_THRESHOLD 视为漂移。"""
    new_sig = _extract_working_set_signature(_as_dict(new_state.get("working_set")))
    if prev_state is None:
        return bool(new_sig)
    prev_sig = _extract_working_set_signature(_as_dict(prev_state.get("working_set")))
    return jaccard_similarity(prev_sig, new_sig) < WORKING_SET_SHIFT_JACCARD_THRESHOLD


def _is_debounced(db, task_id: int, reason: str, user_id: str) -> bool:
    """同 reason 60s 内是否已有 checkpoint（FAILURE 不受 debounce 限制）。"""
    if reason == REASON_FAILURE:
        return False
    row = db.conn.execute(
        """
        SELECT created_at FROM checkpoints
        WHERE task_id = ? AND user_id = ? AND checkpoint_reason = ?
        ORDER BY version DESC LIMIT 1
        """,
        [task_id, user_id, reason],
    ).fetchone()
    if not row:
        return False
    age = (_now_utc() - _to_utc(row[0])).total_seconds()
    return age < DEBOUNCE_SAME_REASON_SECONDS


def _get_last_checkpoint_ts(db, task_id: int, user_id: str):
    row = db.conn.execute(
        """
        SELECT created_at FROM checkpoints
        WHERE task_id = ? AND user_id = ?
        ORDER BY version DESC LIMIT 1
        """,
        [task_id, user_id],
    ).fetchone()
    return row[0] if row else None


def should_create_auto_checkpoint(
    db,
    task_id: int,
    event_type: str,
    payload_state: dict,
    user_id: str = "default",
) -> tuple[bool, Optional[str]]:
    """Event-first 触发判定。

    优先级：
    1. event_type='failure' → (True, FAILURE)，不受 debounce
    2. event_type='progress' → 检测 plan_changed / working_set_shifted（受 debounce）
    3. >5min 无 checkpoint → (True, AUTO_SAVE)（受 debounce）
    4. 其他 → (False, None)
    """
    if event_type == "failure":
        return True, REASON_FAILURE

    prev = get_checkpoint(db, task_id, user_id=user_id)
    prev_state = prev.get("state") if prev else None

    if event_type == "progress":
        candidate: Optional[str] = None
        if detect_plan_changed(prev_state, payload_state):
            candidate = REASON_PLAN_UPDATE
        elif detect_working_set_shifted(prev_state, payload_state):
            candidate = REASON_WORKING_SET_SHIFT
        if candidate and not _is_debounced(db, task_id, candidate, user_id):
            return True, candidate

    last_ts = _get_last_checkpoint_ts(db, task_id, user_id)
    if last_ts is None:
        return True, REASON_AUTO_SAVE
    age = (_now_utc() - _to_utc(last_ts)).total_seconds()
    if age > AUTO_SAVE_FALLBACK_SECONDS and not _is_debounced(
        db, task_id, REASON_AUTO_SAVE, user_id
    ):
        return True, REASON_AUTO_SAVE

    return False, None


# ============================================================
# Diff / Confidence
# ============================================================

_SEMANTIC_FIELDS = (
    "goal", "completed", "in_progress", "blocked",
    "preferred_next", "must_not_redo", "must_preserve", "working_set",
)


def compute_shallow_diff(old_state: dict, new_state: dict) -> dict:
    """计算 semantic 字段的浅 diff。

    Returns:
        {"changed_fields": {field: {"old": ..., "new": ...}}}
    """
    if old_state is None:
        old_state = {}
    changed: dict = {}
    for f in _SEMANTIC_FIELDS:
        old_v = old_state.get(f)
        new_v = new_state.get(f)
        if old_v != new_v:
            changed[f] = {"old": old_v, "new": new_v}
    return {"changed_fields": changed}


def compute_confidence(
    db,
    task_id: int,
    state: dict,
    user_id: str = "default",
) -> tuple[float, dict]:
    """计算 continuation_confidence 与可解释 breakdown（v1.1 公式）。

    入库时 recency=1.0；恢复时由 build_continuation 用 recency_at 重算。
    """
    core = ("goal", "completed", "in_progress", "preferred_next", "must_not_redo")
    filled = sum(1 for f in core if state.get(f))
    state_completeness = filled / len(core)

    recency = 1.0  # 新建即"now"

    # verification_history：暂无 handoff_verifications 表，给中性默认值（下一阶段升级）
    verification_history = 0.7

    rows = db.conn.execute(
        """
        SELECT checkpoint_reason FROM checkpoints
        WHERE task_id = ? AND user_id = ?
        ORDER BY version DESC LIMIT 5
        """,
        [task_id, user_id],
    ).fetchall()
    if rows:
        drift_count = sum(
            1 for r in rows if r[0] in (REASON_PLAN_UPDATE, REASON_WORKING_SET_SHIFT)
        )
        drift_signals = max(0.0, 1.0 - (drift_count / len(rows)) * 0.5)
    else:
        drift_signals = 1.0

    confidence = (
        CONFIDENCE_WEIGHT_STATE_COMPLETENESS * state_completeness
        + CONFIDENCE_WEIGHT_RECENCY * recency
        + CONFIDENCE_WEIGHT_VERIFICATION_HISTORY * verification_history
        + CONFIDENCE_WEIGHT_DRIFT_SIGNALS * drift_signals
    )
    breakdown = {
        "state_completeness": round(state_completeness, 3),
        "recency": round(recency, 3),
        "verification_history": round(verification_history, 3),
        "drift_signals": round(drift_signals, 3),
    }
    return round(confidence, 3), breakdown


def recency_at(checkpoint_created_at) -> float:
    """根据 checkpoint 创建时间计算 recency（半衰期 RECENCY_HALF_LIFE_HOURS）。"""
    import math
    if checkpoint_created_at is None:
        return 0.0
    age_hours = (_now_utc() - _to_utc(checkpoint_created_at)).total_seconds() / 3600
    if age_hours <= 0:
        return 1.0
    decay = math.log(2) / RECENCY_HALF_LIFE_HOURS
    return round(math.exp(-decay * age_hours), 3)


# ============================================================
# 核心 API：create / get / list / build_continuation
# ============================================================

def create_checkpoint(
    db,
    task_id: int,
    reason: str,
    state: dict,
    source_session_id: Optional[str] = None,
    source_memory_id: Optional[int] = None,
    user_id: str = "default",
    failure_signature: Optional[str] = None,
    triggered_by_event: Optional[str] = None,
) -> dict:
    """创建一个 checkpoint。

    Returns:
        {checkpoint_id, version, reason, kind,
         continuation_confidence, confidence_breakdown}
    """
    import json as _json

    if reason not in VALID_REASONS:
        raise ValueError(f"invalid checkpoint_reason: {reason}")

    kind = KIND_HANDOFF if reason == REASON_MANUAL_HANDOFF else KIND_AUTO

    state = dict(state)
    state["must_not_redo"] = normalize_must_not_redo(state.get("must_not_redo"))

    confidence, breakdown = compute_confidence(db, task_id, state, user_id=user_id)

    prev = get_checkpoint(db, task_id, user_id=user_id)
    prev_state = prev.get("state") if prev else None
    parent_version = prev["version"] if prev else None
    state_diff = compute_shallow_diff(prev_state or {}, state)

    db.conn.execute("BEGIN")
    try:
        max_row = db.conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM checkpoints WHERE task_id = ? AND user_id = ?",
            [task_id, user_id],
        ).fetchone()
        new_version = (max_row[0] or 0) + 1

        ckpt_row = db.conn.execute(
            """
            INSERT INTO checkpoints (
                task_id, version, parent_version, kind,
                checkpoint_reason, triggered_by_event,
                goal, completed, in_progress, blocked, preferred_next,
                must_not_redo, must_preserve, working_set,
                state_diff, source_session_id, source_memory_id,
                continuation_confidence, confidence_breakdown,
                failure_signature, user_id
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?,
                ?, ?::JSON, ?::JSON, ?::JSON, ?::JSON,
                ?::JSON, ?::JSON, ?::JSON,
                ?::JSON, ?, ?,
                ?, ?::JSON,
                ?, ?
            )
            RETURNING id
            """,
            [
                task_id, new_version, parent_version, kind,
                reason, triggered_by_event,
                state.get("goal", "") or "",
                _json.dumps(_as_list(state.get("completed")), ensure_ascii=False),
                _json.dumps(_as_list(state.get("in_progress")), ensure_ascii=False),
                _json.dumps(_as_list(state.get("blocked")), ensure_ascii=False),
                _json.dumps(_as_list(state.get("preferred_next")), ensure_ascii=False),
                _json.dumps(state["must_not_redo"], ensure_ascii=False),
                _json.dumps(_as_list(state.get("must_preserve")), ensure_ascii=False),
                _json.dumps(_as_dict(state.get("working_set")), ensure_ascii=False),
                _json.dumps(state_diff, ensure_ascii=False),
                source_session_id, source_memory_id,
                confidence,
                _json.dumps(breakdown, ensure_ascii=False),
                failure_signature, user_id,
            ],
        ).fetchone()
        checkpoint_id = ckpt_row[0]

        db.conn.execute(
            """
            UPDATE tasks
            SET latest_checkpoint_version = ?,
                checkpoint_count = checkpoint_count + 1
            WHERE id = ?
            """,
            [new_version, task_id],
        )
        db.conn.execute("COMMIT")
    except Exception:
        db.conn.execute("ROLLBACK")
        raise

    # Tier 1 — checkpoints are the cognitive backbone of continuity, MUST
    # be in the event log so a destroyed DuckDB can be replayed back.
    # We log the full state payload (no embeddings) so replay is total.
    try:
        db._emit_event("checkpoint.write", {
            "checkpoint_id": checkpoint_id,
            "task_id": task_id,
            "version": new_version,
            "parent_version": parent_version,
            "kind": kind,
            "checkpoint_reason": reason,
            "triggered_by_event": triggered_by_event,
            "user_id": user_id,
            "state": {
                "goal": state.get("goal", "") or "",
                "completed": _as_list(state.get("completed")),
                "in_progress": _as_list(state.get("in_progress")),
                "blocked": _as_list(state.get("blocked")),
                "preferred_next": _as_list(state.get("preferred_next")),
                "must_not_redo": state["must_not_redo"],
                "must_preserve": _as_list(state.get("must_preserve")),
                "working_set": _as_dict(state.get("working_set")),
            },
            "state_diff": state_diff,
            "source_session_id": source_session_id,
            "source_memory_id": source_memory_id,
            "continuation_confidence": confidence,
            "confidence_breakdown": breakdown,
            "failure_signature": failure_signature,
        })
    except Exception as exc:
        log.error("checkpoint.write event log append FAILED: %s", exc)
        # Re-raise so the runtime contract is honoured: a checkpoint that
        # isn't in the event log isn't considered durably persisted.
        raise

    return {
        "checkpoint_id": checkpoint_id,
        "version": new_version,
        "reason": reason,
        "kind": kind,
        "continuation_confidence": confidence,
        "confidence_breakdown": breakdown,
    }


def get_checkpoint(
    db,
    task_id: int,
    version: Optional[int] = None,
    user_id: str = "default",
) -> Optional[dict]:
    """获取 task 的某个 checkpoint。version=None → 最新。"""
    if version is None:
        sql = """
            SELECT * FROM checkpoints
            WHERE task_id = ? AND user_id = ?
            ORDER BY version DESC LIMIT 1
        """
        params = [task_id, user_id]
    else:
        sql = """
            SELECT * FROM checkpoints
            WHERE task_id = ? AND user_id = ? AND version = ?
            LIMIT 1
        """
        params = [task_id, user_id, version]
    row = db._fetchone_dict(sql, params)
    if not row:
        return None
    return _row_to_checkpoint(row)


def list_checkpoints(
    db,
    task_id: int,
    limit: int = 10,
    user_id: str = "default",
) -> list[dict]:
    """返回 task 的 checkpoint 历史（version DESC），不含完整 state。"""
    rows = db._fetchall_dicts(
        """
        SELECT id, version, parent_version, kind, checkpoint_reason,
               triggered_by_event, source_session_id, source_memory_id,
               continuation_confidence, failure_signature, created_at
        FROM checkpoints
        WHERE task_id = ? AND user_id = ?
        ORDER BY version DESC LIMIT ?
        """,
        [task_id, user_id, limit],
    )
    return [
        {
            "checkpoint_id": r["id"],
            "version": r["version"],
            "parent_version": r["parent_version"],
            "kind": r["kind"],
            "reason": r["checkpoint_reason"],
            "triggered_by_event": r["triggered_by_event"],
            "source_session_id": r["source_session_id"],
            "source_memory_id": r["source_memory_id"],
            "continuation_confidence": r["continuation_confidence"],
            "failure_signature": r["failure_signature"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        }
        for r in rows
    ]


def _row_to_checkpoint(row: dict) -> dict:
    """DB row → 完整 checkpoint dict。"""
    return {
        "id": row["id"],
        "task_id": row["task_id"],
        "version": row["version"],
        "parent_version": row["parent_version"],
        "kind": row["kind"],
        "reason": row["checkpoint_reason"],
        "triggered_by_event": row["triggered_by_event"],
        "state": {
            "goal": row.get("goal") or "",
            "completed": _as_list(row.get("completed")),
            "in_progress": _as_list(row.get("in_progress")),
            "blocked": _as_list(row.get("blocked")),
            "preferred_next": _as_list(row.get("preferred_next")),
            "must_not_redo": _as_list(row.get("must_not_redo")),
            "must_preserve": _as_list(row.get("must_preserve")),
            "working_set": _as_dict(row.get("working_set")),
        },
        "state_diff": _as_dict(row.get("state_diff")),
        "source_session_id": row.get("source_session_id"),
        "source_memory_id": row.get("source_memory_id"),
        "continuation_confidence": row.get("continuation_confidence"),
        "confidence_breakdown": _as_dict(row.get("confidence_breakdown")),
        "failure_signature": row.get("failure_signature"),
        "created_at": row["created_at"],
    }


def build_continuation(checkpoint: dict) -> dict:
    """把 get_checkpoint 的结果转成 constrained continuation 包。

    会基于 created_at 重算 recency 并校准 continuation_confidence。
    """
    state = checkpoint["state"]
    breakdown = dict(checkpoint.get("confidence_breakdown") or {})
    new_recency = recency_at(checkpoint.get("created_at"))
    breakdown["recency"] = new_recency

    confidence = round(
        CONFIDENCE_WEIGHT_STATE_COMPLETENESS * breakdown.get("state_completeness", 0.0)
        + CONFIDENCE_WEIGHT_RECENCY * new_recency
        + CONFIDENCE_WEIGHT_VERIFICATION_HISTORY * breakdown.get("verification_history", 0.7)
        + CONFIDENCE_WEIGHT_DRIFT_SIGNALS * breakdown.get("drift_signals", 1.0),
        3,
    )

    return {
        "goal": state["goal"],
        "completed": state["completed"],
        "in_progress": state["in_progress"],
        "blocked": state["blocked"],
        "preferred_next": state["preferred_next"],
        "must_not_redo": state["must_not_redo"],
        "must_preserve": state["must_preserve"],
        "working_set": state["working_set"],
        "continuation_confidence": confidence,
        "confidence_breakdown": breakdown,
    }
