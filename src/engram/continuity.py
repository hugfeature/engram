"""Continuity Metrics — 6-dimensional quality scoring for checkpoint restore.

Measures how well an Agent's cognitive state survives an interruption.
Used by Chaos Continuity Tests and exposed via `evaluate_continuity` MCP tool.

The six dimensions:
    1. Goal Retention        — Is the goal unchanged after restore?
    2. Action Consistency    — Are in_progress / preferred_next preserved?
    3. Failure Recall        — Are prior failures captured in must_not_redo?
    4. Working Set Stability — Are files/tools/artifacts unchanged?
    5. Replanning Rate       — How often did the plan change? (lower is better)
    6. Redundant Exploration — Did the Agent redo something in must_not_redo?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .checkpoint import (
    jaccard_similarity,
    _as_list,
    _as_dict,
    _extract_working_set_signature,
)

log = logging.getLogger("engram.continuity")

# Composite score weights (sum = 1.0)
WEIGHT_GOAL_RETENTION = 0.20
WEIGHT_ACTION_CONSISTENCY = 0.20
WEIGHT_FAILURE_RECALL = 0.20
WEIGHT_WORKING_SET_STABILITY = 0.15
WEIGHT_REPLANNING_RATE = 0.10
WEIGHT_REDUNDANT_EXPLORATION = 0.15


@dataclass
class ContinuityScore:
    """Six-dimensional continuity quality score."""

    goal_retention: float = 0.0
    action_consistency: float = 0.0
    failure_recall: float = 0.0
    working_set_stability: float = 0.0
    replanning_rate: float = 0.0
    redundant_exploration: float = 0.0
    composite: float = 0.0

    def to_dict(self) -> dict:
        return {
            "goal_retention": round(self.goal_retention, 4),
            "action_consistency": round(self.action_consistency, 4),
            "failure_recall": round(self.failure_recall, 4),
            "working_set_stability": round(self.working_set_stability, 4),
            "replanning_rate": round(self.replanning_rate, 4),
            "redundant_exploration": round(self.redundant_exploration, 4),
            "composite": round(self.composite, 4),
        }


def _compute_composite(score: ContinuityScore) -> float:
    """Weighted average of 6 dimensions."""
    return (
        WEIGHT_GOAL_RETENTION * score.goal_retention
        + WEIGHT_ACTION_CONSISTENCY * score.action_consistency
        + WEIGHT_FAILURE_RECALL * score.failure_recall
        + WEIGHT_WORKING_SET_STABILITY * score.working_set_stability
        + WEIGHT_REPLANNING_RATE * score.replanning_rate
        + WEIGHT_REDUNDANT_EXPLORATION * score.redundant_exploration
    )


# ============================================================
# Individual Metric Computations
# ============================================================

def goal_retention(before_state: dict, after_state: dict) -> float:
    """1.0 if goal is identical, 0.0 if completely different.

    Uses exact match — goals are typically short strings set once.
    """
    before_goal = (before_state.get("goal") or "").strip()
    after_goal = (after_state.get("goal") or "").strip()
    if not before_goal and not after_goal:
        return 1.0
    if not before_goal or not after_goal:
        return 0.0
    return 1.0 if before_goal == after_goal else 0.0


def action_consistency(before_state: dict, after_state: dict) -> float:
    """Jaccard similarity of in_progress + preferred_next sets.

    Higher means the Agent preserved the planned actions.
    """
    before_actions = set(_as_list(before_state.get("in_progress")))
    before_actions |= set(_as_list(before_state.get("preferred_next")))
    after_actions = set(_as_list(after_state.get("in_progress")))
    after_actions |= set(_as_list(after_state.get("preferred_next")))
    return jaccard_similarity(before_actions, after_actions)


def failure_recall(before_state: dict, after_state: dict) -> float:
    """Coverage of before must_not_redo in after must_not_redo.

    1.0 = all prior negative memories preserved; 0.0 = all lost.
    """
    before_items = _as_list(before_state.get("must_not_redo"))
    after_items = _as_list(after_state.get("must_not_redo"))
    if not before_items:
        return 1.0
    before_actions = {
        item.get("action", "") for item in before_items
        if isinstance(item, dict) and item.get("action")
    }
    if not before_actions:
        return 1.0
    after_actions = {
        item.get("action", "") for item in after_items
        if isinstance(item, dict) and item.get("action")
    }
    retained = before_actions & after_actions
    return len(retained) / len(before_actions)


def working_set_stability(before_state: dict, after_state: dict) -> float:
    """Jaccard similarity of working_set signatures (files + tools + artifacts)."""
    before_sig = _extract_working_set_signature(_as_dict(before_state.get("working_set")))
    after_sig = _extract_working_set_signature(_as_dict(after_state.get("working_set")))
    return jaccard_similarity(before_sig, after_sig)


def replanning_rate(db, task_id: int, user_id: str = "default") -> float:
    """Fraction of checkpoints that were NOT triggered by plan changes.

    1.0 = no replanning at all (perfect stability).
    0.0 = every checkpoint was a plan change.
    """
    rows = db.conn.execute(
        """SELECT checkpoint_reason FROM checkpoints
           WHERE task_id = ? AND user_id = ?
           ORDER BY version DESC LIMIT 20""",
        [task_id, user_id],
    ).fetchall()
    if not rows:
        return 1.0
    plan_change_reasons = {"PLAN_UPDATE", "WORKING_SET_SHIFT"}
    replan_count = sum(1 for r in rows if r[0] in plan_change_reasons)
    return 1.0 - (replan_count / len(rows))


def redundant_exploration(
    before_must_not_redo: list[dict],
    actions_taken_after_restore: list[str],
) -> float:
    """Fraction of post-restore actions that were NOT in must_not_redo.

    1.0 = zero redundancy (Agent respected all negative memories).
    0.0 = Agent re-did everything it was told not to.
    """
    if not actions_taken_after_restore:
        return 1.0
    forbidden_actions = {
        item.get("action", "").lower() for item in before_must_not_redo
        if isinstance(item, dict) and item.get("action")
    }
    if not forbidden_actions:
        return 1.0
    violations = sum(
        1 for action in actions_taken_after_restore
        if action.lower() in forbidden_actions
    )
    return 1.0 - (violations / len(actions_taken_after_restore))


# ============================================================
# Public API
# ============================================================

def evaluate(
    before_state: dict,
    after_state: dict,
    db=None,
    task_id: int | None = None,
    user_id: str = "default",
    actions_taken_after_restore: list[str] | None = None,
) -> ContinuityScore:
    """Compute all 6 continuity metrics.

    Args:
        before_state: Checkpoint state at interruption time.
        after_state: Checkpoint state after restore / new Agent takeover.
        db: MemoryDB instance (needed for replanning_rate).
        task_id: Task ID (needed for replanning_rate).
        user_id: User ID.
        actions_taken_after_restore: List of action descriptions the new
            Agent performed after restoring (needed for redundant_exploration).

    Returns:
        ContinuityScore with all 6 dimensions + composite.
    """
    score = ContinuityScore(
        goal_retention=goal_retention(before_state, after_state),
        action_consistency=action_consistency(before_state, after_state),
        failure_recall=failure_recall(before_state, after_state),
        working_set_stability=working_set_stability(before_state, after_state),
    )

    if db is not None and task_id is not None:
        score.replanning_rate = replanning_rate(db, task_id, user_id)
    else:
        score.replanning_rate = 1.0

    before_must_not = _as_list(before_state.get("must_not_redo"))
    score.redundant_exploration = redundant_exploration(
        before_must_not,
        actions_taken_after_restore or [],
    )

    score.composite = _compute_composite(score)
    return score


def evaluate_from_checkpoints(
    db,
    task_id: int,
    before_version: int | None = None,
    after_version: int | None = None,
    user_id: str = "default",
    actions_taken_after_restore: list[str] | None = None,
) -> ContinuityScore | None:
    """Evaluate continuity between two checkpoint versions of the same task.

    Convenience wrapper that loads checkpoints from DB. If before_version
    is None, uses the second-to-last checkpoint. If after_version is None,
    uses the latest.
    """
    from .checkpoint import get_checkpoint, list_checkpoints

    if before_version is None or after_version is None:
        history = list_checkpoints(db, task_id, limit=2, user_id=user_id)
        if len(history) < 2:
            log.info("Need at least 2 checkpoints to evaluate continuity")
            return None
        if after_version is None:
            after_version = history[0]["version"]
        if before_version is None:
            before_version = history[1]["version"]

    before_ckpt = get_checkpoint(db, task_id, version=before_version, user_id=user_id)
    after_ckpt = get_checkpoint(db, task_id, version=after_version, user_id=user_id)
    if not before_ckpt or not after_ckpt:
        return None

    return evaluate(
        before_state=before_ckpt["state"],
        after_state=after_ckpt["state"],
        db=db,
        task_id=task_id,
        user_id=user_id,
        actions_taken_after_restore=actions_taken_after_restore,
    )
