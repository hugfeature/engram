"""Semantic Continuity Scoring — v0.17 Runtime Reliability Signals.

Measures how well an Agent retains semantic continuity after a checkpoint
restore. Unlike simple retry metrics (time, count), this module answers:

    "Does the Agent still remember what it was doing, what it finished,
     what it must not redo, and where it is in the execution?"

Four scoring dimensions:
    1. goal_alignment       — Does current goal match checkpoint goal?
    2. constraint_retention — Are must_not_redo / active_constraints preserved?
    3. task_position_alignment — Is the Agent at the correct execution position?
    4. tool_behavior_stability — Are tool usage patterns stable post-restore?

Also produces:
    - retry_degradation: How much quality drops with each successive retry
    - recovery_confidence: Overall confidence that this recovery is sound
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .db import MemoryDB

log = logging.getLogger("engram.reliability")


@dataclass
class SemanticContinuityScore:
    """Multi-dimensional semantic continuity measurement."""

    goal_alignment: float = 1.0           # 1.0 = perfectly aligned
    constraint_retention: float = 1.0     # 1.0 = all constraints remembered
    task_position_alignment: float = 1.0  # 1.0 = correct position
    tool_behavior_stability: float = 1.0  # 1.0 = stable tool usage
    retry_degradation: float = 0.0        # 0.0 = no degradation, 1.0 = fully degraded
    recovery_confidence: float = 1.0      # 0.0 = no confidence, 1.0 = full confidence
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = {
            "goal_alignment": round(self.goal_alignment, 4),
            "constraint_retention": round(self.constraint_retention, 4),
            "task_position_alignment": round(self.task_position_alignment, 4),
            "tool_behavior_stability": round(self.tool_behavior_stability, 4),
            "retry_degradation": round(self.retry_degradation, 4),
            "recovery_confidence": round(self.recovery_confidence, 4),
        }
        if self.details:
            result["details"] = self.details
        return result


def score_recovery(
    db: "MemoryDB",
    task_id: int,
    post_restore_state: dict,
    user_id: str = "default",
) -> SemanticContinuityScore:
    """Score the semantic continuity of a recovery.

    Called after a checkpoint restore to measure how well the Agent
    has re-oriented. The post_restore_state should contain:
        - goal: Agent's current stated goal
        - completed: list of items Agent reports as completed
        - in_progress: list of items Agent is currently working on
        - must_not_redo: items Agent acknowledges it must not redo
        - tools_used: tools used since restore
        - execution_position: Agent's reported position (attempt, step)
    """
    from . import checkpoint as _ckpt

    score = SemanticContinuityScore()

    # Get checkpoint baseline
    ckpt = _ckpt.get_checkpoint(db, task_id, user_id=user_id)
    if ckpt is None:
        score.details["no_checkpoint"] = True
        return score

    ckpt_state = ckpt["state"]

    # 1. Goal Alignment
    score.goal_alignment = _score_goal_alignment(
        ckpt_state.get("goal", ""),
        post_restore_state.get("goal", ""),
    )

    # 2. Constraint Retention
    score.constraint_retention = _score_constraint_retention(
        ckpt_state, post_restore_state,
    )

    # 3. Task Position Alignment
    score.task_position_alignment = _score_position_alignment(
        db, task_id, ckpt_state, post_restore_state,
    )

    # 4. Tool Behavior Stability
    score.tool_behavior_stability = _score_tool_stability(
        ckpt_state, post_restore_state,
    )

    # Retry Degradation — based on retry chain depth
    score.retry_degradation = _compute_retry_degradation(db, task_id)

    # Recovery Confidence — weighted composite
    score.recovery_confidence = _compute_confidence(score)

    # Tier 3 persistence: record continuity signal for analytical queries
    try:
        db.record_signal(
            task_id=task_id,
            signal_type="continuity",
            dimensions=score.to_dict(),
            composite_score=score.recovery_confidence,
            user_id=user_id,
        )
    except Exception as exc:
        log.debug("continuity signal persistence failed (non-fatal): %s", exc)

    return score


def _score_goal_alignment(checkpoint_goal: str, current_goal: str) -> float:
    """How well does the current goal match the checkpoint goal?"""
    if not checkpoint_goal and not current_goal:
        return 1.0
    if not checkpoint_goal or not current_goal:
        return 0.0

    ckpt_words = set(checkpoint_goal.lower().split())
    curr_words = set(current_goal.lower().split())

    if not ckpt_words:
        return 1.0

    # What fraction of checkpoint goal words are preserved?
    preserved = ckpt_words & curr_words
    recall = len(preserved) / len(ckpt_words)
    return round(recall, 4)


def _score_constraint_retention(checkpoint_state: dict, post_state: dict) -> float:
    """How many constraints does the Agent still remember?"""
    ckpt_must_not = set()
    for item in (checkpoint_state.get("must_not_redo") or []):
        if isinstance(item, dict):
            ckpt_must_not.add(item.get("action", str(item)))
        else:
            ckpt_must_not.add(str(item))

    ckpt_constraints = set(checkpoint_state.get("active_constraints") or [])
    total_constraints = ckpt_must_not | ckpt_constraints

    if not total_constraints:
        return 1.0  # Nothing to retain

    # Check what the Agent reports it remembers
    agent_must_not = set()
    for item in (post_state.get("must_not_redo") or []):
        if isinstance(item, dict):
            agent_must_not.add(item.get("action", str(item)))
        else:
            agent_must_not.add(str(item))

    agent_constraints = set(post_state.get("active_constraints") or [])
    agent_remembers = agent_must_not | agent_constraints

    retained = total_constraints & agent_remembers
    return round(len(retained) / len(total_constraints), 4)


def _score_position_alignment(
    db: "MemoryDB",
    task_id: int,
    checkpoint_state: dict,
    post_state: dict,
) -> float:
    """Is the Agent at the correct execution position?"""
    # Check completed items alignment
    ckpt_completed = set(checkpoint_state.get("completed") or [])
    agent_completed = set(post_state.get("completed") or [])

    if not ckpt_completed:
        return 1.0  # Nothing was completed, position is trivially correct

    # Agent should report at least the same completed items
    recognized = ckpt_completed & agent_completed
    recall = len(recognized) / len(ckpt_completed)

    # Also check if Agent is repeating completed work (in_progress overlap)
    agent_in_progress = set(post_state.get("in_progress") or [])
    redoing_completed = ckpt_completed & agent_in_progress
    if redoing_completed:
        # Penalty for re-doing completed work
        penalty = len(redoing_completed) / len(ckpt_completed)
        recall = max(0.0, recall - penalty * 0.5)

    return round(recall, 4)


def _score_tool_stability(checkpoint_state: dict, post_state: dict) -> float:
    """Are tool usage patterns stable after restore?"""
    baseline_tools = set(
        (checkpoint_state.get("working_set") or {}).get("tools", []) or []
    )
    current_tools = set(post_state.get("tools_used") or [])

    if not baseline_tools:
        return 1.0  # No baseline
    if not current_tools:
        return 1.0  # Haven't used tools yet — stable by default

    # Overlap ratio
    overlap = baseline_tools & current_tools
    if not current_tools:
        return 1.0
    stability = len(overlap) / len(current_tools)
    return round(stability, 4)


def _compute_retry_degradation(db: "MemoryDB", task_id: int) -> float:
    """How much has quality degraded across retries?

    Each retry reduces confidence. Formula:
        degradation = 1 - (0.8 ^ retry_depth)
    So: 0 retries = 0.0, 1 retry = 0.2, 2 retries = 0.36, 3 = 0.49, etc.
    """
    retry_chain = db.get_retry_chain(task_id)
    depth = len(retry_chain)
    if depth == 0:
        return 0.0
    return round(1.0 - (0.8 ** depth), 4)


def _compute_confidence(score: SemanticContinuityScore) -> float:
    """Overall recovery confidence as weighted average of signals."""
    raw = (
        0.30 * score.goal_alignment
        + 0.30 * score.constraint_retention
        + 0.20 * score.task_position_alignment
        + 0.20 * score.tool_behavior_stability
    )
    # Retry degradation reduces confidence
    confidence = raw * (1.0 - score.retry_degradation * 0.5)
    return round(max(0.0, min(1.0, confidence)), 4)
