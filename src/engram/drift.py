"""Execution Drift Analysis — v0.17 Runtime Reliability Signals.

Detects when an Agent's execution drifts from its intended trajectory
after a checkpoint restore or retry. Unlike traditional workflow systems,
LLM runtimes exhibit non-deterministic drift: reasoning wanders, goals
shift, constraints are forgotten, tool usage patterns change.

Four drift dimensions:
    1. Goal Drift       — current goal vs checkpoint goal divergence
    2. Tool Drift       — tool call pattern deviation from baseline
    3. Planning Drift   — task graph diverging from expected structure
    4. Constraint Drift — must_not_redo / active_constraints being violated
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .db import MemoryDB

log = logging.getLogger("engram.drift")


@dataclass
class DriftSignal:
    """Multi-dimensional drift measurement."""

    goal_drift: float = 0.0          # 0.0 = aligned, 1.0 = completely diverged
    tool_drift: float = 0.0          # 0.0 = stable pattern, 1.0 = fully changed
    planning_drift: float = 0.0      # 0.0 = on track, 1.0 = uncontrolled branching
    constraint_drift: float = 0.0    # 0.0 = all constraints held, 1.0 = all violated
    composite: float = 0.0           # weighted average
    violations: list[str] | None = None  # specific constraint violations detected

    def to_dict(self) -> dict:
        result = {
            "goal_drift": round(self.goal_drift, 4),
            "tool_drift": round(self.tool_drift, 4),
            "planning_drift": round(self.planning_drift, 4),
            "constraint_drift": round(self.constraint_drift, 4),
            "composite": round(self.composite, 4),
        }
        if self.violations:
            result["violations"] = self.violations
        return result


# Composite weights — constraint drift is heaviest because violation
# of must_not_redo is the most dangerous form of regression.
WEIGHT_GOAL = 0.25
WEIGHT_TOOL = 0.15
WEIGHT_PLANNING = 0.20
WEIGHT_CONSTRAINT = 0.40


def compute_composite(signal: DriftSignal) -> float:
    """Weighted composite of 4 drift dimensions."""
    return round(
        WEIGHT_GOAL * signal.goal_drift
        + WEIGHT_TOOL * signal.tool_drift
        + WEIGHT_PLANNING * signal.planning_drift
        + WEIGHT_CONSTRAINT * signal.constraint_drift,
        4,
    )


def detect_drift(
    db: "MemoryDB",
    task_id: int,
    current_state: dict,
    user_id: str = "default",
) -> DriftSignal:
    """Detect execution drift by comparing current state against checkpoint.

    Args:
        db: database handle
        task_id: the task to check drift for
        current_state: the Agent's current reported state, containing:
            - goal: current goal string
            - tools_used: list of tool names used since restore
            - actions_taken: list of actions performed
            - in_progress: current in_progress items
    Returns:
        DriftSignal with 4 dimensions + composite + violations
    """
    from . import checkpoint as _ckpt

    signal = DriftSignal()

    # Get the latest checkpoint for comparison baseline
    ckpt = _ckpt.get_checkpoint(db, task_id, user_id=user_id)
    if ckpt is None:
        # No checkpoint = no baseline = no drift measurable
        return signal

    ckpt_state = ckpt["state"]

    # 1. Goal Drift
    signal.goal_drift = _measure_goal_drift(
        ckpt_state.get("goal", ""),
        current_state.get("goal", ""),
    )

    # 2. Tool Drift
    signal.tool_drift = _measure_tool_drift(
        ckpt_state.get("working_set", {}),
        current_state.get("tools_used", []),
    )

    # 3. Planning Drift
    signal.planning_drift = _measure_planning_drift(
        db, task_id, ckpt_state, current_state,
    )

    # 4. Constraint Drift (most critical)
    signal.constraint_drift, signal.violations = _measure_constraint_drift(
        ckpt_state, current_state,
    )

    signal.composite = compute_composite(signal)

    # Tier 3 persistence: record drift signal for analytical queries
    try:
        db.record_signal(
            task_id=task_id,
            signal_type="drift",
            dimensions=signal.to_dict(),
            composite_score=signal.composite,
            user_id=user_id,
        )
    except Exception as exc:
        log.debug("drift signal persistence failed (non-fatal): %s", exc)

    return signal


def _measure_goal_drift(checkpoint_goal: str, current_goal: str) -> float:
    """Measure how much the current goal has diverged from the checkpoint goal.

    Uses word-level overlap as a lightweight proxy. 0.0 = identical, 1.0 = no overlap.
    """
    if not checkpoint_goal and not current_goal:
        return 0.0
    if not checkpoint_goal or not current_goal:
        return 1.0

    # Normalize and tokenize
    ckpt_words = set(checkpoint_goal.lower().split())
    curr_words = set(current_goal.lower().split())

    if not ckpt_words or not curr_words:
        return 1.0

    intersection = ckpt_words & curr_words
    union = ckpt_words | curr_words
    jaccard = len(intersection) / len(union) if union else 0.0

    # Invert: high jaccard = low drift
    return round(1.0 - jaccard, 4)


def _measure_tool_drift(
    checkpoint_working_set: dict,
    current_tools_used: list[str],
) -> float:
    """Measure tool usage pattern deviation.

    Compares tools in the checkpoint's working_set against currently used tools.
    High drift = using completely different tools than before the interruption.
    """
    baseline_tools = set(checkpoint_working_set.get("tools", []) or [])
    current_tools = set(current_tools_used or [])

    if not baseline_tools and not current_tools:
        return 0.0
    if not baseline_tools:
        # No baseline tools recorded — can't measure drift
        return 0.0
    if not current_tools:
        return 0.0  # Haven't used any tools yet — not drift, just early

    intersection = baseline_tools & current_tools
    # How many of the current tools are new (not in baseline)?
    new_tools = current_tools - baseline_tools
    if not current_tools:
        return 0.0

    novelty_ratio = len(new_tools) / len(current_tools)
    return round(min(novelty_ratio, 1.0), 4)


def _measure_planning_drift(
    db: "MemoryDB",
    task_id: int,
    checkpoint_state: dict,
    current_state: dict,
) -> float:
    """Measure planning drift — is the execution diverging from expected structure?

    Checks:
    - Are we working on things not in the checkpoint's in_progress?
    - Has the task spawned unexpected subtasks?
    """
    ckpt_planned = set(checkpoint_state.get("in_progress", []) or [])
    curr_planned = set(current_state.get("in_progress", []) or [])

    if not ckpt_planned and not curr_planned:
        return 0.0
    if not ckpt_planned:
        return 0.0  # No plan baseline

    # How much of current work is unplanned?
    if not curr_planned:
        return 0.0

    unplanned = curr_planned - ckpt_planned
    drift_ratio = len(unplanned) / len(curr_planned) if curr_planned else 0.0

    # Also check task graph explosion (too many subtasks = planning drift)
    task = db.get_task(task_id)
    if task and task.execution_id:
        all_tasks = db.get_execution_tasks(task.execution_id)
        # If there are way more tasks than expected, that's drift
        if len(all_tasks) > 5:
            explosion_factor = min((len(all_tasks) - 5) / 10.0, 0.5)
            drift_ratio = min(drift_ratio + explosion_factor, 1.0)

    return round(min(drift_ratio, 1.0), 4)


def _measure_constraint_drift(
    checkpoint_state: dict,
    current_state: dict,
) -> tuple[float, list[str]]:
    """Measure constraint violation — the most dangerous drift.

    Checks:
    - Are any must_not_redo items being redone?
    - Are active_constraints being violated?

    Returns (drift_score, list_of_violations).
    """
    violations: list[str] = []

    # Check must_not_redo violations
    must_not_redo = checkpoint_state.get("must_not_redo", []) or []
    actions_taken = set(current_state.get("actions_taken", []) or [])

    for item in must_not_redo:
        if isinstance(item, dict):
            action = item.get("action", "")
        else:
            action = str(item)
        if action and action in actions_taken:
            violations.append(f"REDO_VIOLATION: '{action}' was in must_not_redo but was redone")

    # Check active_constraints violations
    active_constraints = checkpoint_state.get("active_constraints", []) or []
    violated_constraints = current_state.get("violated_constraints", []) or []
    for constraint in violated_constraints:
        if constraint in active_constraints:
            violations.append(f"CONSTRAINT_VIOLATED: '{constraint}'")

    # Score: proportion of constraints violated
    total_constraints = len(must_not_redo) + len(active_constraints)
    if total_constraints == 0:
        return 0.0, violations

    drift = len(violations) / total_constraints
    return round(min(drift, 1.0), 4), violations
