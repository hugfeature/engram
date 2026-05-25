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

    # Drift Nudge: auto-inject warning memory when drift exceeds threshold
    _maybe_emit_drift_nudge(db, signal, ckpt_state, task_id, user_id)

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


# ============================================================
# Drift Nudge — auto-inject warning memory on high drift
# ============================================================

def _maybe_emit_drift_nudge(
    db: "MemoryDB",
    signal: DriftSignal,
    checkpoint_state: dict,
    task_id: int,
    user_id: str,
) -> None:
    """Emit a high-priority warning memory when drift exceeds threshold.

    This turns drift detection from passive observation into active correction:
    the warning memory will surface in the next recall_memory call, nudging
    the Agent back toward its original trajectory.
    """
    from .config import DRIFT_NUDGE_THRESHOLD, DRIFT_NUDGE_ENABLED

    if not DRIFT_NUDGE_ENABLED:
        return
    if signal.composite < DRIFT_NUDGE_THRESHOLD:
        return

    # Build a concise warning message
    goal = checkpoint_state.get("goal", "unknown")
    violations_text = ""
    if signal.violations:
        violations_text = " Violations: " + "; ".join(signal.violations[:3])

    warning_content = (
        f"⚠️ DRIFT WARNING (task#{task_id}): Execution has drifted significantly "
        f"(composite={signal.composite:.2f}).{violations_text} "
        f"Original goal: '{goal}'. "
        f"Constraint drift={signal.constraint_drift:.2f}, "
        f"Goal drift={signal.goal_drift:.2f}. "
        f"Re-check must_not_redo constraints before continuing."
    )

    # Store as failure category (fast decay ~11d, won't pollute long-term)
    try:
        from .embedding import embed
        warning_embedding = embed(warning_content)
        if warning_embedding is None:
            log.debug("drift nudge: embedding failed, skipping")
            return

        db.insert(
            content=warning_content,
            embedding=warning_embedding,
            importance=0.9,
            category="failure",
            user_id=user_id,
            metadata={
                "type": "drift_nudge",
                "task_id": task_id,
                "composite_drift": signal.composite,
                "constraint_drift": signal.constraint_drift,
                "goal_drift": signal.goal_drift,
            },
        )

        # Emit event for stats tracking
        try:
            db._emit_event("drift.nudge", {
                "task_id": task_id,
                "composite": signal.composite,
                "constraint_drift": signal.constraint_drift,
                "goal_drift": signal.goal_drift,
                "violations": signal.violations[:3] if signal.violations else [],
            })
        except Exception:
            pass  # non-fatal

        log.info(
            "drift nudge emitted for task#%d (composite=%.2f)",
            task_id, signal.composite,
        )
    except Exception as exc:
        log.debug("drift nudge emission failed (non-fatal): %s", exc)


# ============================================================
# Thrashing Circuit Breaker — detect repetitive tool calls
# ============================================================

class ThrashingDetector:
    """Detects when an agent is stuck calling the same tool repeatedly.

    Observation-driven: real sessions show 20-200x consecutive same-tool calls
    when agents hit UI state mismatch or API capability boundaries.

    The detector maintains a sliding window of recent tool calls per task.
    When the same tool appears N+ times consecutively, it injects a warning
    memory suggesting the agent change strategy.

    Unlike drift detection (which needs a checkpoint baseline), thrashing
    detection is purely local — it only looks at the recent call sequence.
    """

    def __init__(self) -> None:
        # Per-task state: {task_id: {"tool_history": [...], "last_nudge_at": int}}
        self._state: dict[int, dict] = {}

    def record_tool_call(
        self,
        db: "MemoryDB",
        task_id: int,
        tool_name: str,
        user_id: str = "default",
    ) -> bool:
        """Record a tool call and check for thrashing.

        Returns True if a thrashing nudge was emitted.
        """
        from .config import THRASHING_ENABLED, THRASHING_THRESHOLD, THRASHING_COOLDOWN

        if not THRASHING_ENABLED:
            return False

        state = self._state.setdefault(task_id, {
            "tool_history": [],
            "last_nudge_at": -THRASHING_COOLDOWN,  # Allow immediate first nudge
        })

        history = state["tool_history"]
        history.append(tool_name)

        # Keep only last 50 entries to bound memory
        if len(history) > 50:
            history[:] = history[-50:]

        # Count consecutive same-tool calls from the end
        consecutive = 0
        for call in reversed(history):
            if call == tool_name:
                consecutive += 1
            else:
                break

        if consecutive < THRASHING_THRESHOLD:
            return False

        # Cooldown check: don't spam nudges
        calls_since_last = len(history) - state["last_nudge_at"]
        if calls_since_last < THRASHING_COOLDOWN:
            return False

        # Fire the circuit breaker
        state["last_nudge_at"] = len(history)
        self._emit_thrashing_nudge(db, task_id, tool_name, consecutive, user_id)
        return True

    def _emit_thrashing_nudge(
        self,
        db: "MemoryDB",
        task_id: int,
        tool_name: str,
        consecutive: int,
        user_id: str,
    ) -> None:
        """Inject a high-priority warning memory about tool thrashing."""
        warning_content = (
            f"⚠️ THRASHING DETECTED (task#{task_id}): '{tool_name}' called "
            f"{consecutive}x consecutively without progress. "
            f"This pattern indicates a feedback loop failure — the tool's output "
            f"is not providing the expected state change. "
            f"RECOMMENDED: 1) Verify assumptions about current state, "
            f"2) Try a fundamentally different approach, "
            f"3) If UI automation: check whether the action actually took effect."
        )

        try:
            from .embedding import embed
            warning_embedding = embed(warning_content)
            if warning_embedding is None:
                log.debug("thrashing nudge: embedding failed, skipping")
                return

            db.insert(
                content=warning_content,
                embedding=warning_embedding,
                importance=0.85,
                category="failure",
                user_id=user_id,
                metadata={
                    "type": "thrashing_nudge",
                    "task_id": task_id,
                    "tool": tool_name,
                    "consecutive_calls": consecutive,
                },
            )

            try:
                db._emit_event("drift.thrashing", {
                    "task_id": task_id,
                    "tool": tool_name,
                    "consecutive": consecutive,
                })
            except Exception:
                pass

            log.info(
                "thrashing nudge emitted: task#%d tool='%s' x%d",
                task_id, tool_name, consecutive,
            )
        except Exception as exc:
            log.debug("thrashing nudge emission failed (non-fatal): %s", exc)

    def reset(self, task_id: int) -> None:
        """Reset thrashing state for a task (e.g. on task completion)."""
        self._state.pop(task_id, None)


# Module-level singleton
_thrashing_detector: ThrashingDetector | None = None


def get_thrashing_detector() -> ThrashingDetector:
    """Get or create the module-level thrashing detector."""
    global _thrashing_detector
    if _thrashing_detector is None:
        _thrashing_detector = ThrashingDetector()
    return _thrashing_detector
