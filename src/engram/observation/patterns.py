"""Automatic Pattern Detection — structural heuristics for pre-annotation.

Detects patterns that are strong candidates for manual annotation labels.
These are NOT labels themselves — they're "suggested annotations" that a human
should verify before promoting to ground truth.

Detected patterns:
- tool_thrashing: 3+ consecutive calls to the same tool without progress
- error_burst: 3+ errors within a 5-event window
- planning_storm: 3+ planning statements in close proximity (replanning)
- recovery_signal: planning statement immediately following an error burst

All detection is purely structural — no LLM inference, no semantic judgment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from engram.observation.events import ObservableEvent, EventType, Anchor
from engram.observation.annotation import Annotation, AnnotationLabel, RecoveryOutcome


class PatternType(str, Enum):
    """Detected pattern types — candidates for annotation."""

    TOOL_THRASHING = "tool_thrashing"
    ERROR_BURST = "error_burst"
    PLANNING_STORM = "planning_storm"
    RECOVERY_SIGNAL = "recovery_signal"


@dataclass
class DetectedPattern:
    """A structurally detected pattern — candidate for human annotation."""

    pattern_type: PatternType
    anchor: Anchor  # Start of the pattern
    span_events: list[int]  # sequence_indices of events in the pattern
    confidence: float  # 0-1, how strong the structural signal is
    description: str  # Human-readable explanation
    suggested_label: AnnotationLabel  # What annotation label this maps to

    def to_annotation(self, note_prefix: str = "[auto] ") -> Annotation:
        """Convert to an Annotation for review."""
        return Annotation(
            label=self.suggested_label,
            anchor=self.anchor,
            note=f"{note_prefix}{self.description}",
            annotator="auto_pattern_detector",
        )


def detect_tool_thrashing(
    events: list[ObservableEvent],
    min_repeat: int = 3,
) -> list[DetectedPattern]:
    """Detect consecutive calls to the same tool without forward progress.

    Heuristic: 3+ tool_called events with the same tool name in a row,
    possibly interleaved with their results but no other tool types.
    """
    patterns: list[DetectedPattern] = []
    tool_events = [e for e in events if e.event_type == EventType.TOOL_CALLED]

    if len(tool_events) < min_repeat:
        return patterns

    streak_start = 0
    for i in range(1, len(tool_events)):
        current_tool = tool_events[i].metadata.get("tool", "")
        prev_tool = tool_events[streak_start].metadata.get("tool", "")

        if current_tool != prev_tool:
            # Check if streak was long enough
            streak_length = i - streak_start
            if streak_length >= min_repeat:
                _emit_thrashing(patterns, tool_events, streak_start, i, prev_tool)
            streak_start = i

    # Check final streak
    streak_length = len(tool_events) - streak_start
    if streak_length >= min_repeat:
        tool_name = tool_events[streak_start].metadata.get("tool", "")
        _emit_thrashing(patterns, tool_events, streak_start, len(tool_events), tool_name)

    return patterns


def _emit_thrashing(
    patterns: list[DetectedPattern],
    tool_events: list[ObservableEvent],
    start: int,
    end: int,
    tool_name: str,
) -> None:
    streak_length = end - start
    confidence = min(0.5 + (streak_length - 3) * 0.15, 0.95)
    patterns.append(DetectedPattern(
        pattern_type=PatternType.TOOL_THRASHING,
        anchor=tool_events[start].anchor,
        span_events=[tool_events[j].sequence_index for j in range(start, end)],
        confidence=confidence,
        description=f"{streak_length}x consecutive '{tool_name}' calls",
        suggested_label=AnnotationLabel.TOOL_THRASHING,
    ))


def detect_error_bursts(
    events: list[ObservableEvent],
    window_size: int = 7,
    min_errors: int = 3,
) -> list[DetectedPattern]:
    """Detect clusters of errors within a sliding window.

    Strong signal for continuity degradation onset.
    """
    patterns: list[DetectedPattern] = []
    error_indices = [
        i for i, e in enumerate(events)
        if e.event_type == EventType.ERROR_OBSERVED
    ]

    if len(error_indices) < min_errors:
        return patterns

    # Sliding window over error positions
    visited_starts: set[int] = set()
    for start_idx in range(len(error_indices)):
        # Count errors within window_size events from this error
        first_event_idx = error_indices[start_idx]
        errors_in_window = []

        for err_idx in error_indices[start_idx:]:
            if events[err_idx].sequence_index - events[first_event_idx].sequence_index <= window_size:
                errors_in_window.append(err_idx)
            else:
                break

        if len(errors_in_window) >= min_errors:
            # Avoid overlapping patterns
            pattern_key = error_indices[start_idx]
            if pattern_key not in visited_starts:
                visited_starts.add(pattern_key)
                confidence = min(0.5 + (len(errors_in_window) - 3) * 0.1, 0.9)
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.ERROR_BURST,
                    anchor=events[first_event_idx].anchor,
                    span_events=[events[ei].sequence_index for ei in errors_in_window],
                    confidence=confidence,
                    description=f"{len(errors_in_window)} errors within {window_size}-event window",
                    suggested_label=AnnotationLabel.CONTINUITY_START_DEGRADING,
                ))

    return patterns


def detect_planning_storms(
    events: list[ObservableEvent],
    window_size: int = 10,
    min_plans: int = 3,
) -> list[DetectedPattern]:
    """Detect clusters of planning statements — agent is replanning heavily.

    Strong signal for planning_shift.
    """
    patterns: list[DetectedPattern] = []
    plan_indices = [
        i for i, e in enumerate(events)
        if e.event_type == EventType.PLANNING_STATEMENT
    ]

    if len(plan_indices) < min_plans:
        return patterns

    visited_starts: set[int] = set()
    for start_idx in range(len(plan_indices)):
        first_event_idx = plan_indices[start_idx]
        plans_in_window = []

        for plan_idx in plan_indices[start_idx:]:
            if events[plan_idx].sequence_index - events[first_event_idx].sequence_index <= window_size:
                plans_in_window.append(plan_idx)
            else:
                break

        if len(plans_in_window) >= min_plans:
            pattern_key = plan_indices[start_idx]
            if pattern_key not in visited_starts:
                visited_starts.add(pattern_key)
                confidence = min(0.4 + (len(plans_in_window) - 3) * 0.15, 0.85)
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.PLANNING_STORM,
                    anchor=events[first_event_idx].anchor,
                    span_events=[events[pi].sequence_index for pi in plans_in_window],
                    confidence=confidence,
                    description=f"{len(plans_in_window)} planning statements within {window_size}-event window",
                    suggested_label=AnnotationLabel.PLANNING_SHIFT,
                ))

    return patterns


def detect_recovery_signals(
    events: list[ObservableEvent],
    lookback: int = 5,
) -> list[DetectedPattern]:
    """Detect recovery attempts: planning statement after error burst.

    Pattern: error_observed → ... → planning_statement (within lookback events).
    """
    patterns: list[DetectedPattern] = []

    for i, event in enumerate(events):
        if event.event_type != EventType.PLANNING_STATEMENT:
            continue

        # Look back for recent errors
        window_start = max(0, i - lookback)
        recent_errors = [
            events[j] for j in range(window_start, i)
            if events[j].event_type == EventType.ERROR_OBSERVED
        ]

        if len(recent_errors) >= 2:
            confidence = min(0.4 + len(recent_errors) * 0.1, 0.8)
            patterns.append(DetectedPattern(
                pattern_type=PatternType.RECOVERY_SIGNAL,
                anchor=event.anchor,
                span_events=[e.sequence_index for e in recent_errors] + [event.sequence_index],
                confidence=confidence,
                description=f"Planning after {len(recent_errors)} errors (possible recovery attempt)",
                suggested_label=AnnotationLabel.RECOVERY_ATTEMPT,
            ))

    return patterns


def detect_all_patterns(events: list[ObservableEvent]) -> list[DetectedPattern]:
    """Run all pattern detectors and return consolidated results."""
    all_patterns: list[DetectedPattern] = []
    all_patterns.extend(detect_tool_thrashing(events))
    all_patterns.extend(detect_error_bursts(events))
    all_patterns.extend(detect_planning_storms(events))
    all_patterns.extend(detect_recovery_signals(events))

    # Sort by sequence position
    all_patterns.sort(key=lambda p: p.span_events[0] if p.span_events else 0)
    return all_patterns
