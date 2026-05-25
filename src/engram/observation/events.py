"""Layer 1: Structurally Observable Events — no inference, fully reversible.

Extracts events that are directly visible in the transcript structure:
- tool_called: a tool_use block exists
- tool_result: a tool_result block exists
- planning_statement: assistant text with planning markers
- error_observed: tool_result with error indicators

Every event carries an Anchor (turn_id + block_index) for reversibility.
From any event, you can always get back to the exact raw transcript span.

NOT in Layer 1 (these belong to Layer 2 / semantic interpretation):
- goal_updated, drift_detected, recovery_scored
- Any classification that requires judgment
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator

from engram.observation.transcript import Transcript, Turn, ContentBlock


class EventType(str, Enum):
    """Structurally observable event types — no inference needed."""

    TOOL_CALLED = "tool_called"
    TOOL_RESULT = "tool_result"
    PLANNING_STATEMENT = "planning_statement"
    ERROR_OBSERVED = "error_observed"
    CONTEXT_WINDOW_WARNING = "context_window_warning"
    USER_INSTRUCTION = "user_instruction"


@dataclass(frozen=True)
class Anchor:
    """Reversible pointer back to raw transcript.

    Uses turn_id + block_index (not character offset) for stability
    across transcript format changes.
    """

    turn_id: str
    block_index: int

    def to_dict(self) -> dict:
        return {"turn_id": self.turn_id, "block_index": self.block_index}

    @classmethod
    def from_dict(cls, data: dict) -> "Anchor":
        return cls(turn_id=data["turn_id"], block_index=data["block_index"])


@dataclass(frozen=True)
class ObservableEvent:
    """A single structurally-observable event extracted from transcript.

    Key invariant: given the anchor + source transcript, you can always
    reconstruct the exact raw text that produced this event.
    """

    event_type: EventType
    anchor: Anchor
    raw_text: str  # Extracted text — for quick access without re-reading transcript
    metadata: dict = field(default_factory=dict)  # Structured fields (tool name, error type, etc.)
    sequence_index: int = 0  # Position in the global event sequence

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "anchor": self.anchor.to_dict(),
            "raw_text": self.raw_text,
            "metadata": self.metadata,
            "sequence_index": self.sequence_index,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ObservableEvent":
        return cls(
            event_type=EventType(data["event_type"]),
            anchor=Anchor.from_dict(data["anchor"]),
            raw_text=data.get("raw_text", ""),
            metadata=data.get("metadata", {}),
            sequence_index=data.get("sequence_index", 0),
        )


# --- Planning detection heuristics (structural, not semantic) ---

_PLANNING_MARKERS = [
    "I'll ", "I will ", "Let me ", "First,", "Next,", "Then,",
    "My plan", "The approach", "Step 1", "Here's my plan",
    "I need to", "I should", "The strategy",
]


def _is_planning_text(text: str) -> bool:
    """Detect planning statements by structural markers (not semantic analysis)."""
    if len(text) < 20:
        return False
    text_lower = text[:200].lower()
    return any(marker.lower() in text_lower for marker in _PLANNING_MARKERS)


_ERROR_MARKERS = [
    "Permission denied", "No such file", "command not found",
    "traceback", "Traceback", "TRACEBACK",
    "Worker execution error",
    "Exit code ", "exit code ",
    "ENOENT", "EACCES", "EPERM",
    "ConnectionRefusedError", "TimeoutError",
    "ModuleNotFoundError", "ImportError",
    "SyntaxError", "IndentationError",
    "API_ERROR:", "API error",
]

# Patterns that look like errors but aren't (documentation, normal data)
_ERROR_FALSE_POSITIVE_MARKERS = [
    '"ok":true',
    '"ok": true',
    "## 1. 主要功能",  # Documentation content
    '"document":',  # Code search results
    '"code_url":',  # Code search results
]


def _is_error_text(text: str) -> bool:
    """Detect error indicators in tool results (structural pattern matching).

    Prioritizes precision over recall — avoids marking documentation content
    or normal API responses as errors.
    """
    # Skip if it looks like normal content (documentation, successful API responses)
    text_prefix = text[:500]
    if any(fp in text_prefix for fp in _ERROR_FALSE_POSITIVE_MARKERS):
        return False

    return any(marker in text for marker in _ERROR_MARKERS)


def extract_events(transcript: Transcript) -> list[ObservableEvent]:
    """Extract all structurally observable events from a transcript.

    This is a pure structural extraction — no inference, no judgment.
    Every event is reversible: anchor points back to exact source location.
    """
    events: list[ObservableEvent] = []
    sequence_counter = 0

    for turn in transcript:
        for block in turn.blocks:
            extracted = _extract_block_event(turn, block, sequence_counter)
            if extracted is not None:
                events.append(extracted)
                sequence_counter += 1

    return events


def _extract_block_event(
    turn: Turn, block: ContentBlock, sequence_index: int
) -> ObservableEvent | None:
    """Extract an observable event from a single content block, if applicable."""
    anchor = Anchor(turn_id=turn.turn_id, block_index=block.block_index)

    if block.block_type == "tool_use":
        tool_name = block.raw.get("name", "unknown")
        tool_input = block.raw.get("input", {})
        # Truncate raw_text for storage efficiency
        input_preview = json.dumps(tool_input, ensure_ascii=False)[:500]
        return ObservableEvent(
            event_type=EventType.TOOL_CALLED,
            anchor=anchor,
            raw_text=f"[{tool_name}] {input_preview}",
            metadata={"tool": tool_name, "has_input": bool(tool_input)},
            sequence_index=sequence_index,
        )

    if block.block_type == "tool_result":
        text = block.text
        is_error = block.raw.get("is_error", False) or _is_error_text(text)
        if is_error:
            return ObservableEvent(
                event_type=EventType.ERROR_OBSERVED,
                anchor=anchor,
                raw_text=text[:1000],
                metadata={"source": "tool_result", "is_error_flag": block.raw.get("is_error", False)},
                sequence_index=sequence_index,
            )
        return ObservableEvent(
            event_type=EventType.TOOL_RESULT,
            anchor=anchor,
            raw_text=text[:500],
            metadata={"length": len(text)},
            sequence_index=sequence_index,
        )

    if block.block_type == "text" and turn.role == "assistant":
        text = block.text
        if _is_planning_text(text):
            return ObservableEvent(
                event_type=EventType.PLANNING_STATEMENT,
                anchor=anchor,
                raw_text=text[:500],
                metadata={},
                sequence_index=sequence_index,
            )

    if block.block_type == "text" and turn.role == "user":
        text = block.text
        if len(text) > 10:
            return ObservableEvent(
                event_type=EventType.USER_INSTRUCTION,
                anchor=anchor,
                raw_text=text[:500],
                metadata={},
                sequence_index=sequence_index,
            )

    return None


def verify_reversibility(
    events: list[ObservableEvent], transcript: Transcript
) -> list[str]:
    """Verify that all events can be traced back to their source spans.

    Returns list of error messages (empty = all OK).
    """
    errors: list[str] = []
    for event in events:
        block = transcript.get_span(event.anchor.turn_id, event.anchor.block_index)
        if block is None:
            errors.append(
                f"Event #{event.sequence_index} ({event.event_type.value}): "
                f"anchor {event.anchor.to_dict()} not found in transcript"
            )
    return errors
