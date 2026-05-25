"""Runtime Observation Layer — structured analysis of agent session transcripts.

Three-layer architecture with hard boundaries:

- Layer 0: Raw Transcript (immutable JSONL, indexed by turn_id + block_index)
- Layer 1: Structurally Observable Events (reversible extraction, no inference)
- Layer 2: Semantic Interpretation (drift/recovery scoring — separate module, not here)

Design principles:
- Observation and interpretation NEVER couple
- Every Layer 1 event anchors back to Layer 0 span (reversible)
- Corpus distinguishes naturalistic vs induced sessions
"""

from engram.observation.transcript import Transcript, Turn, ContentBlock
from engram.observation.events import (
    ObservableEvent,
    EventType,
    Anchor,
    extract_events,
)
from engram.observation.annotation import (
    Annotation,
    AnnotationLabel,
    RecoveryOutcome,
    AnnotationStore,
)
from engram.observation.corpus import (
    SessionKind,
    SessionMeta,
    Corpus,
)

__all__ = [
    "Transcript",
    "Turn",
    "ContentBlock",
    "ObservableEvent",
    "EventType",
    "Anchor",
    "extract_events",
    "Annotation",
    "AnnotationLabel",
    "RecoveryOutcome",
    "AnnotationStore",
    "SessionKind",
    "SessionMeta",
    "Corpus",
]
