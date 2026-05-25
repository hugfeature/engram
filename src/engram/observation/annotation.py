"""Annotation Ontology — minimal human labeling for runtime observation.

First-version ontology: exactly 5 labels. No more until 3 sessions are annotated
and label discriminability is evaluated.

Labels:
- continuity_start_degrading: agent begins losing context
- planning_shift: strategy visibly changes (not necessarily failure)
- tool_thrashing: repeated tool calls without forward progress
- recovery_attempt: agent tries to rebuild state
- recovery_outcome: whether the recovery succeeded (ok/fail)

Storage: JSON file per session, co-located with transcript.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Iterator

from engram.observation.events import Anchor


class AnnotationLabel(str, Enum):
    """Minimal annotation ontology — 5 labels only."""

    CONTINUITY_START_DEGRADING = "continuity_start_degrading"
    PLANNING_SHIFT = "planning_shift"
    TOOL_THRASHING = "tool_thrashing"
    RECOVERY_ATTEMPT = "recovery_attempt"
    RECOVERY_OUTCOME = "recovery_outcome"


class RecoveryOutcome(str, Enum):
    """Outcome of a recovery attempt."""

    OK = "ok"
    FAIL = "fail"


@dataclass
class Annotation:
    """A single human annotation anchored to a transcript location."""

    label: AnnotationLabel
    anchor: Anchor
    note: str = ""  # Free-text explanation by annotator
    recovery_outcome: RecoveryOutcome | None = None  # Only for RECOVERY_OUTCOME label
    annotator: str = "human"
    timestamp: str = ""  # ISO format, filled at creation time

    def to_dict(self) -> dict:
        result = {
            "label": self.label.value,
            "anchor": self.anchor.to_dict(),
            "note": self.note,
            "annotator": self.annotator,
            "timestamp": self.timestamp,
        }
        if self.recovery_outcome is not None:
            result["recovery_outcome"] = self.recovery_outcome.value
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "Annotation":
        recovery = None
        if data.get("recovery_outcome"):
            recovery = RecoveryOutcome(data["recovery_outcome"])
        return cls(
            label=AnnotationLabel(data["label"]),
            anchor=Anchor.from_dict(data["anchor"]),
            note=data.get("note", ""),
            recovery_outcome=recovery,
            annotator=data.get("annotator", "human"),
            timestamp=data.get("timestamp", ""),
        )


class AnnotationStore:
    """Persists annotations for a single session transcript.

    Storage format: JSON file with list of annotation dicts.
    Co-located with the transcript file (same directory, .annotations.json suffix).
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._annotations: list[Annotation] = []
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        with self._path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        self._annotations = [Annotation.from_dict(item) for item in data]

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as fh:
            json.dump(
                [a.to_dict() for a in self._annotations],
                fh,
                indent=2,
                ensure_ascii=False,
            )

    def add(self, annotation: Annotation) -> None:
        """Add an annotation and persist."""
        from datetime import datetime, timezone

        if not annotation.timestamp:
            annotation.timestamp = datetime.now(timezone.utc).isoformat()
        self._annotations.append(annotation)
        self._save()

    def get_all(self) -> list[Annotation]:
        return list(self._annotations)

    def get_by_label(self, label: AnnotationLabel) -> list[Annotation]:
        return [a for a in self._annotations if a.label == label]

    def count(self) -> int:
        return len(self._annotations)

    def summary(self) -> dict[str, int]:
        """Count annotations per label."""
        counts: dict[str, int] = {}
        for annotation in self._annotations:
            key = annotation.label.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def __iter__(self) -> Iterator[Annotation]:
        return iter(self._annotations)

    def __len__(self) -> int:
        return len(self._annotations)
