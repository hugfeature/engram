"""Corpus Manager — session registration, classification, and querying.

Distinguishes:
- naturalistic: agent encountered the situation organically during real work
- induced: human deliberately triggered the failure condition

Both are valuable but MUST carry different tags for downstream analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator


class SessionKind(str, Enum):
    """How the session's failure/recovery conditions arose."""

    NATURALISTIC = "naturalistic"  # Organic — real task, no intervention
    INDUCED = "induced"  # Deliberately triggered (interruption, context collapse)


@dataclass
class SessionMeta:
    """Metadata for a single observed session in the corpus."""

    session_id: str
    kind: SessionKind
    transcript_path: str  # Relative path to transcript file
    description: str = ""  # What task was being performed
    tags: list[str] = field(default_factory=list)  # Free-form tags
    agent: str = "claude-code"  # Which agent produced this session
    created_at: str = ""  # ISO timestamp
    annotation_count: int = 0  # Populated on load

    # Induced-specific fields
    induced_conditions: list[str] = field(default_factory=list)
    # e.g. ["context_collapse", "mid_task_interruption"]

    # Naturalistic-specific fields
    task_complexity: str = ""  # "low" / "medium" / "high"
    session_duration_turns: int = 0

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "kind": self.kind.value,
            "transcript_path": self.transcript_path,
            "description": self.description,
            "tags": self.tags,
            "agent": self.agent,
            "created_at": self.created_at,
            "annotation_count": self.annotation_count,
            "induced_conditions": self.induced_conditions,
            "task_complexity": self.task_complexity,
            "session_duration_turns": self.session_duration_turns,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionMeta":
        return cls(
            session_id=data["session_id"],
            kind=SessionKind(data["kind"]),
            transcript_path=data["transcript_path"],
            description=data.get("description", ""),
            tags=data.get("tags", []),
            agent=data.get("agent", "claude-code"),
            created_at=data.get("created_at", ""),
            annotation_count=data.get("annotation_count", 0),
            induced_conditions=data.get("induced_conditions", []),
            task_complexity=data.get("task_complexity", ""),
            session_duration_turns=data.get("session_duration_turns", 0),
        )


class Corpus:
    """Manages a collection of observed session transcripts.

    Storage: a single corpus.json index file + individual transcript/annotation files.
    """

    def __init__(self, corpus_dir: str | Path):
        self._dir = Path(corpus_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "corpus.json"
        self._sessions: list[SessionMeta] = []
        if self._index_path.exists():
            self._load()

    def _load(self) -> None:
        with self._index_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        self._sessions = [SessionMeta.from_dict(item) for item in data.get("sessions", [])]

    def _save(self) -> None:
        with self._index_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {"sessions": [s.to_dict() for s in self._sessions]},
                fh,
                indent=2,
                ensure_ascii=False,
            )

    def register(self, meta: SessionMeta) -> None:
        """Register a new session in the corpus."""
        from datetime import datetime, timezone

        if not meta.created_at:
            meta.created_at = datetime.now(timezone.utc).isoformat()

        # Prevent duplicate session_id
        existing_ids = {s.session_id for s in self._sessions}
        if meta.session_id in existing_ids:
            raise ValueError(f"Session '{meta.session_id}' already registered")

        self._sessions.append(meta)
        self._save()

    def get(self, session_id: str) -> SessionMeta | None:
        """Get session metadata by ID."""
        for session in self._sessions:
            if session.session_id == session_id:
                return session
        return None

    def list_all(self) -> list[SessionMeta]:
        return list(self._sessions)

    def list_by_kind(self, kind: SessionKind) -> list[SessionMeta]:
        return [s for s in self._sessions if s.kind == kind]

    def list_naturalistic(self) -> list[SessionMeta]:
        return self.list_by_kind(SessionKind.NATURALISTIC)

    def list_induced(self) -> list[SessionMeta]:
        return self.list_by_kind(SessionKind.INDUCED)

    def count(self) -> int:
        return len(self._sessions)

    def summary(self) -> dict:
        """Corpus-level statistics."""
        naturalistic = len(self.list_naturalistic())
        induced = len(self.list_induced())
        total_annotations = sum(s.annotation_count for s in self._sessions)
        return {
            "total_sessions": len(self._sessions),
            "naturalistic": naturalistic,
            "induced": induced,
            "total_annotations": total_annotations,
        }

    def __iter__(self) -> Iterator[SessionMeta]:
        return iter(self._sessions)

    def __len__(self) -> int:
        return len(self._sessions)
