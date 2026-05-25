"""Tests for the Runtime Observation Layer."""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from engram.observation.transcript import Transcript, Turn, ContentBlock
from engram.observation.events import (
    ObservableEvent,
    EventType,
    Anchor,
    extract_events,
    verify_reversibility,
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


# --- Fixtures ---


@pytest.fixture
def sample_transcript_path(tmp_path) -> Path:
    """Create a minimal Claude Code-style JSONL transcript."""
    transcript_file = tmp_path / "session_001.jsonl"
    turns = [
        {
            "id": "msg_user_01",
            "role": "user",
            "content": [{"type": "text", "text": "Fix the auth bug in login.py"}],
        },
        {
            "id": "msg_asst_01",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "I need to look at login.py first"},
                {"type": "text", "text": "I'll start by examining the login.py file to understand the current auth flow."},
                {"type": "tool_use", "name": "bash", "input": {"command": "cat login.py"}},
            ],
        },
        {
            "id": "msg_asst_02",
            "role": "assistant",
            "content": [
                {"type": "tool_result", "content": [{"type": "text", "text": "def login(user, password):\n    return check_auth(user, password)"}]},
                {"type": "text", "text": "Now let me fix the authentication logic."},
                {"type": "tool_use", "name": "file_edit", "input": {"path": "login.py", "content": "fixed"}},
            ],
        },
        {
            "id": "msg_asst_03",
            "role": "assistant",
            "content": [
                {"type": "tool_result", "content": "Error: Permission denied", "is_error": True},
                {"type": "text", "text": "Let me try a different approach."},
                {"type": "tool_use", "name": "bash", "input": {"command": "chmod +w login.py"}},
                {"type": "tool_use", "name": "bash", "input": {"command": "cat login.py"}},
                {"type": "tool_use", "name": "bash", "input": {"command": "cat login.py"}},
            ],
        },
    ]
    with transcript_file.open("w") as fh:
        for turn in turns:
            fh.write(json.dumps(turn) + "\n")
    return transcript_file


# --- Layer 0: Transcript Tests ---


class TestTranscript:
    def test_load_jsonl(self, sample_transcript_path):
        transcript = Transcript.from_jsonl(sample_transcript_path)
        assert len(transcript) == 4
        assert transcript.source_path == sample_transcript_path

    def test_turn_id_lookup(self, sample_transcript_path):
        transcript = Transcript.from_jsonl(sample_transcript_path)
        turn = transcript.get_turn("msg_asst_01")
        assert turn is not None
        assert turn.role == "assistant"
        assert turn.block_count == 3

    def test_block_access_by_index(self, sample_transcript_path):
        transcript = Transcript.from_jsonl(sample_transcript_path)
        turn = transcript.get_turn("msg_asst_01")
        block = turn.get_block(2)
        assert block is not None
        assert block.block_type == "tool_use"
        assert block.raw["name"] == "bash"

    def test_get_span(self, sample_transcript_path):
        transcript = Transcript.from_jsonl(sample_transcript_path)
        block = transcript.get_span("msg_asst_01", 2)
        assert block is not None
        assert block.block_type == "tool_use"

    def test_get_span_invalid(self, sample_transcript_path):
        transcript = Transcript.from_jsonl(sample_transcript_path)
        assert transcript.get_span("nonexistent", 0) is None
        assert transcript.get_span("msg_asst_01", 99) is None

    def test_tool_uses(self, sample_transcript_path):
        transcript = Transcript.from_jsonl(sample_transcript_path)
        turn = transcript.get_turn("msg_asst_01")
        assert len(turn.tool_uses) == 1
        assert turn.tool_uses[0].raw["name"] == "bash"

    def test_assistant_and_user_turns(self, sample_transcript_path):
        transcript = Transcript.from_jsonl(sample_transcript_path)
        assert len(transcript.user_turns) == 1
        assert len(transcript.assistant_turns) == 3

    def test_from_json_array(self, tmp_path):
        array_file = tmp_path / "session.json"
        data = [
            {"id": "t1", "role": "user", "content": "hello"},
            {"id": "t2", "role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ]
        array_file.write_text(json.dumps(data))
        transcript = Transcript.from_json_array(array_file)
        assert len(transcript) == 2


# --- Layer 1: Events Tests ---


class TestEvents:
    def test_extract_events(self, sample_transcript_path):
        transcript = Transcript.from_jsonl(sample_transcript_path)
        events = extract_events(transcript)
        assert len(events) > 0
        # Should have tool_called, tool_result, planning, error events
        event_types = {e.event_type for e in events}
        assert EventType.TOOL_CALLED in event_types
        assert EventType.ERROR_OBSERVED in event_types
        assert EventType.PLANNING_STATEMENT in event_types

    def test_anchor_structure(self, sample_transcript_path):
        transcript = Transcript.from_jsonl(sample_transcript_path)
        events = extract_events(transcript)
        for event in events:
            assert event.anchor.turn_id != ""
            assert event.anchor.block_index >= 0

    def test_reversibility(self, sample_transcript_path):
        transcript = Transcript.from_jsonl(sample_transcript_path)
        events = extract_events(transcript)
        errors = verify_reversibility(events, transcript)
        assert errors == [], f"Reversibility broken: {errors}"

    def test_tool_called_metadata(self, sample_transcript_path):
        transcript = Transcript.from_jsonl(sample_transcript_path)
        events = extract_events(transcript)
        tool_events = [e for e in events if e.event_type == EventType.TOOL_CALLED]
        assert len(tool_events) >= 1
        assert "tool" in tool_events[0].metadata
        assert tool_events[0].metadata["tool"] == "bash"

    def test_error_detected(self, sample_transcript_path):
        transcript = Transcript.from_jsonl(sample_transcript_path)
        events = extract_events(transcript)
        errors = [e for e in events if e.event_type == EventType.ERROR_OBSERVED]
        assert len(errors) == 1
        assert "Permission denied" in errors[0].raw_text

    def test_sequence_ordering(self, sample_transcript_path):
        transcript = Transcript.from_jsonl(sample_transcript_path)
        events = extract_events(transcript)
        indices = [e.sequence_index for e in events]
        assert indices == sorted(indices)
        assert len(set(indices)) == len(indices)  # All unique

    def test_event_serialization(self):
        event = ObservableEvent(
            event_type=EventType.TOOL_CALLED,
            anchor=Anchor(turn_id="msg_01", block_index=2),
            raw_text="[bash] ls -la",
            metadata={"tool": "bash"},
            sequence_index=0,
        )
        data = event.to_dict()
        restored = ObservableEvent.from_dict(data)
        assert restored.event_type == event.event_type
        assert restored.anchor.turn_id == "msg_01"
        assert restored.anchor.block_index == 2


# --- Annotation Tests ---


class TestAnnotation:
    def test_create_annotation_store(self, tmp_path):
        store_path = tmp_path / "annotations.json"
        store = AnnotationStore(store_path)
        assert store.count() == 0

    def test_add_and_retrieve(self, tmp_path):
        store_path = tmp_path / "annotations.json"
        store = AnnotationStore(store_path)

        annotation = Annotation(
            label=AnnotationLabel.CONTINUITY_START_DEGRADING,
            anchor=Anchor(turn_id="msg_05", block_index=1),
            note="Agent starts repeating the same file reads",
        )
        store.add(annotation)
        assert store.count() == 1
        assert store.get_all()[0].label == AnnotationLabel.CONTINUITY_START_DEGRADING

    def test_persistence(self, tmp_path):
        store_path = tmp_path / "annotations.json"
        store = AnnotationStore(store_path)
        store.add(Annotation(
            label=AnnotationLabel.TOOL_THRASHING,
            anchor=Anchor(turn_id="msg_10", block_index=0),
            note="3x repeated bash calls",
        ))

        # Reload
        store2 = AnnotationStore(store_path)
        assert store2.count() == 1
        assert store2.get_all()[0].note == "3x repeated bash calls"

    def test_recovery_outcome(self, tmp_path):
        store_path = tmp_path / "annotations.json"
        store = AnnotationStore(store_path)
        store.add(Annotation(
            label=AnnotationLabel.RECOVERY_OUTCOME,
            anchor=Anchor(turn_id="msg_20", block_index=0),
            recovery_outcome=RecoveryOutcome.FAIL,
            note="Agent failed to restore working set after interruption",
        ))

        loaded = store.get_by_label(AnnotationLabel.RECOVERY_OUTCOME)
        assert len(loaded) == 1
        assert loaded[0].recovery_outcome == RecoveryOutcome.FAIL

    def test_summary(self, tmp_path):
        store_path = tmp_path / "annotations.json"
        store = AnnotationStore(store_path)
        store.add(Annotation(label=AnnotationLabel.TOOL_THRASHING, anchor=Anchor("t1", 0)))
        store.add(Annotation(label=AnnotationLabel.TOOL_THRASHING, anchor=Anchor("t2", 0)))
        store.add(Annotation(label=AnnotationLabel.PLANNING_SHIFT, anchor=Anchor("t3", 0)))

        summary = store.summary()
        assert summary["tool_thrashing"] == 2
        assert summary["planning_shift"] == 1


# --- Corpus Tests ---


class TestCorpus:
    def test_register_session(self, tmp_path):
        corpus = Corpus(tmp_path / "corpus")
        meta = SessionMeta(
            session_id="session_001",
            kind=SessionKind.NATURALISTIC,
            transcript_path="transcripts/session_001.jsonl",
            description="Implementing drift feature in engram",
            task_complexity="high",
        )
        corpus.register(meta)
        assert corpus.count() == 1

    def test_prevent_duplicate(self, tmp_path):
        corpus = Corpus(tmp_path / "corpus")
        meta = SessionMeta(
            session_id="s1",
            kind=SessionKind.NATURALISTIC,
            transcript_path="t.jsonl",
        )
        corpus.register(meta)
        with pytest.raises(ValueError, match="already registered"):
            corpus.register(meta)

    def test_filter_by_kind(self, tmp_path):
        corpus = Corpus(tmp_path / "corpus")
        corpus.register(SessionMeta(
            session_id="nat_1", kind=SessionKind.NATURALISTIC, transcript_path="a.jsonl",
        ))
        corpus.register(SessionMeta(
            session_id="ind_1", kind=SessionKind.INDUCED, transcript_path="b.jsonl",
            induced_conditions=["context_collapse"],
        ))
        corpus.register(SessionMeta(
            session_id="nat_2", kind=SessionKind.NATURALISTIC, transcript_path="c.jsonl",
        ))

        assert len(corpus.list_naturalistic()) == 2
        assert len(corpus.list_induced()) == 1

    def test_persistence(self, tmp_path):
        corpus_dir = tmp_path / "corpus"
        corpus = Corpus(corpus_dir)
        corpus.register(SessionMeta(
            session_id="s1", kind=SessionKind.INDUCED, transcript_path="x.jsonl",
            induced_conditions=["mid_task_interruption"],
        ))

        # Reload
        corpus2 = Corpus(corpus_dir)
        assert corpus2.count() == 1
        session = corpus2.get("s1")
        assert session.kind == SessionKind.INDUCED
        assert "mid_task_interruption" in session.induced_conditions

    def test_summary(self, tmp_path):
        corpus = Corpus(tmp_path / "corpus")
        corpus.register(SessionMeta(
            session_id="n1", kind=SessionKind.NATURALISTIC, transcript_path="a.jsonl",
            annotation_count=5,
        ))
        corpus.register(SessionMeta(
            session_id="i1", kind=SessionKind.INDUCED, transcript_path="b.jsonl",
            annotation_count=3,
        ))

        summary = corpus.summary()
        assert summary["total_sessions"] == 2
        assert summary["naturalistic"] == 1
        assert summary["induced"] == 1
        assert summary["total_annotations"] == 8
