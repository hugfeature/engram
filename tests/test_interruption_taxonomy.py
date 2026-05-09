"""Tests for v0.12 Interruption Taxonomy.

Covers:
- Schema migration (new columns on session_lifecycle)
- end_session with interruption_reason + context
- cleanup_stale_sessions heuristic classification
- get_interrupted_sessions returns taxonomy fields
- report_interruption handler + shared state
- _build_recovery_hint per-reason routing
- recover replay preserves interruption_reason
"""

import json
import time

import pytest

from engram.db import (
    MemoryDB,
    INTERRUPTION_OVERFLOW,
    INTERRUPTION_USER_AWAY,
    INTERRUPTION_TOOL_FAILURE,
    INTERRUPTION_CRASH,
    INTERRUPTION_RATE_LIMIT,
    INTERRUPTION_UNKNOWN,
    VALID_INTERRUPTION_REASONS,
    RECOVERY_STRATEGIES,
)
from engram.handlers import (
    handle_report_interruption,
    _build_recovery_hint,
    TOOL_HANDLERS,
    ARG_MAPPING,
)
from engram.tools import TOOL_SCHEMAS

FAKE_EMBED = [0.1] * 768


@pytest.fixture
def db(tmp_path):
    return MemoryDB(str(tmp_path / "test.duckdb"), dim=768)


# --- Schema & Constants ---

def test_interruption_reason_enum_has_six_values():
    assert len(VALID_INTERRUPTION_REASONS) == 6
    expected = {"overflow", "user_away", "tool_failure", "crash", "rate_limit", "unknown"}
    assert VALID_INTERRUPTION_REASONS == expected


def test_recovery_strategies_cover_all_reasons():
    for reason in VALID_INTERRUPTION_REASONS:
        assert reason in RECOVERY_STRATEGIES, f"Missing recovery strategy for {reason}"
        strategy = RECOVERY_STRATEGIES[reason]
        assert "action" in strategy
        assert "hint" in strategy


def test_schema_has_new_columns(db):
    """session_lifecycle table should have interruption_reason and interruption_context."""
    columns = db.conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'session_lifecycle' ORDER BY column_name"
    ).fetchall()
    col_names = {r[0] for r in columns}
    assert "interruption_reason" in col_names
    assert "interruption_context" in col_names


# --- end_session with taxonomy ---

def test_end_session_stores_interruption_reason(db):
    db.upsert_session("sess-1", "default")
    db.end_session("sess-1", end_type="interrupted",
                   interruption_reason=INTERRUPTION_OVERFLOW,
                   interruption_context={"token_count": 195000})

    row = db.conn.execute(
        "SELECT end_type, interruption_reason, interruption_context "
        "FROM session_lifecycle WHERE session_id = 'sess-1'"
    ).fetchone()
    assert row[0] == "interrupted"
    assert row[1] == "overflow"
    context = json.loads(row[2]) if isinstance(row[2], str) else row[2]
    assert context["token_count"] == 195000


def test_end_session_without_reason_leaves_null(db):
    db.upsert_session("sess-2", "default")
    db.end_session("sess-2", end_type="handoff")

    row = db.conn.execute(
        "SELECT interruption_reason FROM session_lifecycle WHERE session_id = 'sess-2'"
    ).fetchone()
    assert row[0] is None


# --- cleanup_stale_sessions classification ---

def test_cleanup_classifies_short_session_as_crash(db):
    """A session < 2 minutes active should be classified as crash."""
    db.upsert_session("short-sess", "default")
    # Simulate: started_at == last_active_at (0 duration, definitely < 2 min)
    db.conn.execute(
        "UPDATE session_lifecycle SET last_active_at = started_at "
        "WHERE session_id = 'short-sess'"
    )
    # Make it stale (> 0 minutes for cleanup)
    db.conn.execute(
        "UPDATE session_lifecycle "
        "SET started_at = now() - INTERVAL '60 MINUTES', "
        "    last_active_at = now() - INTERVAL '60 MINUTES' "
        "WHERE session_id = 'short-sess'"
    )
    db.cleanup_stale_sessions("default", stale_minutes=30)

    row = db.conn.execute(
        "SELECT end_type, interruption_reason "
        "FROM session_lifecycle WHERE session_id = 'short-sess'"
    ).fetchone()
    assert row[0] == "interrupted"
    assert row[1] == INTERRUPTION_CRASH


def test_cleanup_classifies_normal_session_as_user_away(db):
    """A session with normal duration and no failures → user_away."""
    db.upsert_session("normal-sess", "default")
    db.conn.execute(
        "UPDATE session_lifecycle "
        "SET started_at = now() - INTERVAL '120 MINUTES', "
        "    last_active_at = now() - INTERVAL '60 MINUTES' "
        "WHERE session_id = 'normal-sess'"
    )
    db.cleanup_stale_sessions("default", stale_minutes=30)

    row = db.conn.execute(
        "SELECT interruption_reason FROM session_lifecycle WHERE session_id = 'normal-sess'"
    ).fetchone()
    assert row[0] == INTERRUPTION_USER_AWAY


def test_cleanup_classifies_failure_heavy_session_as_tool_failure(db, monkeypatch):
    """A session with ≥ 2 failure memories → tool_failure."""
    monkeypatch.setattr("engram.handlers.embed", lambda t: FAKE_EMBED)

    db.upsert_session("fail-sess", "default")
    # Set session window to past so failures fall within it
    db.conn.execute(
        "UPDATE session_lifecycle "
        "SET started_at = now() - INTERVAL '120 MINUTES', "
        "    last_active_at = now() - INTERVAL '60 MINUTES' "
        "WHERE session_id = 'fail-sess'"
    )
    # Insert 2 failure memories within the session window
    for i in range(2):
        db.insert(f"failure #{i}", FAKE_EMBED, 0.8, "failure", "default",
                  metadata={"type": "failure", "component": "test"})
    # Backdate the failures to fall within the session window
    db.conn.execute(
        "UPDATE memories SET created_at = now() - INTERVAL '90 MINUTES' "
        "WHERE category = 'failure'"
    )

    db.cleanup_stale_sessions("default", stale_minutes=30)

    row = db.conn.execute(
        "SELECT interruption_reason FROM session_lifecycle WHERE session_id = 'fail-sess'"
    ).fetchone()
    assert row[0] == INTERRUPTION_TOOL_FAILURE


# --- get_interrupted_sessions returns taxonomy ---

def test_get_interrupted_sessions_includes_reason(db):
    db.upsert_session("int-sess", "default")
    db.end_session("int-sess", end_type="interrupted",
                   interruption_reason=INTERRUPTION_RATE_LIMIT)

    sessions = db.get_interrupted_sessions("default", stale_minutes=0)
    # Should find it in the "recently classified" results
    rate_limited = [s for s in sessions if s["session_id"] == "int-sess"]
    assert len(rate_limited) == 1
    assert rate_limited[0]["interruption_reason"] == INTERRUPTION_RATE_LIMIT


# --- report_interruption handler ---

def test_report_interruption_valid_reason(db, tmp_path):
    from engram.graph import MemoryGraph
    graph = MemoryGraph(str(tmp_path / "g.json"))

    result = handle_report_interruption(
        db, graph,
        reason="overflow",
        context={"token_count": 190000},
        session_id=None,
        user_id="default",
    )
    assert result["ok"] is True
    assert result["reason"] == "overflow"


def test_report_interruption_invalid_reason(db, tmp_path):
    from engram.graph import MemoryGraph
    graph = MemoryGraph(str(tmp_path / "g.json"))

    result = handle_report_interruption(
        db, graph,
        reason="alien_invasion",
        user_id="default",
    )
    assert result.get("ok") is False
    assert "Invalid" in result["error"]


def test_report_interruption_with_session_closes_session(db, tmp_path):
    from engram.graph import MemoryGraph
    graph = MemoryGraph(str(tmp_path / "g.json"))

    db.upsert_session("close-me", "default")
    result = handle_report_interruption(
        db, graph,
        reason="rate_limit",
        context={"error": "429 Too Many Requests"},
        session_id="close-me",
        user_id="default",
    )
    assert result["ok"] is True

    row = db.conn.execute(
        "SELECT end_type, interruption_reason "
        "FROM session_lifecycle WHERE session_id = 'close-me'"
    ).fetchone()
    assert row[0] == "interrupted"
    assert row[1] == "rate_limit"


# --- _build_recovery_hint ---

def test_build_recovery_hint_overflow():
    session = {
        "session_id": "s1",
        "started_at": "2026-05-09 10:00:00",
        "last_active_at": "2026-05-09 11:00:00",
        "interruption_reason": "overflow",
        "interruption_context": None,
    }
    hint = _build_recovery_hint(session)
    assert hint["recovery_strategy"] == "restore_checkpoint"
    assert hint["memory_restore_mode"] == "SELECTIVE"
    assert "overflow" in hint["interruption_reason"]
    assert "Context window" in hint["hint"]


def test_build_recovery_hint_crash():
    session = {
        "session_id": "s2",
        "started_at": "2026-05-09 10:00:00",
        "last_active_at": "2026-05-09 10:01:00",
        "interruption_reason": "crash",
        "interruption_context": {},
    }
    hint = _build_recovery_hint(session)
    assert hint["recovery_strategy"] == "restore_checkpoint"
    assert hint["memory_restore_mode"] == "FULL"
    assert "crashed" in hint["hint"].lower()


def test_build_recovery_hint_unknown_fallback():
    session = {
        "session_id": "s3",
        "started_at": "2026-05-09 10:00:00",
        "last_active_at": "2026-05-09 10:30:00",
        "interruption_reason": None,
        "interruption_context": None,
    }
    hint = _build_recovery_hint(session)
    assert hint["interruption_reason"] == "unknown"
    assert hint["recovery_strategy"] == "show_summary"


# --- Tool registration ---

def test_report_interruption_registered():
    assert "report_interruption" in TOOL_HANDLERS
    assert "report_interruption" in ARG_MAPPING
    tool_names = {t.name for t in TOOL_SCHEMAS}
    assert "report_interruption" in tool_names


# --- Event replay (recover) ---

def test_replay_session_end_preserves_interruption_reason(tmp_path):
    """_replay_session_end should write interruption_reason to session_lifecycle."""
    from engram.recover import _replay_session_end

    db = MemoryDB(str(tmp_path / "replay.duckdb"), dim=768)
    db.upsert_session("replay-sess", "default")

    _replay_session_end(db, {
        "session_id": "replay-sess",
        "end_type": "interrupted",
        "interruption_reason": "tool_failure",
        "interruption_context": {"failure_count": 5},
    })

    row = db.conn.execute(
        "SELECT end_type, interruption_reason, interruption_context "
        "FROM session_lifecycle WHERE session_id = 'replay-sess'"
    ).fetchone()
    assert row[0] == "interrupted"
    assert row[1] == "tool_failure"
    context = json.loads(row[2]) if isinstance(row[2], str) else row[2]
    assert context["failure_count"] == 5
