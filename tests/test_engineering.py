"""v0.4 engineering state hub tests — track_failure, track_progress, enhanced stats."""

import pytest

from engram.db import MemoryDB
from engram.graph import MemoryGraph
from engram.handlers import (
    handle_track_failure, handle_track_progress, handle_stats,
)


@pytest.fixture
def env(tmp_path, monkeypatch):
    db = MemoryDB(str(tmp_path / "eng.duckdb"), dim=768)
    graph = MemoryGraph(str(tmp_path / "eng.json"))
    monkeypatch.setattr("engram.handlers.embed", lambda t: [0.1] * 768)
    monkeypatch.setattr("engram.retrieve.embed", lambda t: [0.1] * 768)
    return db, graph


class TestTrackFailure:
    def test_basic(self, env):
        db, graph = env
        result = handle_track_failure(db, graph, error="NPE on login", component="auth")
        assert "memory_id" in result
        assert "Failure tracked" in result["result"]

    def test_empty_error_rejected(self, env):
        db, graph = env
        result = handle_track_failure(db, graph, error="", component="auth")
        assert "error" in result

    def test_empty_component_rejected(self, env):
        db, graph = env
        result = handle_track_failure(db, graph, error="NPE", component="")
        assert "error" in result

    def test_invalid_severity_fallback(self, env):
        db, graph = env
        result = handle_track_failure(db, graph, error="NPE", component="auth", severity="bogus")
        mid = result["memory_id"]
        row = db.get_by_id(mid)
        assert row.metadata["severity"] == "major"

    def test_severity_importance_mapping(self, env):
        db, graph = env
        for sev, expected_imp in [("critical", 0.9), ("major", 0.7), ("minor", 0.5)]:
            result = handle_track_failure(db, graph, error=f"err-{sev}", component="x", severity=sev)
            row = db.get_by_id(result["memory_id"])
            assert abs(row.importance - expected_imp) < 0.01, f"{sev}: expected {expected_imp}, got {row.importance}"

    def test_uses_failure_category(self, env):
        db, graph = env
        result = handle_track_failure(db, graph, error="err", component="comp")
        row = db.get_by_id(result["memory_id"])
        assert row.category == "failure"

    def test_metadata_structure(self, env):
        db, graph = env
        result = handle_track_failure(db, graph, error="timeout", component="payment",
                                       root_cause="slow DB", fix="add index",
                                       related_test_ids=["test_pay_01"])
        meta = db.get_by_id(result["memory_id"]).metadata
        assert meta["type"] == "failure"
        assert meta["error"] == "timeout"
        assert meta["component"] == "payment"
        assert meta["severity"] == "major"
        assert meta["root_cause"] == "slow DB"
        assert meta["fix"] == "add index"
        assert meta["related_test_ids"] == ["test_pay_01"]
        assert "timestamp" in meta

    def test_searchable_content(self, env):
        db, graph = env
        result = handle_track_failure(db, graph, error="CSRF missing", component="auth",
                                       root_cause="middleware not loaded")
        row = db.get_by_id(result["memory_id"])
        assert "auth" in row.content
        assert "CSRF missing" in row.content
        assert "middleware not loaded" in row.content

    def test_with_all_fields(self, env):
        db, graph = env
        result = handle_track_failure(
            db, graph, error="500 on checkout", component="payment",
            root_cause="null pointer", severity="critical",
            fix="add null guard", related_test_ids=["t1", "t2"],
        )
        assert "memory_id" in result
        row = db.get_by_id(result["memory_id"])
        assert abs(row.importance - 0.9) < 0.01


class TestTrackProgress:
    def test_basic(self, env):
        db, graph = env
        result = handle_track_progress(db, graph, feature="login-refactor", status="in_progress")
        assert "memory_id" in result
        assert "Progress tracked" in result["result"]

    def test_empty_feature_rejected(self, env):
        db, graph = env
        result = handle_track_progress(db, graph, feature="", status="planning")
        assert "error" in result

    def test_invalid_status_rejected(self, env):
        db, graph = env
        result = handle_track_progress(db, graph, feature="x", status="invalid")
        assert "error" in result

    def test_clamps_completion(self, env):
        db, graph = env
        result = handle_track_progress(db, graph, feature="x", status="done", completion=150)
        meta = db.get_by_id(result["memory_id"]).metadata
        assert meta["completion"] == 100

    def test_uses_strategy_category(self, env):
        db, graph = env
        result = handle_track_progress(db, graph, feature="x", status="planning")
        row = db.get_by_id(result["memory_id"])
        assert row.category == "strategy"

    def test_blocked_high_importance(self, env):
        db, graph = env
        result = handle_track_progress(db, graph, feature="x", status="blocked",
                                        blockers=["waiting for API"])
        row = db.get_by_id(result["memory_id"])
        assert abs(row.importance - 0.9) < 0.01

    def test_done_low_importance(self, env):
        db, graph = env
        result = handle_track_progress(db, graph, feature="x", status="done", completion=100)
        row = db.get_by_id(result["memory_id"])
        assert abs(row.importance - 0.5) < 0.01

    def test_metadata_structure(self, env):
        db, graph = env
        result = handle_track_progress(db, graph, feature="auth-v2", status="in_progress",
                                        completion=60, blockers=["API design"],
                                        quality_score=0.85, notes="on track")
        meta = db.get_by_id(result["memory_id"]).metadata
        assert meta["type"] == "progress"
        assert meta["feature"] == "auth-v2"
        assert meta["status"] == "in_progress"
        assert meta["completion"] == 60
        assert meta["blockers"] == ["API design"]
        assert meta["quality_score"] == 0.85
        assert meta["notes"] == "on track"
        assert "timestamp" in meta

    def test_searchable_content(self, env):
        db, graph = env
        result = handle_track_progress(db, graph, feature="payment-flow", status="blocked",
                                        blockers=["vendor API down"])
        row = db.get_by_id(result["memory_id"])
        assert "payment-flow" in row.content
        assert "blocked" in row.content
        assert "vendor API down" in row.content

    def test_with_quality_score(self, env):
        db, graph = env
        result = handle_track_progress(db, graph, feature="x", status="review",
                                        quality_score=0.92)
        meta = db.get_by_id(result["memory_id"]).metadata
        assert meta["quality_score"] == 0.92


class TestEnhancedStats:
    def test_no_engineering_data(self, env):
        db, graph = env
        result = handle_stats(db)
        assert "engineering" not in result

    def test_with_failures(self, env):
        db, graph = env
        handle_track_failure(db, graph, error="e1", component="auth", severity="critical")
        handle_track_failure(db, graph, error="e2", component="auth", severity="major")
        handle_track_failure(db, graph, error="e3", component="payment", severity="minor")
        result = handle_stats(db)
        eng = result["engineering"]
        assert eng["failures"]["total"] == 3
        assert eng["failures"]["by_component"]["auth"] == 2
        assert eng["failures"]["by_component"]["payment"] == 1
        assert eng["failures"]["by_severity"]["critical"] == 1

    def test_with_progress(self, env):
        db, graph = env
        handle_track_progress(db, graph, feature="login", status="in_progress", completion=60)
        handle_track_progress(db, graph, feature="signup", status="done", completion=100)
        result = handle_stats(db)
        eng = result["engineering"]
        assert eng["features"]["total_tracked"] == 2
        assert "login" in eng["features"]["active"]
        assert "signup" not in eng["features"]["active"]

    def test_progress_latest_wins(self, env):
        db, graph = env
        handle_track_progress(db, graph, feature="x", status="planning", completion=0)
        handle_track_progress(db, graph, feature="x", status="in_progress", completion=50)
        result = handle_stats(db)
        active = result["engineering"]["features"]["active"]
        assert active["x"]["status"] == "in_progress"
        assert active["x"]["completion"] == 50

    def test_backward_compat_existing_stats(self, env):
        db, graph = env
        db.insert("plain memory", [0.1] * 768, 0.5, "fact", "default")
        result = handle_stats(db)
        assert result["total"] == 1
        assert "fact" in result["categories"]
        assert "engineering" not in result


class TestToolRegistration:
    def test_all_8_tools_registered(self):
        from engram.tools import TOOL_SCHEMAS
        names = {t.name for t in TOOL_SCHEMAS}
        assert names == {
            "recall_memory", "store_memory", "update_memory",
            "session_handoff", "consolidate_memory", "memory_stats",
            "track_failure", "track_progress",
        }

    def test_handlers_registered(self):
        from engram.handlers import TOOL_HANDLERS, ARG_MAPPING
        assert "track_failure" in TOOL_HANDLERS
        assert "track_progress" in TOOL_HANDLERS
        assert "track_failure" in ARG_MAPPING
        assert "track_progress" in ARG_MAPPING
