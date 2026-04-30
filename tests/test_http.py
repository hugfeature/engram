"""HTTP/REST endpoint tests using FastAPI TestClient."""

import pytest
from unittest.mock import patch

from fastapi.testclient import TestClient

from engram.db import MemoryDB
from engram.graph import MemoryGraph


@pytest.fixture
def client(tmp_path):
    db = MemoryDB(str(tmp_path / "http.duckdb"))
    graph = MemoryGraph(str(tmp_path / "http.json"))

    with patch("engram.http_server._get_db", return_value=db), \
         patch("engram.http_server._get_graph", return_value=graph), \
         patch("engram.handlers.embed", return_value=[0.1] * 768), \
         patch("engram.retrieve.embed", return_value=[0.1] * 768):
        from engram.http_server import app
        yield TestClient(app)


class TestHealthAndTools:
    def test_health(self, client):
        r = client.get("/v1/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] in ("ok", "degraded")
        assert data["version"] == "0.4.4"
        assert "db" in data
        assert "embedding_degraded" in data

    def test_tools_list(self, client):
        r = client.get("/v1/tools")
        assert r.status_code == 200
        names = [t["name"] for t in r.json()["tools"]]
        assert "recall" in names
        assert "store" in names


class TestStore:
    def test_store_new(self, client):
        r = client.post("/v1/store", json={"content": "test http store", "importance": 0.5})
        assert r.status_code == 200
        assert "memory_id" in r.json()

    def test_store_empty_rejected(self, client):
        r = client.post("/v1/store", json={"content": "", "importance": 0.5})
        assert r.status_code == 200
        assert "error" in r.json()

    def test_store_missing_field(self, client):
        r = client.post("/v1/store", json={"content": "no importance"})
        assert r.status_code == 422


class TestRecall:
    def test_recall_empty(self, client):
        r = client.post("/v1/recall", json={"query": ""})
        assert "error" in r.json()

    def test_recall_after_store(self, client):
        client.post("/v1/store", json={"content": "http recall target", "importance": 0.5})
        r = client.post("/v1/recall", json={"query": "http recall target"})
        assert r.json()["memoriesFound"] >= 1


class TestUpdate:
    def test_update_nonexistent(self, client):
        r = client.post("/v1/update", json={"memory_id": 9999, "new_content": "x"})
        assert "error" in r.json()


class TestHandoff:
    def test_basic_handoff(self, client):
        r = client.post("/v1/handoff", json={
            "summary": "HTTP handoff test",
            "completed": ["a"],
        })
        assert "memory_id" in r.json()

    def test_handoff_empty_rejected(self, client):
        r = client.post("/v1/handoff", json={"summary": ""})
        assert "error" in r.json()


class TestConsolidateAndStats:
    def test_consolidate(self, client):
        r = client.post("/v1/consolidate", json={})
        assert "result" in r.json()

    def test_stats(self, client):
        r = client.post("/v1/stats", json={})
        data = r.json()
        assert "total" in data
        assert data["total"] == 0


class TestTrackFailureHTTP:
    def test_basic(self, client):
        r = client.post("/v1/failure", json={"error": "NPE", "component": "auth"})
        assert r.status_code == 200
        assert "memory_id" in r.json()

    def test_missing_required(self, client):
        r = client.post("/v1/failure", json={"error": "NPE"})
        assert r.status_code == 422

    def test_with_all_fields(self, client):
        r = client.post("/v1/failure", json={
            "error": "timeout", "component": "payment",
            "root_cause": "slow query", "severity": "critical",
            "fix": "add index", "related_test_ids": ["t1"],
        })
        assert r.status_code == 200
        assert "memory_id" in r.json()


class TestTrackProgressHTTP:
    def test_basic(self, client):
        r = client.post("/v1/progress", json={"feature": "login", "status": "in_progress"})
        assert r.status_code == 200
        assert "memory_id" in r.json()

    def test_missing_required(self, client):
        r = client.post("/v1/progress", json={"feature": "login"})
        assert r.status_code == 422

    def test_invalid_status(self, client):
        r = client.post("/v1/progress", json={"feature": "x", "status": "invalid"})
        assert r.status_code == 200
        assert "error" in r.json()

    def test_with_all_fields(self, client):
        r = client.post("/v1/progress", json={
            "feature": "auth-v2", "status": "blocked",
            "completion": 60, "blockers": ["API"],
            "quality_score": 0.85, "notes": "waiting",
        })
        assert r.status_code == 200
        assert "memory_id" in r.json()
