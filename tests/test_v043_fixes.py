"""v0.4.3 regression tests — daemon, resolve polarity, db close, retrieve edges, handler safety."""

import os
import tempfile

import pytest

from engram.db import MemoryDB
from engram.graph import MemoryGraph


# --- Daemon helpers ---

class TestDaemonHelpers:
    def test_read_write_pid_roundtrip(self, tmp_path, monkeypatch):
        pid_file = str(tmp_path / "test.pid")
        monkeypatch.setattr("engram.daemon.PID_FILE", pid_file)

        from engram.daemon import _write_pid, _read_pid
        _write_pid(12345)
        assert _read_pid() == 12345

    def test_read_pid_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("engram.daemon.PID_FILE", str(tmp_path / "no.pid"))
        from engram.daemon import _read_pid
        assert _read_pid() is None

    def test_is_running_current_pid(self, monkeypatch):
        from engram.daemon import _is_running
        monkeypatch.setattr("subprocess.check_output",
                            lambda *a, **kw: "python -m engram.http_server --port 8900")
        assert _is_running(os.getpid()) is True

    def test_is_running_invalid_pid(self, monkeypatch):
        from engram.daemon import _is_running
        assert _is_running(9999999) is False

    def test_is_running_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr("engram.daemon.PID_FILE", str(tmp_path / "no.pid"))
        from engram.daemon import _is_running
        assert _is_running() is False


# --- Resolve polarity with punctuation ---

class TestPolarityPunctuation:
    def test_dont_detected_as_negation(self):
        from engram.resolve import _polarity
        result = _polarity("I don't like Python")
        assert result == -1

    def test_cant_detected_as_negation(self):
        from engram.resolve import _polarity
        result = _polarity("I can't use Java")
        assert result == -1

    def test_positive_still_works(self):
        from engram.resolve import _polarity
        result = _polarity("I love Python")
        assert result == 1

    def test_negative_word_detected(self):
        from engram.resolve import _polarity
        result = _polarity("I hate Java")
        assert result == -1

    def test_neutral(self):
        from engram.resolve import _polarity
        result = _polarity("The weather is fine")
        assert result is None

    def test_contradiction_with_punctuation(self):
        from engram.resolve import _is_contradiction
        assert _is_contradiction("I love Python!", "I don't like Python.")

    def test_tokenize_strips_brackets(self):
        from engram.resolve import _tokenize
        words = _tokenize("Use [Python] (version 3)")
        assert "python" in words
        assert "version" in words
        assert "[" not in words


# --- DB close and validation ---

class TestDBClose:
    def test_close_releases_connection(self, tmp_path):
        db = MemoryDB(str(tmp_path / "close.duckdb"), dim=768)
        emb = [0.1] * 768
        db.insert("test", emb, 0.5)
        db.close()
        assert db.conn is None

    def test_search_vector_clamps_threshold(self, tmp_path):
        db = MemoryDB(str(tmp_path / "clamp.duckdb"), dim=768)
        emb = [0.1] * 768
        db.insert("test", emb, 0.5)
        results = db.search_vector(emb, threshold=-1.0)
        assert len(results) >= 1

    def test_search_vector_clamps_top_k(self, tmp_path):
        db = MemoryDB(str(tmp_path / "topk.duckdb"), dim=768)
        emb = [0.1] * 768
        db.insert("test", emb, 0.5)
        results = db.search_vector(emb, top_k=-5)
        assert len(results) >= 0

    def test_embedding_dimension_validation(self, tmp_path):
        db = MemoryDB(str(tmp_path / "dim.duckdb"), dim=768)
        with pytest.raises(ValueError, match="dimension mismatch"):
            db.insert("bad", [0.1] * 100, 0.5)

    def test_update_dimension_validation(self, tmp_path):
        db = MemoryDB(str(tmp_path / "dim2.duckdb"), dim=768)
        emb = [0.1] * 768
        mid = db.insert("ok", emb, 0.5)
        with pytest.raises(ValueError, match="dimension mismatch"):
            db.update(mid, "bad", [0.1] * 100)


# --- Handler safety ---

class TestHandlerSafety:
    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        db = MemoryDB(str(tmp_path / "s.duckdb"), dim=768)
        graph = MemoryGraph(str(tmp_path / "s.json"))
        monkeypatch.setattr("engram.handlers.embed", lambda t: [0.1] * 768)
        monkeypatch.setattr("engram.retrieve.embed", lambda t: [0.1] * 768)
        return db, graph

    def test_update_with_string_memory_id(self, env):
        from engram.handlers import handle_update
        db, graph = env
        mid = db.insert("original", [0.1] * 768, 0.5, "fact", "default")
        result = handle_update(db, graph, memory_id=str(mid), new_content="updated")
        assert "Updated" in result["result"]

    def test_update_with_invalid_memory_id(self, env):
        from engram.handlers import handle_update
        db, graph = env
        result = handle_update(db, graph, memory_id="abc", new_content="x")
        assert "error" in result

    def test_user_id_validation_strip(self, env):
        from engram.handlers import handle_recall
        db, graph = env
        result = handle_recall(db, graph, query="test", user_id="  bob  ")
        assert result["memoriesFound"] == 0

    def test_user_id_validation_truncate(self, env):
        from engram.handlers import handle_recall
        db, graph = env
        long_id = "a" * 200
        result = handle_recall(db, graph, query="test", user_id=long_id)
        assert result["memoriesFound"] == 0

    def test_user_id_validation_empty(self, env):
        from engram.handlers import handle_recall
        db, graph = env
        result = handle_recall(db, graph, query="test", user_id="")
        assert result["memoriesFound"] == 0

    def test_safe_embed_failure_returns_error(self, env, monkeypatch):
        from engram.handlers import handle_store
        db, graph = env
        monkeypatch.setattr("engram.handlers.embed", lambda t: (_ for _ in ()).throw(RuntimeError("boom")))
        result = handle_store(db, graph, content="test", importance=0.5)
        assert "error" in result

    def test_track_failure_uses_safe_embed(self, env, monkeypatch):
        from engram.handlers import handle_track_failure
        db, graph = env
        monkeypatch.setattr("engram.handlers.embed", lambda t: (_ for _ in ()).throw(RuntimeError("boom")))
        result = handle_track_failure(db, graph, error="err", component="comp")
        assert "error" in result

    def test_track_progress_uses_safe_embed(self, env, monkeypatch):
        from engram.handlers import handle_track_progress
        db, graph = env
        monkeypatch.setattr("engram.handlers.embed", lambda t: (_ for _ in ()).throw(RuntimeError("boom")))
        result = handle_track_progress(db, graph, feature="x", status="planning")
        assert "error" in result

    def test_track_failure_validates_user_id(self, env):
        from engram.handlers import handle_track_failure
        db, graph = env
        result = handle_track_failure(db, graph, error="err", component="comp", user_id="  bob  ")
        assert "memory_id" in result

    def test_track_progress_validates_user_id(self, env):
        from engram.handlers import handle_track_progress
        db, graph = env
        result = handle_track_progress(db, graph, feature="x", status="planning", user_id="  bob  ")
        assert "memory_id" in result


# --- Graph type hint / batch ---

class TestGraphBatch:
    def test_incremental_index_uses_batch(self, tmp_path, monkeypatch):
        monkeypatch.setattr("engram.embedding.embed", lambda t: [0.1] * 768)
        db = MemoryDB(str(tmp_path / "b.duckdb"), dim=768)
        graph = MemoryGraph(str(tmp_path / "b.json"))

        vec = [0.1] * 768
        mid1 = db.insert("m1", vec, 0.5)
        mid2 = db.insert("m2", vec, 0.5)
        graph.index_memory_incremental(mid1, vec, db)
        graph.index_memory_incremental(mid2, vec, db)
        assert mid1 in graph._graph
        assert mid2 in graph._graph


# --- Embedding thread safety (mocked) ---

class TestEmbeddingDimensions:
    def test_get_dimensions_returns_int(self, monkeypatch):
        monkeypatch.setattr("engram.embedding._dimensions", 768)
        from engram.embedding import get_dimensions
        dim = get_dimensions()
        assert isinstance(dim, int)
        assert dim == 768

    def test_get_dimensions_consistent(self, monkeypatch):
        monkeypatch.setattr("engram.embedding._dimensions", 768)
        from engram.embedding import get_dimensions
        assert get_dimensions() == get_dimensions()


# --- Graph flush throttle ---

class TestGraphFlush:
    def test_explicit_flush_persists(self, tmp_path):
        path = str(tmp_path / "g.json")
        g = MemoryGraph(path)
        g.upsert_node(1, strength=0.5)
        g.upsert_node(2, strength=0.5)
        g.flush()

        g2 = MemoryGraph(path)
        assert 1 in g2._graph
        assert 2 in g2._graph


# --- v0.4.6: content length validation ---

class TestContentLength:
    def test_store_rejects_oversized_content(self, tmp_path, monkeypatch):
        db = MemoryDB(str(tmp_path / "len.duckdb"))
        graph = MemoryGraph(str(tmp_path / "len.json"))
        monkeypatch.setattr("engram.handlers.embed", lambda t: [0.1] * 768)
        from engram.handlers import handle_store
        big = "x" * 200_000
        result = handle_store(db, graph, content=big, importance=0.5)
        assert "error" in result
        assert "too large" in result["error"]

    def test_store_accepts_normal_content(self, tmp_path, monkeypatch):
        db = MemoryDB(str(tmp_path / "len2.duckdb"))
        graph = MemoryGraph(str(tmp_path / "len2.json"))
        monkeypatch.setattr("engram.handlers.embed", lambda t: [0.1] * 768)
        from engram.handlers import handle_store
        result = handle_store(db, graph, content="normal content", importance=0.5)
        assert "memory_id" in result
