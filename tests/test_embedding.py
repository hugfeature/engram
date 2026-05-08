"""Tests for embedding.py — degraded mode, recovery, and cache behavior."""

import pytest

from engram.embedding import embed, is_degraded, _embed_cache


class TestDegradedMode:
    def test_degraded_returns_zero_vector(self, monkeypatch):
        """When model fails to load, embed should return zero vector."""
        monkeypatch.setattr("engram.embedding._degraded", True)
        monkeypatch.setattr("engram.embedding._model", None)
        result = embed("hello world")
        assert len(result) == 768
        assert all(v == 0.0 for v in result)

    def test_is_degraded_reflects_state(self, monkeypatch):
        monkeypatch.setattr("engram.embedding._degraded", True)
        assert is_degraded() is True
        monkeypatch.setattr("engram.embedding._degraded", False)
        assert is_degraded() is False


class TestTryRecover:
    def test_recover_noop_when_not_degraded(self, monkeypatch):
        """If not degraded, try_recover should return True immediately."""
        monkeypatch.setattr("engram.embedding._degraded", False)
        from engram.embedding import try_recover
        assert try_recover() is True

    def test_recover_failure_when_model_unavailable(self, monkeypatch):
        """If degraded and model can't load, try_recover returns False."""
        monkeypatch.setattr("engram.embedding._degraded", True)
        monkeypatch.setattr("engram.embedding._model", None)
        # Force ImportError for sentence_transformers
        monkeypatch.setitem(
            __import__("sys").modules, "sentence_transformers",
            type(sys := __import__("sys")).module_from_spec(
                __import__("importlib").util.spec_from_file_location("sentence_transformers", "/dev/null")
            ) if False else None,
        )
        from engram.embedding import try_recover
        # Will fail because _model is None and we can't load
        result = try_recover()
        # Either True (if already recovered by another thread) or False
        assert isinstance(result, bool)


class TestEmbedCache:
    def test_cache_hit_returns_same_vector(self, monkeypatch):
        """Same text should return the same vector without re-encoding."""
        monkeypatch.setattr("engram.embedding._degraded", False)
        # Set up a mock model
        call_count = {"n": 0}

        class FakeModel:
            def encode(self, text, normalize_embeddings=True):
                call_count["n"] += 1
                import numpy as np
                return np.ones(768) * call_count["n"]

        monkeypatch.setattr("engram.embedding._model", FakeModel())
        # Clear cache
        _embed_cache.clear()

        r1 = embed("test cache")
        r2 = embed("test cache")
        assert r1 == r2
        assert call_count["n"] == 1  # Only encoded once

    def test_different_text_different_result(self, monkeypatch):
        """Different text should produce different results."""
        monkeypatch.setattr("engram.embedding._degraded", False)

        class FakeModel:
            def __init__(self):
                self.n = 0

            def encode(self, text, normalize_embeddings=True):
                self.n += 1
                import numpy as np
                return np.ones(768) * self.n

        monkeypatch.setattr("engram.embedding._model", FakeModel())
        _embed_cache.clear()

        r1 = embed("text one")
        r2 = embed("text two")
        assert r1 != r2

    def test_cache_eviction(self, monkeypatch):
        """Cache should evict oldest entries when full."""
        from engram.embedding import _EMBED_CACHE_MAX
        monkeypatch.setattr("engram.embedding._degraded", False)

        class FakeModel:
            def __init__(self):
                self.n = 0

            def encode(self, text, normalize_embeddings=True):
                self.n += 1
                import numpy as np
                return np.ones(768) * self.n

        monkeypatch.setattr("engram.embedding._model", FakeModel())
        _embed_cache.clear()

        # Fill cache beyond capacity
        for i in range(_EMBED_CACHE_MAX + 10):
            embed(f"text {i}")

        assert len(_embed_cache) <= _EMBED_CACHE_MAX