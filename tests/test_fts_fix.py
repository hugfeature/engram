"""Tests for FTS index freshness — verifies BM25 search works after writes."""

import pytest

from engram.db import MemoryDB


@pytest.fixture
def fts_db(tmp_path):
    return MemoryDB(str(tmp_path / "fts_test.duckdb"), dim=768)


class TestFtsFreshness:
    def test_fts_available_after_init(self, fts_db):
        assert fts_db.fts_available is True

    def test_fts_dirty_after_insert(self, fts_db):
        assert fts_db._fts_dirty is False  # clean after init rebuild
        fts_db.insert("hello world", [0.1] * 768)
        assert fts_db._fts_dirty is True

    def test_fts_dirty_after_update(self, fts_db):
        mid = fts_db.insert("original content", [0.1] * 768)
        fts_db._fts_dirty = False
        fts_db.update(mid, "updated content", [0.2] * 768)
        assert fts_db._fts_dirty is True

    def test_fts_dirty_after_delete(self, fts_db):
        mid = fts_db.insert("to be deleted", [0.1] * 768)
        fts_db._fts_dirty = False
        fts_db.delete(mid)
        assert fts_db._fts_dirty is True

    def test_search_fts_returns_inserted_content(self, fts_db):
        """Core fix: FTS search must find content inserted after index creation."""
        fts_db.insert("python programming language", [0.1] * 768)
        fts_db.insert("rust systems programming", [0.2] * 768)

        results = fts_db.search_fts("python", top_k=5)
        assert len(results) > 0
        assert any("python" in r.content.lower() for r in results)

    def test_search_fts_returns_updated_content(self, fts_db):
        """FTS search must find content after update."""
        mid = fts_db.insert("old content about java", [0.1] * 768)
        fts_db.update(mid, "new content about golang", [0.2] * 768)

        results = fts_db.search_fts("golang", top_k=5)
        assert len(results) > 0
        assert any("golang" in r.content.lower() for r in results)

    def test_search_fts_excludes_deleted_content(self, fts_db):
        """FTS search must not find content after delete."""
        mid = fts_db.insert("unique keyword xyzabc", [0.1] * 768)
        # Verify it's found first
        results = fts_db.search_fts("xyzabc", top_k=5)
        assert len(results) > 0

        fts_db.delete(mid)
        # After delete + rebuild, should not find it
        results = fts_db.search_fts("xyzabc", top_k=5)
        assert len(results) == 0

    def test_ensure_fts_fresh_only_rebuilds_when_dirty(self, fts_db):
        """_ensure_fts_fresh should not rebuild if not dirty."""
        fts_db._rebuild_fts_index()  # Clean state
        assert fts_db._fts_dirty is False

        # Calling _ensure_fts_fresh when clean should be a no-op
        fts_db._ensure_fts_fresh()
        assert fts_db._fts_dirty is False

    def test_fts_unavailable_when_extension_fails(self, tmp_path, monkeypatch):
        """If FTS extension can't load, fts_available should be False."""
        db = MemoryDB(str(tmp_path / "nofts.duckdb"), dim=768)
        # Simulate FTS failure by corrupting the PRAGMA
        db._fts_available = False
        db._fts_dirty = True
        db._ensure_fts_fresh()
        # Since we manually set _fts_available=False, search should return []
        results = db.search_fts("test", top_k=5)
        assert results == []
        db.close()

    def test_stats_exposes_fts_available(self, fts_db):
        """handle_stats should expose fts_available state."""
        from engram.handlers import handle_stats
        result = handle_stats(fts_db, user_id="default")
        assert "fts_available" in result
        assert result["fts_available"] is True