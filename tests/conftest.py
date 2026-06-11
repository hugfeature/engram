"""Shared test fixtures for engram test suite."""

import os
import tempfile

# Redirect the engram state root to a throwaway dir BEFORE any engram module is
# imported. The path constants (ENGRAM_DIR / DB_PATH / SNAPSHOT_DIR / ...) are
# resolved at import time from ENGRAM_HOME, so this must run first — otherwise the
# suite reads/writes the real ~/.engram and fights the live daemon for the DuckDB
# WAL lock (FTS rebuild / CHECKPOINT "Conflicting lock is held" errors).
if not os.environ.get("ENGRAM_HOME"):
    os.environ["ENGRAM_HOME"] = tempfile.mkdtemp(prefix="engram-test-home-")

import pytest

from engram.db import MemoryDB
from engram.graph import MemoryGraph

FAKE_EMBED = [0.1] * 768


@pytest.fixture(autouse=True)
def _disable_sqlite_tier2(monkeypatch):
    """Disable SQLite Tier2 by default in tests.

    Most tests directly query DuckDB tables. test_sqlite_tier2.py explicitly
    enables it via its own env override.
    """
    monkeypatch.setenv("ENGRAM_SQLITE_TIER2", "0")


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Create a DB + Graph pair with fake embedding for isolated unit tests."""
    db = MemoryDB(str(tmp_path / "test.duckdb"), dim=768)
    graph = MemoryGraph(str(tmp_path / "test.json"))
    monkeypatch.setattr("engram.handlers.embed", lambda t: FAKE_EMBED)
    monkeypatch.setattr("engram.retrieve.embed", lambda t: FAKE_EMBED)
    return db, graph


@pytest.fixture
def db(tmp_path):
    """Create a bare MemoryDB with dim=768."""
    return MemoryDB(str(tmp_path / "test.duckdb"), dim=768)


@pytest.fixture
def graph(tmp_path):
    """Create a bare MemoryGraph."""
    return MemoryGraph(str(tmp_path / "test.json"))
