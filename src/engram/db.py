"""DuckDB storage layer — schema, CRUD, vector search, FTS."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from dataclasses import dataclass, field

import duckdb

from .embedding import get_dimensions, MODEL_NAME

log = logging.getLogger("engram.db")

ENGRAM_DIR = os.path.join(os.path.expanduser("~"), ".engram")
DB_PATH = os.path.join(ENGRAM_DIR, "memories.duckdb")

_DEFAULT_DIM = 768


def _dim() -> int:
    cache = os.path.join(ENGRAM_DIR, ".dim_cache")
    try:
        with open(cache) as f:
            line = f.read().strip()
            if ":" in line:
                model, dim_str = line.rsplit(":", 1)
                if model == MODEL_NAME:
                    return int(dim_str)
            else:
                return int(line)
    except Exception:
        pass
    try:
        dim = get_dimensions()
        os.makedirs(ENGRAM_DIR, exist_ok=True)
        with open(cache, "w") as f:
            f.write(f"{MODEL_NAME}:{dim}")
        return dim
    except Exception:
        return _DEFAULT_DIM


def _recover_wal(db_path: str) -> None:
    """Remove corrupted WAL file so DuckDB can start fresh from checkpoint."""
    wal = db_path + ".wal"
    if not os.path.exists(wal):
        return
    log.warning("WAL file exists at startup: %s (%d bytes)", wal, os.path.getsize(wal))
    bak = wal + ".recovery"
    try:
        os.replace(wal, bak)
        log.warning("WAL moved to %s for recovery", bak)
    except OSError as e:
        log.error("Failed to move WAL: %s", e)


def _connect_with_retry(db_path: str) -> duckdb.DuckDBPyConnection:
    """Connect to DuckDB with automatic WAL recovery on failure."""
    try:
        return duckdb.connect(db_path)
    except (duckdb.IOException, duckdb.InternalException) as e:
        log.warning("DuckDB connect failed: %s — attempting WAL recovery", e)
        _recover_wal(db_path)
        try:
            return duckdb.connect(db_path)
        except Exception:
            log.error("DuckDB still fails after WAL recovery, creating fresh DB")
            if os.path.exists(db_path):
                os.replace(db_path, db_path + ".corrupt")
            return duckdb.connect(db_path)


def _schema_sql(dim: int) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY DEFAULT nextval('memory_id_seq'),
    user_id VARCHAR DEFAULT 'default',
    content TEXT NOT NULL,
    embedding FLOAT[{dim}],
    importance FLOAT NOT NULL DEFAULT 0.5,
    category VARCHAR NOT NULL DEFAULT 'fact',
    recall_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT now(),
    last_accessed_at TIMESTAMP NOT NULL DEFAULT now(),
    metadata JSON DEFAULT '{{}}'::JSON
);
"""

FTS_SQL = """
INSTALL fts;
LOAD fts;
"""


@dataclass
class MemoryRow:
    id: int
    user_id: str
    content: str
    importance: float
    category: str
    recall_count: int
    created_at: datetime
    last_accessed_at: datetime
    metadata: dict | None = field(default=None)
    similarity: float = 0.0
    bm25_score: float = 0.0


def _parse_metadata(raw) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _row_to_memory(row: dict) -> MemoryRow:
    """Map a dict row to MemoryRow. Uses column names for resilience."""
    return MemoryRow(
        id=row["id"],
        user_id=row["user_id"],
        content=row["content"],
        importance=row["importance"],
        category=row["category"],
        recall_count=row["recall_count"],
        created_at=row["created_at"],
        last_accessed_at=row["last_accessed_at"],
        metadata=_parse_metadata(row["metadata"]),
    )


class MemoryDB:
    def __init__(self, db_path: str = DB_PATH, dim: int | None = None):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_path = db_path
        self._fts_available = False
        self._fts_dirty = True  # Force rebuild on first search
        self.conn = _connect_with_retry(db_path)
        self._dim = dim or _dim()
        self._init_schema()

    @property
    def fts_available(self) -> bool:
        return self._fts_available

    def _fetchone_dict(self, sql: str, params=None) -> dict | None:
        """Execute query and return first row as dict, or None."""
        result = self.conn.execute(sql, params or [])
        cols = [d[0] for d in result.description]
        row = result.fetchone()
        if row is None:
            return None
        return dict(zip(cols, row))

    def _fetchall_dicts(self, sql: str, params=None) -> list[dict]:
        """Execute query and return all rows as list of dicts."""
        result = self.conn.execute(sql, params or [])
        cols = [d[0] for d in result.description]
        return [dict(zip(cols, row)) for row in result.fetchall()]

    def _init_schema(self):
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS memory_id_seq START 1")
        self.conn.execute(_schema_sql(self._dim))
        try:
            self.conn.execute(
                "ALTER TABLE memories ADD COLUMN IF NOT EXISTS metadata JSON DEFAULT '{}'::JSON"
            )
        except Exception as e:
            log.debug("ALTER TABLE metadata column (already exists): %s", e)
        try:
            self.conn.execute(FTS_SQL)
        except Exception as e:
            log.warning("FTS extension load failed: %s", e)
        self._rebuild_fts_index()

    def _rebuild_fts_index(self):
        """Rebuild the FTS index from scratch to include all current data."""
        try:
            self.conn.execute(
                "PRAGMA create_fts_index('memories', 'id', 'content', overwrite=1)"
            )
            self._fts_available = True
            self._fts_dirty = False
            log.info("FTS index rebuilt successfully")
        except Exception as e:
            self._fts_available = False
            log.warning("FTS index rebuild failed: %s — BM25 search unavailable", e)

    def _ensure_fts_fresh(self):
        """Rebuild FTS index if writes have occurred since last rebuild."""
        if self._fts_available and self._fts_dirty:
            self._rebuild_fts_index()

    def _float_cast(self) -> str:
        return f"FLOAT[{self._dim}]"

    def _validate_embedding(self, embedding: list[float]) -> None:
        if len(embedding) != self._dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dim}, got {len(embedding)}"
            )

    def insert(
        self,
        content: str,
        embedding: list[float],
        importance: float = 0.5,
        category: str = "fact",
        user_id: str = "default",
        metadata: dict | None = None,
    ) -> int:
        self._validate_embedding(embedding)
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        cast = self._float_cast()
        result = self.conn.execute(
            f"""
            INSERT INTO memories (user_id, content, embedding, importance, category, metadata)
            VALUES (?, ?, ?::{cast}, ?, ?, ?::JSON)
            RETURNING id
            """,
            [user_id, content, embedding, importance, category, meta_json],
        ).fetchone()
        self._fts_dirty = True
        return result[0]

    def update(
        self,
        memory_id: int,
        content: str,
        embedding: list[float],
        importance: float | None = None,
        metadata: dict | None = None,
    ):
        self._validate_embedding(embedding)
        cast = self._float_cast()
        sets = [f"content = ?", f"embedding = ?::{cast}", "last_accessed_at = now()"]
        params: list = [content, embedding]
        if importance is not None:
            sets.append("importance = ?")
            params.append(importance)
        if metadata is not None:
            sets.append("metadata = ?::JSON")
            params.append(json.dumps(metadata, ensure_ascii=False))
        params.append(memory_id)
        self.conn.execute(
            f"UPDATE memories SET {', '.join(sets)} WHERE id = ?",
            params,
        )
        self._fts_dirty = True

    def bump_recall(self, memory_id: int):
        self.conn.execute(
            """
            UPDATE memories
            SET recall_count = recall_count + 1, last_accessed_at = now()
            WHERE id = ?
            """,
            [memory_id],
        )

    def delete(self, memory_id: int):
        self.conn.execute("DELETE FROM memories WHERE id = ?", [memory_id])
        self._fts_dirty = True

    def get_by_id(self, memory_id: int) -> MemoryRow | None:
        row = self._fetchone_dict(
            """
            SELECT id, user_id, content, importance, category,
                   recall_count, created_at, last_accessed_at, metadata
            FROM memories WHERE id = ?
            """,
            [memory_id],
        )
        if not row:
            return None
        return _row_to_memory(row)

    def get_all(self, user_id: str = "default") -> list[MemoryRow]:
        rows = self._fetchall_dicts(
            """
            SELECT id, user_id, content, importance, category,
                   recall_count, created_at, last_accessed_at, metadata
            FROM memories WHERE user_id = ?
            """,
            [user_id],
        )
        return [_row_to_memory(r) for r in rows]

    def search_vector(
        self,
        query_embedding: list[float],
        user_id: str = "default",
        top_k: int = 20,
        threshold: float = 0.20,
    ) -> list[MemoryRow]:
        self._validate_embedding(query_embedding)
        threshold = max(0.0, min(1.0, float(threshold)))
        top_k = max(1, min(1000, int(top_k)))
        cast = self._float_cast()
        rows = self._fetchall_dicts(
            f"""
            WITH scored AS (
                SELECT id, user_id, content, importance, category,
                       recall_count, created_at, last_accessed_at, metadata,
                       array_cosine_similarity(embedding, ?::{cast}) AS sim
                FROM memories
                WHERE user_id = ?
            )
            SELECT id, user_id, content, importance, category,
                   recall_count, created_at, last_accessed_at, metadata, sim
            FROM scored
            WHERE sim >= ?
            ORDER BY sim DESC
            LIMIT ?
            """,
            [query_embedding, user_id, threshold, top_k],
        )
        results = []
        for r in rows:
            m = _row_to_memory(r)
            m.similarity = r["sim"]
            results.append(m)
        return results

    def search_fts(
        self,
        query: str,
        user_id: str = "default",
        top_k: int = 20,
    ) -> list[MemoryRow]:
        self._ensure_fts_fresh()
        if not self._fts_available:
            return []
        try:
            rows = self._fetchall_dicts(
                """
                SELECT m.id, m.user_id, m.content, m.importance, m.category,
                       m.recall_count, m.created_at, m.last_accessed_at,
                       m.metadata, fts.score
                FROM (
                    SELECT id, fts_main_memories.match_bm25(id, ?) AS score
                    FROM memories
                ) fts
                JOIN memories m ON m.id = fts.id
                WHERE m.user_id = ? AND fts.score IS NOT NULL
                ORDER BY fts.score DESC
                LIMIT ?
                """,
                [query, user_id, top_k],
            )
            results = []
            for r in rows:
                m = _row_to_memory(r)
                m.bm25_score = r["score"] if r["score"] else 0.0
                results.append(m)
            return results
        except Exception as e:
            log.error("FTS search failed (query=%r): %s", query, e)
            return []

    def search_similar_for_dedup(
        self,
        query_embedding: list[float],
        user_id: str = "default",
        top_k: int = 10,
        threshold: float = 0.60,
    ) -> list[tuple[int, str, list[float]]]:
        cast = self._float_cast()
        rows = self._fetchall_dicts(
            f"""
            WITH scored AS (
                SELECT id, content, embedding,
                       array_cosine_similarity(embedding, ?::{cast}) AS sim
                FROM memories
                WHERE user_id = ?
            )
            SELECT id, content, embedding, sim
            FROM scored
            WHERE sim >= ?
            ORDER BY sim DESC
            LIMIT ?
            """,
            [query_embedding, user_id, threshold, top_k],
        )
        return [(r["id"], r["content"], list(r["embedding"])) for r in rows]

    def get_embedding(self, memory_id: int) -> list[float] | None:
        row = self.conn.execute(
            "SELECT embedding FROM memories WHERE id = ?", [memory_id]
        ).fetchone()
        if not row or not row[0]:
            return None
        return list(row[0])

    def count(self, user_id: str = "default") -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE user_id = ?", [user_id]
        ).fetchone()
        return row[0]

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def get_all_user_ids(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT user_id FROM memories"
        ).fetchall()
        return [r[0] for r in rows]

    def get_stats_aggregate(self, user_id: str = "default") -> dict:
        """Return category counts and total count via SQL aggregation."""
        rows = self.conn.execute(
            """
            SELECT category, COUNT(*) as cnt
            FROM memories WHERE user_id = ?
            GROUP BY category
            """,
            [user_id],
        ).fetchall()
        categories = {r[0]: r[1] for r in rows}
        total = sum(categories.values())
        return {"total": total, "categories": categories}

    def get_metadata_for_stats(self, user_id: str = "default") -> list[dict]:
        """Return only metadata JSON for engineering stats (lightweight)."""
        rows = self.conn.execute(
            "SELECT metadata FROM memories WHERE user_id = ?",
            [user_id],
        ).fetchall()
        return [_parse_metadata(r[0]) for r in rows]

    def get_strength_data(self, user_id: str = "default") -> list[tuple]:
        """Return (category, importance, last_accessed_at, recall_count) for strength calc."""
        rows = self._fetchall_dicts(
            """
            SELECT category, importance, last_accessed_at, recall_count
            FROM memories WHERE user_id = ?
            """,
            [user_id],
        )
        return [(r["category"], r["importance"], r["last_accessed_at"], r["recall_count"]) for r in rows]

    def get_by_ids_batch(self, memory_ids: list[int]) -> dict[int, MemoryRow]:
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        rows = self._fetchall_dicts(
            f"""
            SELECT id, user_id, content, importance, category,
                   recall_count, created_at, last_accessed_at, metadata
            FROM memories WHERE id IN ({placeholders})
            """,
            memory_ids,
        )
        return {r["id"]: _row_to_memory(r) for r in rows}

    def get_embeddings_batch(self, memory_ids: list[int]) -> dict[int, list[float]]:
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        rows = self._fetchall_dicts(
            f"SELECT id, embedding FROM memories WHERE id IN ({placeholders})",
            memory_ids,
        )
        return {r["id"]: list(r["embedding"]) for r in rows if r["embedding"]}

    def get_metadata_batch(self, memory_ids: list[int]) -> dict[int, dict]:
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        rows = self._fetchall_dicts(
            f"SELECT id, metadata FROM memories WHERE id IN ({placeholders})",
            memory_ids,
        )
        return {r["id"]: _parse_metadata(r["metadata"]) for r in rows}

    def get_neighbors_batch(
        self, memory_ids: list[int]
    ) -> dict[int, tuple[MemoryRow, list[float]]]:
        """Single query returning both row data and embedding for neighbor expansion."""
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        rows = self._fetchall_dicts(
            f"""
            SELECT id, user_id, content, importance, category,
                   recall_count, created_at, last_accessed_at, metadata,
                   embedding
            FROM memories WHERE id IN ({placeholders})
            """,
            memory_ids,
        )
        result: dict[int, tuple[MemoryRow, list[float]]] = {}
        for r in rows:
            m = _row_to_memory(r)
            emb = list(r["embedding"]) if r["embedding"] else []
            result[r["id"]] = (m, emb)
        return result
