"""DuckDB storage layer — schema, CRUD, vector search, FTS."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from dataclasses import dataclass, field

import duckdb

from .embedding import get_dimensions, MODEL_NAME
from .config import DEDUP_SEARCH_THRESHOLD, SIMILARITY_LOW

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

SESSION_LOG_SQL = """
CREATE SEQUENCE IF NOT EXISTS session_log_id_seq START 1;

CREATE TABLE IF NOT EXISTS session_memory_log (
    id INTEGER PRIMARY KEY DEFAULT nextval('session_log_id_seq'),
    session_id VARCHAR NOT NULL,
    memory_id INTEGER NOT NULL,
    user_id VARCHAR NOT NULL DEFAULT 'default',
    recalled_at TIMESTAMP NOT NULL DEFAULT now()
);
"""

SESSION_OUTCOME_SQL = """
CREATE SEQUENCE IF NOT EXISTS session_outcome_id_seq START 1;

CREATE TABLE IF NOT EXISTS session_outcome_log (
    id INTEGER PRIMARY KEY DEFAULT nextval('session_outcome_id_seq'),
    session_id VARCHAR NOT NULL,
    user_id VARCHAR NOT NULL DEFAULT 'default',
    outcome VARCHAR NOT NULL,
    recorded_at TIMESTAMP NOT NULL DEFAULT now()
);
"""


SESSION_LIFECYCLE_SQL = """
CREATE TABLE IF NOT EXISTS session_lifecycle (
    session_id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL DEFAULT 'default',
    started_at TIMESTAMP NOT NULL DEFAULT now(),
    last_active_at TIMESTAMP NOT NULL DEFAULT now(),
    ended_at TIMESTAMP,
    end_type VARCHAR
);
"""

TASK_SQL = """
CREATE SEQUENCE IF NOT EXISTS task_id_seq START 1;

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY DEFAULT nextval('task_id_seq'),
    name VARCHAR NOT NULL,
    goal TEXT NOT NULL DEFAULT '',
    status VARCHAR NOT NULL DEFAULT 'planning',
    user_id VARCHAR NOT NULL DEFAULT 'default',
    created_at TIMESTAMP NOT NULL DEFAULT now(),
    updated_at TIMESTAMP NOT NULL DEFAULT now(),
    metadata JSON DEFAULT '{}'::JSON
);
"""

@dataclass
class TaskRow:
    id: int
    name: str
    goal: str
    status: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    metadata: dict | None = field(default=None)

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
        self.conn.execute(SESSION_LOG_SQL)
        self.conn.execute(SESSION_OUTCOME_SQL)
        self.conn.execute(SESSION_LIFECYCLE_SQL)
        self.conn.execute(TASK_SQL)
        self._init_vss()

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

    def _init_vss(self):
        """Load VSS extension and create HNSW index if beneficial."""
        self._vss_available = False
        try:
            self.conn.execute("INSTALL vss")
            self.conn.execute("LOAD vss")
            self.conn.execute("SET hnsw_enable_experimental_persistence = true")
            self._vss_available = True
            log.info("VSS extension loaded")
            self._ensure_hnsw_index()
        except Exception as e:
            log.info("VSS extension unavailable: %s — using brute-force vector search", e)

    def _ensure_hnsw_index(self):
        """Create HNSW index on embedding column when memory count exceeds threshold."""
        if not self._vss_available:
            return
        try:
            total = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            if total < 1000:
                log.debug("Only %d memories, skipping HNSW index (threshold: 1000)", total)
                return

            existing = self.conn.execute(
                "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'memories'"
            ).fetchall()
            hnsw_names = [r[0] for r in existing if "hnsw" in r[0].lower()]
            if hnsw_names:
                log.debug("HNSW index already exists: %s", hnsw_names)
                return

            self.conn.execute(
                "CREATE INDEX memories_hnsw_idx ON memories USING HNSW (embedding) WITH (metric = 'cosine')"
            )
            log.info("HNSW index created on memories.embedding (%d rows)", total)
        except Exception as e:
            log.warning("HNSW index creation failed (non-fatal): %s", e)

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
        self._check_conn()
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

    def log_session_recall(self, session_id: str, memory_ids: list[int], user_id: str = "default"):
        if not memory_ids:
            return
        rows = [[session_id, mid, user_id] for mid in memory_ids]
        self.conn.executemany(
            "INSERT INTO session_memory_log (session_id, memory_id, user_id) VALUES (?, ?, ?)",
            rows,
        )

    def get_session_memories(self, session_id: str, user_id: str = "default") -> list[int]:
        rows = self._fetchall_dicts(
            """
            SELECT DISTINCT memory_id FROM session_memory_log
            WHERE session_id = ? AND user_id = ?
            ORDER BY memory_id
            """,
            [session_id, user_id],
        )
        return [r["memory_id"] for r in rows]

    def adjust_importance_batch(self, memory_ids: list[int], delta: float) -> int:
        if not memory_ids:
            return 0
        placeholders = ",".join("?" for _ in memory_ids)
        row = self.conn.execute(
            f"""
            UPDATE memories
            SET importance = CASE
                WHEN importance + ? > 1.0 THEN 1.0
                WHEN importance + ? < 0.0 THEN 0.0
                ELSE importance + ?
            END
            WHERE id IN ({placeholders})
            RETURNING id
            """,
            [delta, delta, delta] + memory_ids,
        ).fetchall()
        return len(row)

    def log_session_outcome(self, session_id: str, outcome: str, user_id: str = "default"):
        self.conn.execute(
            "INSERT INTO session_outcome_log (session_id, user_id, outcome) VALUES (?, ?, ?)",
            [session_id, user_id, outcome],
        )

    def get_memory_failure_count(self, memory_ids: list[int], user_id: str = "default") -> dict[int, int]:
        """Count how many failed sessions each memory was involved in."""
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        rows = self._fetchall_dicts(
            f"""
            SELECT sml.memory_id, COUNT(DISTINCT sol.session_id) AS fail_count
            FROM session_memory_log sml
            JOIN session_outcome_log sol
              ON sml.session_id = sol.session_id AND sml.user_id = sol.user_id
            WHERE sol.outcome = 'failure'
              AND sml.user_id = ?
              AND sml.memory_id IN ({placeholders})
            GROUP BY sml.memory_id
            """,
            [user_id] + memory_ids,
        )
        return {r["memory_id"]: r["fail_count"] for r in rows}

    def get_memory_outcome_counts(self, memory_ids: list[int],
                                   user_id: str = "default") -> dict[int, dict[str, int]]:
        """Count success and failure sessions for each memory.

        Returns {memory_id: {"success": N, "failure": M}}.
        """
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        rows = self._fetchall_dicts(
            f"""
            SELECT sml.memory_id, sol.outcome,
                   COUNT(DISTINCT sol.session_id) AS cnt
            FROM session_memory_log sml
            JOIN session_outcome_log sol
              ON sml.session_id = sol.session_id AND sml.user_id = sol.user_id
            WHERE sml.user_id = ?
              AND sml.memory_id IN ({placeholders})
            GROUP BY sml.memory_id, sol.outcome
            """,
            [user_id] + memory_ids,
        )
        result: dict[int, dict[str, int]] = {}
        for r in rows:
            mid = r["memory_id"]
            if mid not in result:
                result[mid] = {"success": 0, "failure": 0}
            outcome = r["outcome"]
            if outcome in ("success", "failure"):
                result[mid][outcome] = r["cnt"]
        return result

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
        threshold: float = SIMILARITY_LOW,
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
        threshold: float = DEDUP_SEARCH_THRESHOLD,
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

    def _check_conn(self):
        """Raise if connection has been closed."""
        if self.conn is None:
            raise RuntimeError("MemoryDB connection is closed")

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

    def get_metadata_for_stats(self, user_id: str = "default",
                               types: tuple[str, ...] | None = None) -> list[dict]:
        """Return only metadata JSON for engineering stats (lightweight).

        When *types* is given, only rows whose metadata->type matches are returned,
        avoiding a full-table scan of all memories.
        """
        if types:
            placeholders = ",".join("?" for _ in types)
            rows = self.conn.execute(
                f"""
                SELECT metadata FROM memories
                WHERE user_id = ?
                  AND json_extract_string(metadata, '$.type') IN ({placeholders})
                """,
                [user_id] + list(types),
            ).fetchall()
        else:
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

    # --- Task CRUD ---

    def _row_to_task(self, row: dict) -> TaskRow:
        return TaskRow(
            id=row["id"],
            name=row["name"],
            goal=row["goal"],
            status=row["status"],
            user_id=row["user_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=_parse_metadata(row.get("metadata")),
        )

    def create_task(self, name: str, goal: str = "", status: str = "planning",
                    user_id: str = "default", metadata: dict | None = None) -> int:
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        result = self.conn.execute(
            """
            INSERT INTO tasks (name, goal, status, user_id, metadata)
            VALUES (?, ?, ?, ?, ?::JSON)
            RETURNING id
            """,
            [name, goal, status, user_id, meta_json],
        ).fetchone()
        return result[0]

    def get_task(self, task_id: int) -> TaskRow | None:
        row = self._fetchone_dict(
            """
            SELECT id, name, goal, status, user_id, created_at, updated_at, metadata
            FROM tasks WHERE id = ?
            """,
            [task_id],
        )
        if not row:
            return None
        return self._row_to_task(row)

    def update_task(self, task_id: int, status: str | None = None,
                    goal: str | None = None, metadata: dict | None = None) -> bool:
        sets = ["updated_at = now()"]
        params: list = []
        if status is not None:
            sets.append("status = ?")
            params.append(status)
        if goal is not None:
            sets.append("goal = ?")
            params.append(goal)
        if metadata is not None:
            sets.append("metadata = ?::JSON")
            params.append(json.dumps(metadata, ensure_ascii=False))
        params.append(task_id)
        row = self.conn.execute(
            f"UPDATE tasks SET {', '.join(sets)} WHERE id = ? RETURNING id",
            params,
        ).fetchone()
        return row is not None

    def list_tasks(self, user_id: str = "default", status: str | None = None) -> list[TaskRow]:
        if status:
            rows = self._fetchall_dicts(
                """
                SELECT id, name, goal, status, user_id, created_at, updated_at, metadata
                FROM tasks WHERE user_id = ? AND status = ?
                ORDER BY updated_at DESC
                """,
                [user_id, status],
            )
        else:
            rows = self._fetchall_dicts(
                """
                SELECT id, name, goal, status, user_id, created_at, updated_at, metadata
                FROM tasks WHERE user_id = ?
                ORDER BY updated_at DESC
                """,
                [user_id],
            )
        return [self._row_to_task(r) for r in rows]

    def get_task_memories(self, task_id: int, user_id: str = "default") -> list[MemoryRow]:
        """Get all memories associated with a task via metadata.task_id."""
        rows = self._fetchall_dicts(
            """
            SELECT id, user_id, content, importance, category,
                   recall_count, created_at, last_accessed_at, metadata
            FROM memories
            WHERE user_id = ? AND json_extract_string(metadata, '$.task_id') = ?
            ORDER BY created_at DESC
            """,
            [user_id, str(task_id)],
        )
        return [_row_to_memory(r) for r in rows]

    def get_failures_by_component(self, component: str, user_id: str = "default",
                                  limit: int = 5) -> list[MemoryRow]:
        """Get failure memories for a given component, most recent first."""
        rows = self._fetchall_dicts(
            """
            SELECT id, user_id, content, importance, category,
                   recall_count, created_at, last_accessed_at, metadata
            FROM memories
            WHERE user_id = ?
              AND category = 'failure'
              AND json_extract_string(metadata, '$.component') = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            [user_id, component, limit],
        )
        return [_row_to_memory(r) for r in rows]

    # --- Session Lifecycle ---

    def upsert_session(self, session_id: str, user_id: str = "default"):
        """Register a new session or refresh its heartbeat."""
        self.conn.execute(
            """INSERT INTO session_lifecycle (session_id, user_id)
            VALUES (?, ?)
            ON CONFLICT (session_id) DO UPDATE SET last_active_at = now()""",
            [session_id, user_id],
        )

    def end_session(self, session_id: str, end_type: str = "handoff"):
        """Mark a session as ended (handoff / outcome)."""
        self.conn.execute(
            """UPDATE session_lifecycle
            SET ended_at = now(), last_active_at = now(), end_type = ?
            WHERE session_id = ?""",
            [end_type, session_id],
        )

    def get_interrupted_sessions(self, user_id: str = "default",
                                 stale_minutes: int = 30) -> list[dict]:
        """Find sessions that started but never ended.

        A session is interrupted if ended_at IS NULL and last_active_at
        is older than stale_minutes ago.
        """
        return self._fetchall_dicts(
            """SELECT session_id, started_at, last_active_at
            FROM session_lifecycle
            WHERE user_id = ? AND ended_at IS NULL
              AND last_active_at < now() - INTERVAL '1 MINUTE' * ?
            ORDER BY last_active_at DESC LIMIT 5""",
            [user_id, stale_minutes],
        )

    def get_session_activity_summary(self, session_id: str,
                                     user_id: str = "default") -> dict:
        """Reconstruct what happened in a session from existing data."""
        recalled_ids = self.get_session_memories(session_id, user_id)

        session_row = self._fetchone_dict(
            "SELECT started_at, last_active_at FROM session_lifecycle WHERE session_id = ?",
            [session_id],
        )
        recent_writes: list[dict] = []
        active_tasks: list[dict] = []

        if session_row:
            started = session_row["started_at"]
            last_active = session_row["last_active_at"]
            recent_writes = self._fetchall_dicts(
                """SELECT id, content, category,
                       json_extract_string(metadata, '$.type') AS mem_type
                FROM memories
                WHERE user_id = ? AND created_at BETWEEN ? AND ?
                ORDER BY created_at DESC LIMIT 10""",
                [user_id, started, last_active],
            )
            active_tasks = self._fetchall_dicts(
                """SELECT id, name, status, goal FROM tasks
                WHERE user_id = ? AND status NOT IN ('done', 'cancelled')
                ORDER BY updated_at DESC LIMIT 5""",
                [user_id],
            )

        return {
            "session_id": session_id,
            "recalled_memory_ids": recalled_ids,
            "memories_written": [
                {"id": w["id"], "snippet": w["content"][:100],
                 "type": w.get("mem_type", "")}
                for w in recent_writes
            ],
            "active_tasks": [
                {"id": t["id"], "name": t["name"], "status": t["status"]}
                for t in active_tasks
            ],
        }

    def cleanup_stale_sessions(self, user_id: str = "default",
                               stale_minutes: int = 30):
        """Mark old interrupted sessions as ended to avoid accumulation."""
        self.conn.execute(
            """UPDATE session_lifecycle
            SET ended_at = last_active_at, end_type = 'interrupted'
            WHERE user_id = ? AND ended_at IS NULL
              AND last_active_at < now() - INTERVAL '1 MINUTE' * ?""",
            [user_id, stale_minutes],
        )