"""Hybrid retrieval — vector search + graph expansion + BM25."""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass

import logging

import numpy as np

from .db import MemoryDB, MemoryRow
from .embedding import embed
from .decay import compute_strength
from .graph import MemoryGraph
from .config import (
    SIMILARITY_HIGH, SIMILARITY_LOW, REINFORCE_SIM,
    W_BM25, W_VECTOR, GRAPH_MAX_DEPTH,
)


@dataclass
class RecallResult:
    id: int
    content: str
    category: str
    importance: float
    strength: float
    similarity: float
    bm25: float
    score: float


def recall(
    query: str,
    db: MemoryDB,
    graph: MemoryGraph,
    user_id: str = "default",
    top_k: int = 5,
) -> list[RecallResult]:
    query_embedding = embed(query)

    internal_k = max(20, top_k)

    log = logging.getLogger("engram.retrieve")

    vector_results = db.search_vector(
        query_embedding, user_id, top_k=internal_k, threshold=SIMILARITY_HIGH
    )
    if not vector_results:
        log.info("No results at threshold=%.2f, falling back to %.2f", SIMILARITY_HIGH, SIMILARITY_LOW)
        vector_results = db.search_vector(
            query_embedding, user_id, top_k=internal_k, threshold=SIMILARITY_LOW
        )

    fts_results = db.search_fts(query, user_id, top_k=internal_k)
    fts_by_id = {m.id: m.bm25_score for m in fts_results}

    max_bm25 = max(fts_by_id.values()) if fts_by_id else 1.0
    if max_bm25 == 0:
        max_bm25 = 1.0

    now = datetime.now(timezone.utc)
    scored: dict[int, RecallResult] = {}

    for m in vector_results:
        days = (now - m.last_accessed_at.replace(tzinfo=timezone.utc)).total_seconds() / 86400
        strength = compute_strength(m.category, m.importance, days, m.recall_count)
        bm25_raw = fts_by_id.get(m.id, 0.0)
        bm25_norm = bm25_raw / max_bm25

        vector_score = m.similarity * strength
        hybrid = W_BM25 * bm25_norm + W_VECTOR * vector_score

        scored[m.id] = RecallResult(
            id=m.id,
            content=m.content,
            category=m.category,
            importance=m.importance,
            strength=strength,
            similarity=m.similarity,
            bm25=bm25_norm,
            score=hybrid,
        )

    seed_ids = [m.id for m in vector_results[:5]]
    if seed_ids:
        graph_neighbors = graph.expand(seed_ids, max_depth=GRAPH_MAX_DEPTH, user_id=user_id)
        neighbor_ids = [mid for mid, _ in graph_neighbors[:top_k] if mid not in scored]
        neighbor_batch = db.get_by_ids_batch(neighbor_ids) if neighbor_ids else {}
        neighbor_emb_batch = db.get_embeddings_batch(neighbor_ids) if neighbor_ids else {}
        for mid, cum_weight in graph_neighbors[:top_k]:
            if mid in scored:
                continue
            m = neighbor_batch.get(mid)
            if not m or m.user_id != user_id:
                continue
            days = (now - m.last_accessed_at.replace(tzinfo=timezone.utc)).total_seconds() / 86400
            strength = compute_strength(m.category, m.importance, days, m.recall_count)
            bm25_raw = fts_by_id.get(mid, 0.0)
            bm25_norm = bm25_raw / max_bm25

            emb = neighbor_emb_batch.get(mid)
            sim = 0.0
            if emb:
                sim = float(np.dot(np.array(query_embedding), np.array(emb)))

            vector_score = sim * strength
            graph_boost = cum_weight * 0.3
            hybrid = W_BM25 * bm25_norm + W_VECTOR * vector_score + graph_boost

            scored[mid] = RecallResult(
                id=mid,
                content=m.content,
                category=m.category,
                importance=m.importance,
                strength=strength,
                similarity=sim,
                bm25=bm25_norm,
                score=hybrid,
            )

    results = sorted(scored.values(), key=lambda r: r.score, reverse=True)[:top_k]

    for r in results:
        if r.similarity >= REINFORCE_SIM:
            db.bump_recall(r.id)
            graph.boost(r.id, amount=0.2, max_depth=1)

    return results
