"""Auto-consolidation — find similar memory clusters and merge them."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from itertools import combinations

import numpy as np

from .db import MemoryDB
from .embedding import embed
from .graph import MemoryGraph
from .config import CONSOLIDATE_THRESHOLD

log = logging.getLogger("engram.consolidator")


def _merge_pair(content_a: str, content_b: str) -> str:
    words_a = set(content_a.lower().split())
    words_b = set(content_b.lower().split())
    unique_b = words_b - words_a

    if len(unique_b) < 3:
        return max(content_a, content_b, key=len)

    if len(content_a) >= len(content_b):
        base, extra = content_a, content_b
    else:
        base, extra = content_b, content_a

    if extra.lower() in base.lower():
        return base
    return f"{base}; {extra}"


def _find_clusters(
    ids: list[int],
    embeddings: dict[int, list[float]],
    threshold: float,
) -> list[list[int]]:
    vecs = {mid: np.array(emb) for mid, emb in embeddings.items() if mid in ids}
    visited: set[int] = set()
    clusters: list[list[int]] = []

    for a, b in combinations(vecs.keys(), 2):
        sim = float(np.dot(vecs[a], vecs[b]))
        if sim < threshold:
            continue
        placed = False
        for cluster in clusters:
            if a in cluster or b in cluster:
                cluster_set = set(cluster)
                cluster_set.add(a)
                cluster_set.add(b)
                cluster.clear()
                cluster.extend(cluster_set)
                placed = True
                break
        if not placed:
            clusters.append([a, b])

    return [c for c in clusters if len(c) >= 2]


def run_consolidate(
    db: MemoryDB,
    graph: MemoryGraph,
    user_id: str = "default",
) -> list[dict]:
    memories = db.get_all(user_id)
    if len(memories) < 2:
        return []

    embeddings: dict[int, list[float]] = {}
    content_map: dict[int, str] = {}
    meta_map: dict[int, dict] = {}

    for m in memories:
        emb = db.get_embedding(m.id)
        if emb:
            embeddings[m.id] = emb
            content_map[m.id] = m.content
            meta_map[m.id] = {
                "importance": m.importance,
                "category": m.category,
                "recall_count": m.recall_count,
            }

    ids = list(embeddings.keys())
    clusters = _find_clusters(ids, embeddings, CONSOLIDATE_THRESHOLD)

    results = []
    for cluster in clusters:
        sorted_ids = sorted(
            cluster,
            key=lambda mid: (
                meta_map[mid]["importance"],
                meta_map[mid]["recall_count"],
            ),
            reverse=True,
        )
        keep_id = sorted_ids[0]
        remove_ids = sorted_ids[1:]

        merged_content = content_map[keep_id]
        for rid in remove_ids:
            merged_content = _merge_pair(merged_content, content_map[rid])

        best_importance = max(meta_map[mid]["importance"] for mid in cluster)
        merged_emb = embed(merged_content)
        db.update(keep_id, merged_content, merged_emb, best_importance)

        all_emb = {mid: emb for mid, emb in embeddings.items() if mid not in remove_ids}
        all_emb[keep_id] = merged_emb
        graph.index_memory(
            keep_id, merged_emb, all_emb,
            user_id, best_importance, meta_map[keep_id]["category"],
        )

        for rid in remove_ids:
            db.delete(rid)
            graph.remove_node(rid)

        results.append({
            "kept": keep_id,
            "removed": remove_ids,
            "content": merged_content,
        })
        log.info(
            "Consolidated cluster: kept=%d, removed=%s, content=%s",
            keep_id, remove_ids, merged_content[:80],
        )

    return results
