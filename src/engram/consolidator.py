"""Auto-consolidation — find similar memory clusters and merge them."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
import numpy as np

from .db import MemoryDB
from .embedding import embed
from .graph import MemoryGraph
from .config import CONSOLIDATE_THRESHOLD

log = logging.getLogger("engram.consolidator")

_CJK_PATTERN = re.compile(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]')


def _has_cjk(text: str) -> bool:
    return bool(_CJK_PATTERN.search(text))


def _merge_pair(content_a: str, content_b: str) -> str:
    # CJK text: whitespace splitting is meaningless, use character-level diff
    if _has_cjk(content_a) or _has_cjk(content_b):
        chars_b = set(content_b) - set(content_a)
        if len(chars_b) < 2:
            return max(content_a, content_b, key=len)
    else:
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
    id_list = [mid for mid in ids if mid in embeddings]
    if len(id_list) < 2:
        return []

    mat = np.array([embeddings[mid] for mid in id_list])
    sims = mat @ mat.T

    parent: dict[int, int] = {mid: mid for mid in id_list}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    rows, cols = np.where(np.triu(sims >= threshold, k=1))
    for i, j in zip(rows, cols):
        union(id_list[i], id_list[j])

    groups: dict[int, list[int]] = {}
    for mid in id_list:
        root = find(mid)
        groups.setdefault(root, []).append(mid)

    return [members for members in groups.values() if len(members) >= 2]


MAX_CONSOLIDATE_BATCH = 500


# Structured memory types that must not be merged/consolidated
_PROTECTED_TYPES = {"handoff", "failure", "progress", "session_failure"}

def run_consolidate(
    db: MemoryDB,
    graph: MemoryGraph,
    user_id: str = "default",
) -> list[dict]:
    memories = db.get_all(user_id)
    if len(memories) < 2:
        return []

    # Exclude structured memories from consolidation — they carry metadata
    # that would be lost or corrupted by merging
    memories = [
        m for m in memories
        if (m.metadata or {}).get("type") not in _PROTECTED_TYPES
    ]
    if len(memories) < 2:
        return []

    if len(memories) > MAX_CONSOLIDATE_BATCH:
        memories.sort(key=lambda m: m.last_accessed_at, reverse=True)
        memories = memories[:MAX_CONSOLIDATE_BATCH]
        log.info("Consolidation limited to %d most recent memories", MAX_CONSOLIDATE_BATCH)

    embeddings: dict[int, list[float]] = {}
    content_map: dict[int, str] = {}
    meta_map: dict[int, dict] = {}

    all_ids = [m.id for m in memories]
    emb_batch = db.get_embeddings_batch(all_ids)

    for m in memories:
        emb = emb_batch.get(m.id)
        if emb:
            embeddings[m.id] = emb
            content_map[m.id] = m.content
            meta_map[m.id] = {
                "importance": m.importance,
                "category": m.category,
                "recall_count": m.recall_count,
            }
        else:
            log.debug("Skipping memory %d: no embedding", m.id)

    ids = list(embeddings.keys())
    clusters = _find_clusters(ids, embeddings, CONSOLIDATE_THRESHOLD)

    results = []
    with graph.batch_mode():
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
            try:
                merged_emb = embed(merged_content)
            except Exception as e:
                log.error("Embedding failed during consolidation: %s", e)
                continue
            db.update(keep_id, merged_content, merged_emb, best_importance)

            graph.index_memory_incremental(
                keep_id, merged_emb, db,
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
