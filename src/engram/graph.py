"""NetworkX graph layer — semantic edges, BFS expansion, chain-safe pruning."""

from __future__ import annotations

import atexit
import fcntl
import json
import logging
import os
import threading
import time
from collections import deque
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from .config import EDGE_THRESHOLD, EDGE_WEIGHT, MAX_EDGES, GRAPH_MAX_DEPTH, PRUNE_THRESHOLD

if TYPE_CHECKING:
    from .db import MemoryDB

log = logging.getLogger("engram.graph")

GRAPH_PATH = os.path.join(os.path.expanduser("~"), ".engram", "graph.json")


class MemoryGraph:
    _FLUSH_INTERVAL = 2.0

    def __init__(self, graph_path: str = GRAPH_PATH):
        self._path = graph_path
        self._lock = threading.RLock()
        self._dirty = False
        self._last_flush_time = 0.0
        self._pkl_pending_delete: str | None = None
        atexit.register(self.flush)
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        self._graph: nx.DiGraph = self._load_graph(graph_path)
        if self._dirty:
            self._flush()

    def _load_graph(self, graph_path: str) -> nx.DiGraph:
        if os.path.exists(graph_path):
            try:
                with open(graph_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return nx.node_link_graph(data, directed=True)
            except Exception as e:
                log.error("Failed to load graph.json: %s — starting empty", e)
                bak = graph_path + ".corrupt"
                try:
                    os.replace(graph_path, bak)
                except OSError:
                    pass
                return nx.DiGraph()

        pkl_path = graph_path.replace(".json", ".pkl")
        if os.path.exists(pkl_path):
            try:
                import pickle
                with open(pkl_path, "rb") as f:
                    g = pickle.load(f)
                self._dirty = True
                # Mark pkl for deletion — will be removed after successful json flush
                self._pkl_pending_delete = pkl_path
                return g
            except Exception as e:
                log.error("Failed to load graph.pkl: %s — starting empty", e)
                return nx.DiGraph()

        return nx.DiGraph()

    def _mark_dirty(self):
        self._dirty = True

    @property
    def _batch_active(self) -> bool:
        return getattr(self, '_in_batch', False)

    def batch_mode(self):
        """Context manager that suppresses auto-save during bulk operations.

        Usage:
            with graph.batch_mode():
                graph.index_memory_incremental(...)
                graph.remove_node(...)
            # single flush happens on exit
        """
        return _BatchContext(self)

    def _flush(self):
        # Guard: skip flush if parent directory was already cleaned up (e.g. tmpdir)
        parent_dir = os.path.dirname(self._path)
        if parent_dir and not os.path.exists(parent_dir):
            self._dirty = False
            return

        tmp_path = self._path + ".tmp"
        lock_path = self._path + ".lock"
        lock_fd = None
        try:
            data = nx.node_link_data(self._graph)
            # Acquire file lock to prevent concurrent writes from multiple processes
            lock_fd = open(lock_path, "w")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp_path, self._path)
            self._dirty = False
            # Delete old pickle file only after successful json write
            if self._pkl_pending_delete:
                try:
                    os.remove(self._pkl_pending_delete)
                except OSError:
                    pass
                self._pkl_pending_delete = None
        except Exception as e:
            # Clean up temp file on failure
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            log.error("Failed to flush graph to %s: %s (will retry)", self._path, e)
        finally:
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
                except OSError:
                    pass

    def _save(self):
        if not self._dirty:
            return
        if self._batch_active:
            return
        now = time.monotonic()
        if now - self._last_flush_time < self._FLUSH_INTERVAL:
            return
        self._flush()
        self._last_flush_time = now

    def flush(self):
        with self._lock:
            if self._dirty:
                self._flush()
                self._last_flush_time = time.monotonic()

    def upsert_node(
        self,
        memory_id: int,
        user_id: str = "default",
        strength: float = 1.0,
        importance: float = 0.5,
        category: str = "fact",
    ):
        with self._lock:
            self._graph.add_node(
                memory_id,
                user_id=user_id,
                strength=strength,
                importance=importance,
                category=category,
            )
            self._mark_dirty()
            self._save()

    def remove_node(self, memory_id: int):
        with self._lock:
            if memory_id in self._graph:
                self._graph.remove_node(memory_id)
                self._mark_dirty()
                self._save()

    def index_memory_incremental(
        self,
        memory_id: int,
        embedding: list[float],
        db: "MemoryDB",
        user_id: str = "default",
        importance: float = 0.5,
        category: str = "fact",
        top_k: int = 20,
    ):
        """Build graph edges using DB vector search instead of all_embeddings dict."""
        with self._lock:
            self._graph.add_node(
                memory_id,
                user_id=user_id,
                strength=1.0,
                importance=importance,
                category=category,
            )

            candidates = db.search_vector(
                embedding, user_id, top_k=top_k, threshold=EDGE_THRESHOLD
            )

            candidate_ids = [m.id for m in candidates if m.id != memory_id]
            emb_batch = db.get_embeddings_batch(candidate_ids) if candidate_ids else {}

            new_vec = np.array(embedding)
            similarities = []
            for m in candidates:
                if m.id == memory_id:
                    continue
                emb = emb_batch.get(m.id)
                if not emb:
                    continue
                sim = float(np.dot(new_vec, np.array(emb)))
                if sim >= EDGE_THRESHOLD:
                    similarities.append((m.id, sim))

            similarities.sort(key=lambda x: x[1], reverse=True)
            for mid, sim in similarities[:MAX_EDGES]:
                weight = sim * EDGE_WEIGHT
                if self._graph.has_edge(memory_id, mid):
                    old_w = self._graph[memory_id][mid].get("weight", 0)
                    weight = min(1.0, old_w + weight * 0.1)
                self._graph.add_edge(
                    memory_id, mid, relation="semantic", weight=weight
                )
                self._graph.add_edge(
                    mid, memory_id, relation="semantic", weight=weight
                )

            self._mark_dirty()
            self._save()

    def expand(
        self,
        seed_ids: list[int],
        max_depth: int = GRAPH_MAX_DEPTH,
        user_id: str | None = None,
    ) -> list[tuple[int, float]]:
        with self._lock:
            visited = set(seed_ids)
            queue: deque[tuple[int, float, int]] = deque()
            for sid in seed_ids:
                if sid in self._graph:
                    queue.append((sid, 1.0, 0))

            neighbors: list[tuple[int, float]] = []
            while queue:
                node, cum_weight, depth = queue.popleft()
                if depth >= max_depth:
                    continue
                for succ in set(self._graph.successors(node)) | set(
                    self._graph.predecessors(node)
                ):
                    if succ in visited:
                        continue
                    if user_id is not None:
                        node_data = self._graph.nodes.get(succ, {})
                        if node_data.get("user_id", "default") != user_id:
                            continue
                    visited.add(succ)
                    edge_w = self._graph[node].get(succ, {}).get("weight", 0)
                    if edge_w == 0 and self._graph.has_edge(succ, node):
                        edge_w = self._graph[succ][node].get("weight", 0)
                    new_cum = cum_weight * edge_w
                    neighbors.append((succ, new_cum))
                    queue.append((succ, new_cum, depth + 1))

            neighbors.sort(key=lambda x: x[1], reverse=True)
            return neighbors

    def boost(self, memory_id: int, amount: float = 0.2, max_depth: int = 1):
        with self._lock:
            if memory_id not in self._graph:
                return
            boosted: set[int] = set()
            for neighbor, _ in self.expand([memory_id], max_depth=max_depth):
                if neighbor in boosted or neighbor not in self._graph:
                    continue
                boosted.add(neighbor)
                data = self._graph.nodes[neighbor]
                edge_w = 1.0
                if self._graph.has_edge(memory_id, neighbor):
                    edge_w = self._graph[memory_id][neighbor].get("weight", 1.0)
                boost_val = amount * edge_w
                data["strength"] = min(1.0, data.get("strength", 0) + boost_val)
            self._mark_dirty()
            self._save()

    def chain_safe_to_prune(self, memory_id: int, threshold: float = PRUNE_THRESHOLD,
                            user_id: str | None = None) -> bool:
        with self._lock:
            if memory_id not in self._graph:
                return True
            for neighbor in set(self._graph.successors(memory_id)) | set(
                self._graph.predecessors(memory_id)
            ):
                data = self._graph.nodes.get(neighbor, {})
                if user_id is not None and data.get("user_id", "default") != user_id:
                    continue
                if data.get("strength", 0) >= threshold:
                    return False
            return True

    def update_node_strength(self, memory_id: int, strength: float):
        with self._lock:
            if memory_id in self._graph:
                self._graph.nodes[memory_id]["strength"] = strength
                self._mark_dirty()
                self._save()


class _BatchContext:
    """Suppresses per-operation auto-save; flushes once on exit."""

    def __init__(self, graph: MemoryGraph):
        self._graph = graph

    def __enter__(self):
        self._graph._in_batch = True
        return self._graph

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._graph._in_batch = False
        try:
            self._graph.flush()
        except Exception as flush_err:
            log.error("Batch flush failed: %s", flush_err)
        return False
