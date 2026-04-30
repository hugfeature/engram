"""NetworkX graph layer — semantic edges, BFS expansion, chain-safe pruning."""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque

import networkx as nx
import numpy as np

from .config import EDGE_THRESHOLD, EDGE_WEIGHT, MAX_EDGES

log = logging.getLogger("engram.graph")

GRAPH_PATH = os.path.join(os.path.expanduser("~"), ".engram", "graph.json")


class MemoryGraph:
    def __init__(self, graph_path: str = GRAPH_PATH):
        self._path = graph_path
        self._lock = threading.RLock()
        self._dirty = False
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        if os.path.exists(graph_path):
            with open(graph_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._graph: nx.DiGraph = nx.node_link_graph(data, directed=True)
        else:
            pkl_path = graph_path.replace(".json", ".pkl")
            if os.path.exists(pkl_path):
                import pickle
                with open(pkl_path, "rb") as f:
                    self._graph = pickle.load(f)
                self._flush()
                os.remove(pkl_path)
            else:
                self._graph = nx.DiGraph()

    def _mark_dirty(self):
        self._dirty = True

    def _flush(self):
        data = nx.node_link_data(self._graph)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        self._dirty = False

    def _save(self):
        if self._dirty:
            self._flush()

    def flush(self):
        with self._lock:
            self._save()

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

    def index_memory(
        self,
        memory_id: int,
        embedding: list[float],
        all_embeddings: dict[int, list[float]],
        user_id: str = "default",
        importance: float = 0.5,
        category: str = "fact",
    ):
        with self._lock:
            self._graph.add_node(
                memory_id,
                user_id=user_id,
                strength=1.0,
                importance=importance,
                category=category,
            )

            if not all_embeddings:
                self._mark_dirty()
                self._save()
                return

            new_vec = np.array(embedding)
            similarities = []
            for mid, emb in all_embeddings.items():
                if mid == memory_id:
                    continue
                node_data = self._graph.nodes.get(mid, {})
                if node_data.get("user_id", "default") != user_id:
                    continue
                sim = float(np.dot(new_vec, np.array(emb)))
                if sim >= EDGE_THRESHOLD:
                    similarities.append((mid, sim))

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

    def index_memory_incremental(
        self,
        memory_id: int,
        embedding: list[float],
        db,
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

            from .config import EDGE_THRESHOLD as threshold
            candidates = db.search_vector(
                embedding, user_id, top_k=top_k, threshold=threshold
            )

            new_vec = np.array(embedding)
            similarities = []
            for m in candidates:
                if m.id == memory_id:
                    continue
                emb = db.get_embedding(m.id)
                if not emb:
                    continue
                sim = float(np.dot(new_vec, np.array(emb)))
                if sim >= threshold:
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
        max_depth: int = 2,
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
            for neighbor, _ in self.expand([memory_id], max_depth=max_depth):
                if neighbor in self._graph:
                    data = self._graph.nodes[neighbor]
                    edge_w = 1.0
                    if self._graph.has_edge(memory_id, neighbor):
                        edge_w = self._graph[memory_id][neighbor].get("weight", 1.0)
                    boost_val = amount * edge_w
                    data["strength"] = min(1.0, data.get("strength", 0) + boost_val)
            self._mark_dirty()
            self._save()

    def chain_safe_to_prune(self, memory_id: int, threshold: float = 0.05) -> bool:
        with self._lock:
            if memory_id not in self._graph:
                return True
            for neighbor in set(self._graph.successors(memory_id)) | set(
                self._graph.predecessors(memory_id)
            ):
                data = self._graph.nodes.get(neighbor, {})
                if data.get("strength", 0) >= threshold:
                    return False
            return True

    def update_node_strength(self, memory_id: int, strength: float):
        with self._lock:
            if memory_id in self._graph:
                self._graph.nodes[memory_id]["strength"] = strength
                self._mark_dirty()
                self._save()
