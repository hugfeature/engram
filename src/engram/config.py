"""Centralized configuration — all tunable thresholds and weights."""

import os


def _float(key: str, default: float, lo: float = 0.0, hi: float = 1.0) -> float:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return max(lo, min(hi, float(val)))
    except (ValueError, TypeError):
        return default


def _int(key: str, default: int, lo: int = 1, hi: int = 1000) -> int:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return max(lo, min(hi, int(val)))
    except (ValueError, TypeError):
        return default


# --- Decay (decay.py) ---
DECAY_RATES = {
    "fact": 0.16,
    "assumption": 0.20,
    "failure": 0.35,
    "strategy": 0.10,
}
PRUNE_THRESHOLD = _float("ENGRAM_PRUNE_THRESHOLD", 0.05)

# --- Resolve (resolve.py) ---
DEDUP_THRESHOLD = _float("ENGRAM_DEDUP_THRESHOLD", 0.65)
REINFORCE_THRESHOLD = _float("ENGRAM_REINFORCE_THRESHOLD", 0.85)

# --- Retrieve (retrieve.py) ---
SIMILARITY_HIGH = _float("ENGRAM_SIM_HIGH", 0.50)
SIMILARITY_LOW = _float("ENGRAM_SIM_LOW", 0.20)
REINFORCE_SIM = _float("ENGRAM_REINFORCE_SIM", 0.75)
W_BM25 = _float("ENGRAM_W_BM25", 0.30)
W_VECTOR = _float("ENGRAM_W_VECTOR", 0.70)
GRAPH_MAX_DEPTH = _int("ENGRAM_GRAPH_DEPTH", 3, lo=1, hi=10)

# --- Graph (graph.py) ---
EDGE_THRESHOLD = _float("ENGRAM_EDGE_THRESHOLD", 0.40)
EDGE_WEIGHT = _float("ENGRAM_EDGE_WEIGHT", 0.50)
MAX_EDGES = _int("ENGRAM_MAX_EDGES", 5, lo=1, hi=50)

# --- Consolidator (consolidator.py) ---
CONSOLIDATE_THRESHOLD = _float("ENGRAM_CONSOLIDATE_THRESHOLD", 0.70)

# --- DB (db.py) ---
DEDUP_SEARCH_THRESHOLD = _float("ENGRAM_DEDUP_SEARCH_THRESHOLD", 0.60)
