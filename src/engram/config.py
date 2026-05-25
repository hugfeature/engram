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

# --- Drift Nudge (drift.py) ---
DRIFT_NUDGE_THRESHOLD = _float("ENGRAM_DRIFT_NUDGE_THRESHOLD", 0.7)
DRIFT_NUDGE_ENABLED = os.environ.get("ENGRAM_DRIFT_NUDGE", "1") != "0"

# --- Thrashing Circuit Breaker (drift.py) ---
# Triggers when same tool is called N+ times consecutively without progress
THRASHING_ENABLED = os.environ.get("ENGRAM_THRASHING_BREAKER", "1") != "0"
THRASHING_THRESHOLD = _int("ENGRAM_THRASHING_THRESHOLD", 5, lo=3, hi=50)
# Cooldown: don't fire again for same tool within N calls after last nudge
THRASHING_COOLDOWN = _int("ENGRAM_THRASHING_COOLDOWN", 10, lo=3, hi=100)

# --- Adaptive Checkpoint (checkpoint.py) ---
ADAPTIVE_CHECKPOINT_ENABLED = os.environ.get("ENGRAM_ADAPTIVE_CHECKPOINT", "1") != "0"
# If auto_save restore rate < this threshold over last 7 days, double the interval
ADAPTIVE_LOW_RESTORE_RATE = _float("ENGRAM_ADAPTIVE_LOW_RESTORE_RATE", 0.10)
# Maximum auto_save interval (seconds) after adaptive expansion
ADAPTIVE_MAX_INTERVAL_SECONDS = _int("ENGRAM_ADAPTIVE_MAX_INTERVAL", 600, lo=300, hi=3600)
