"""Ebbinghaus forgetting curve — strength decay with importance modulation."""

import math

from .config import DECAY_RATES, PRUNE_THRESHOLD  # noqa: F401 (PRUNE_THRESHOLD re-exported)


def compute_strength(
    category: str,
    importance: float,
    days_since_access: float,
    recall_count: int = 0,
) -> float:
    base_lambda = DECAY_RATES.get(category, DECAY_RATES["fact"])
    effective_lambda = base_lambda * (1 - importance * 0.8)
    strength = (
        importance
        * math.exp(-effective_lambda * days_since_access)
        * (1 + recall_count * 0.2)
    )
    return min(strength, 1.0)
