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


def compute_quality_score(
    importance: float,
    recall_count: int,
    success_count: int = 0,
    failure_count: int = 0,
) -> float:
    """Compute a dynamic quality score for a memory.

    Factors:
    - Base importance (author-assigned weight)
    - Recall frequency (how often agents find this useful)
    - Outcome signal (success/failure ratio from session_outcome)

    Returns a score in [0.0, 1.0].
    """
    # Recall utility: more recalls = more useful, with diminishing returns
    recall_bonus = math.log1p(recall_count) * 0.1

    # Outcome signal: net positive outcomes boost, net negative penalize
    total_outcomes = success_count + failure_count
    if total_outcomes > 0:
        success_ratio = success_count / total_outcomes
        # Scale from -0.15 (all failures) to +0.15 (all successes)
        outcome_modifier = (success_ratio - 0.5) * 0.30
    else:
        outcome_modifier = 0.0

    quality = importance + recall_bonus + outcome_modifier
    return max(0.0, min(1.0, quality))