"""Deduplication and contradiction resolution for incoming memories."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .config import DEDUP_THRESHOLD, REINFORCE_THRESHOLD

POSITIVE_WORDS = {
    "love", "like", "prefer", "use", "adopt", "enjoy", "want", "choose",
    "recommend", "support", "favor", "embrace",
}
NEGATIVE_WORDS = {
    "hate", "dislike", "avoid", "abandon", "reject", "stop", "drop",
    "refuse", "oppose", "remove", "never",
}
NEGATION_WORDS = {"not", "no", "never", "don't", "doesn't", "didn't", "won't", "can't", "isn't", "aren't"}


class Action(Enum):
    NEW = "new"
    REINFORCE = "reinforce"
    REPLACE = "replace"
    MERGE = "merge"


@dataclass
class Resolution:
    action: Action
    existing_id: int | None = None
    merged_content: str | None = None


def _polarity(text: str) -> int | None:
    words = set(text.lower().split())
    has_neg = bool(words & NEGATION_WORDS)
    has_pos = bool(words & POSITIVE_WORDS)
    has_neg_word = bool(words & NEGATIVE_WORDS)

    if has_pos and not has_neg and not has_neg_word:
        return 1
    if has_neg_word or (has_pos and has_neg):
        return -1
    return None


def _is_contradiction(text_a: str, text_b: str) -> bool:
    pa = _polarity(text_a)
    pb = _polarity(text_b)
    if pa is not None and pb is not None and pa != pb:
        return True
    return False


def _merge_content(existing: str, incoming: str) -> str:
    incoming_words = set(incoming.lower().split()) - set(existing.lower().split())
    if len(incoming_words) < 3:
        return existing
    return f"{existing}; {incoming}"


def resolve(
    new_content: str,
    new_embedding: list[float],
    existing_memories: list[tuple[int, str, list[float]]],
) -> Resolution:
    if not existing_memories:
        return Resolution(action=Action.NEW)

    new_vec = np.array(new_embedding)
    best_sim = 0.0
    best_id = None
    best_content = ""

    for mid, content, emb in existing_memories:
        sim = float(np.dot(new_vec, np.array(emb)))
        if sim > best_sim:
            best_sim = sim
            best_id = mid
            best_content = content

    if best_sim >= REINFORCE_THRESHOLD:
        return Resolution(action=Action.REINFORCE, existing_id=best_id)

    if best_sim >= DEDUP_THRESHOLD:
        if _is_contradiction(best_content, new_content):
            return Resolution(action=Action.REPLACE, existing_id=best_id)
        merged = _merge_content(best_content, new_content)
        if merged != best_content:
            return Resolution(
                action=Action.MERGE,
                existing_id=best_id,
                merged_content=merged,
            )
        return Resolution(action=Action.REINFORCE, existing_id=best_id)

    return Resolution(action=Action.NEW)
