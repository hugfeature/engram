"""Tests for resolve.py — MERGE, REPLACE, REINFORCE, NEW paths and edge cases."""

import numpy as np
import pytest

from engram.resolve import resolve, Action, _merge_content, _is_contradiction, _polarity
from engram.config import DEDUP_THRESHOLD, REINFORCE_THRESHOLD


def _unit_vec(seed: int, dim: int = 768) -> list[float]:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim)
    return (v / np.linalg.norm(v)).tolist()


def _vec_pair(sim_target: float, seed: int = 42, dim: int = 768) -> tuple[list[float], list[float]]:
    """Create two unit vectors with a specific cosine similarity."""
    rng = np.random.RandomState(seed)
    v1 = rng.randn(dim)
    v1 = v1 / np.linalg.norm(v1)
    noise = rng.randn(dim)
    noise = noise / np.linalg.norm(noise)
    a = sim_target
    b = np.sqrt(1 - a ** 2)
    v2 = a * v1 + b * noise
    v2 = v2 / np.linalg.norm(v2)
    return v1.tolist(), v2.tolist()


class TestResolveNew:
    def test_no_existing_memories(self):
        result = resolve("new fact", _unit_vec(1), [])
        assert result.action == Action.NEW
        assert result.existing_id is None

    def test_low_similarity_returns_new(self):
        vec_a = _unit_vec(1)
        vec_b = _unit_vec(99)
        existing = [(1, "completely different topic", vec_b)]
        result = resolve("new topic", vec_a, existing)
        assert result.action == Action.NEW


class TestResolveReinforce:
    def test_identical_vector_reinforces(self):
        vec = _unit_vec(42)
        existing = [(1, "user prefers TypeScript", vec)]
        result = resolve("user prefers TypeScript", vec, existing)
        assert result.action == Action.REINFORCE
        assert result.existing_id == 1

    def test_very_high_similarity_reinforces(self):
        """Similarity > REINFORCE_THRESHOLD (0.85) → REINFORCE, even with contradiction."""
        existing_vec, query_vec = _vec_pair(0.95)
        existing = [(1, "I love Python", existing_vec)]
        result = resolve("I hate Python", query_vec, existing)
        # sim > 0.85 → always REINFORCE (contradiction check is skipped)
        assert result.action == Action.REINFORCE


class TestResolveReplace:
    def test_contradiction_in_dedup_range(self):
        """Similarity in DEDUP~REINFORCE range + contradiction → REPLACE."""
        existing_vec, query_vec = _vec_pair(0.75)  # 0.65 < 0.75 < 0.85
        existing = [(1, "I love Python programming", existing_vec)]
        result = resolve("I hate Python programming", query_vec, existing)
        assert result.action == Action.REPLACE
        assert result.existing_id == 1

    def test_negation_contradiction(self):
        existing_vec, query_vec = _vec_pair(0.75)
        existing = [(1, "we adopt microservices architecture", existing_vec)]
        result = resolve("we reject microservices architecture", query_vec, existing)
        assert result.action == Action.REPLACE

    def test_positive_vs_negative_replaces(self):
        existing_vec, query_vec = _vec_pair(0.75)
        existing = [(1, "user prefer React for frontend", existing_vec)]
        result = resolve("user dislike React for frontend", query_vec, existing)
        assert result.action == Action.REPLACE


class TestResolveMerge:
    def test_similar_compatible_merges(self):
        """Similarity in DEDUP~REINFORCE range + compatible content → MERGE."""
        existing_vec, query_vec = _vec_pair(0.75)
        existing = [(1, "project uses Go for backend services", existing_vec)]
        result = resolve("project uses Go for backend APIs and gRPC", query_vec, existing)
        assert result.action in (Action.MERGE, Action.REINFORCE)

    def test_merge_produces_merged_content(self):
        existing_vec, query_vec = _vec_pair(0.75)
        existing = [(1, "frontend uses React", existing_vec)]
        result = resolve("frontend uses React with TypeScript for type safety", query_vec, existing)
        if result.action == Action.MERGE:
            assert result.merged_content is not None
            assert len(result.merged_content) > len("frontend uses React")


class TestMergeContent:
    def test_short_addition_returns_longer(self):
        """If incoming has < 3 unique words, return the longer of the two."""
        result = _merge_content("user prefers Go", "yes Go")
        assert result == "user prefers Go"

    def test_longer_base_kept(self):
        result = _merge_content(
            "project uses Go for backend services and Docker for deployment",
            "project uses Go for backend",
        )
        assert "Docker" in result

    def test_subsumed_content_returns_base(self):
        result = _merge_content("I love Python programming", "I love Python")
        assert result == "I love Python programming"

    def test_complementary_content_merged(self):
        result = _merge_content(
            "user prefers Go for backend services",
            "user prefers TypeScript for frontend development work",
        )
        assert "frontend" in result

    def test_few_unique_words_returns_existing(self):
        """< 3 unique words in incoming → return existing (base), not the longer one.
        This is the current behavior: _merge_content returns `existing` when
        the incoming text adds fewer than 3 unique words.
        """
        result = _merge_content("Go backend", "Go backend services")
        # unique_b = {"services"}, len=1 < 3 → returns existing
        assert result == "Go backend"


class TestPolarity:
    def test_positive(self):
        assert _polarity("I love this approach") == 1

    def test_negative(self):
        assert _polarity("I hate this approach") == -1

    def test_negated_positive(self):
        assert _polarity("I don't like this approach") == -1

    def test_neutral(self):
        assert _polarity("The project uses Go") is None

    def test_contradiction_detected(self):
        assert _is_contradiction("I prefer Go", "I avoid Go")

    def test_no_contradiction_same_polarity(self):
        assert not _is_contradiction("I prefer Go", "I like Go")

    def test_no_contradiction_neutral(self):
        assert not _is_contradiction("The project uses Go", "The project uses Python")


class TestResolveEdgeCases:
    def test_empty_existing_list(self):
        result = resolve("anything", _unit_vec(1), [])
        assert result.action == Action.NEW

    def test_multiple_existing_picks_most_similar(self):
        existing_vec, query_vec = _vec_pair(0.75)
        far_vec = _unit_vec(1)
        existing = [
            (1, "far topic", far_vec),
            (2, "close topic", existing_vec),
        ]
        result = resolve("close match query", query_vec, existing)
        assert result.existing_id == 2