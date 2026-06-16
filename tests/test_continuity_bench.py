"""CI smoke + invariant guard for the Continuity Benchmark (Core).

This is NOT the full benchmark run (that lives in benchmark/continuity_bench.py
and is for reporting). These tests guard the two things that must never
silently break:

  1. Every scenario JSON is schema-valid and loadable.
  2. The core recovery-quality invariant holds: SELECTIVE >= FULL > NONE.
     If an engram change ever makes a full-history restore beat the
     selective restore, or makes NONE competitive, that is a regression in
     recovery quality and this test must catch it.

Kept fast: 1 run per (scenario, mode), default seed.
"""

import json
import sys
from pathlib import Path

import pytest

# Make the benchmark module importable.
BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmark"
sys.path.insert(0, str(BENCH_DIR))

import continuity_bench as cb  # noqa: E402


REQUIRED_TOP_KEYS = {
    "scenario_id", "axis", "description", "stresses",
    "interrupt_reason", "pre_interrupt_state", "ground_truth", "agent_replay",
}
VALID_AXES = {"A_interruption", "B_state_drift", "C_failure_recall"}


@pytest.fixture(scope="module")
def scenarios():
    return cb.load_scenarios(None)


# --- Schema validity ---

def test_scenarios_present(scenarios):
    assert len(scenarios) >= 20, f"expected >=20 scenarios, found {len(scenarios)}"


def test_every_scenario_schema_valid(scenarios):
    seen_ids = set()
    for sc in scenarios:
        sid = sc.get("scenario_id", "<missing>")
        missing = REQUIRED_TOP_KEYS - sc.keys()
        assert not missing, f"{sid}: missing keys {missing}"
        assert sc["axis"] in VALID_AXES, f"{sid}: bad axis {sc['axis']}"
        assert sid not in seen_ids, f"duplicate scenario_id {sid}"
        seen_ids.add(sid)
        # agent_replay must define all three modes.
        for mode in cb.MODES:
            assert mode in sc["agent_replay"], f"{sid}: agent_replay missing {mode}"
        # ground_truth must declare forbidden_actions (drives redundant metric).
        assert "forbidden_actions" in sc["ground_truth"], f"{sid}: no forbidden_actions"


def test_axis_coverage(scenarios):
    """All three axes must be represented (the bench's design backbone)."""
    axes = {sc["axis"] for sc in scenarios}
    assert axes == VALID_AXES, f"axis coverage gap: {VALID_AXES - axes}"


# --- Determinism ---

def test_runs_are_deterministic(scenarios):
    """Same scenario+mode run twice must give identical composite (std=0)."""
    sc = scenarios[0]
    r1 = cb.run_scenario_mode(sc, "SELECTIVE", 0)
    r2 = cb.run_scenario_mode(sc, "SELECTIVE", 1)
    assert r1["composite"] == r2["composite"]
    assert r1["raw"] == r2["raw"]


# --- The core invariant: SELECTIVE >= FULL > NONE ---

def _mean_composite(scenarios, mode):
    vals = [cb.run_scenario_mode(sc, mode, 0)["composite"] for sc in scenarios]
    return sum(vals) / len(vals)


def test_recovery_quality_invariant(scenarios):
    none_c = _mean_composite(scenarios, "NONE")
    sel_c = _mean_composite(scenarios, "SELECTIVE")
    full_c = _mean_composite(scenarios, "FULL")

    # SELECTIVE must beat FULL (context-pollution thesis) — strict.
    assert sel_c > full_c, f"SELECTIVE ({sel_c:.3f}) should beat FULL ({full_c:.3f})"
    # Both real-restore modes must crush the empty baseline.
    assert full_c > none_c, f"FULL ({full_c:.3f}) should beat NONE ({none_c:.3f})"
    # NONE should be clearly weak (sanity floor).
    assert none_c < 0.5, f"NONE ({none_c:.3f}) unexpectedly high — baseline leaked context?"


def test_selective_recalls_failures_none_does_not(scenarios):
    """Real-restore signal: SELECTIVE recalls memories, NONE recalls nothing."""
    # Pick a C-axis scenario (failure-heavy).
    c_scen = next(sc for sc in scenarios if sc["axis"] == "C_failure_recall")
    none_r = cb.run_scenario_mode(c_scen, "NONE", 0)
    sel_r = cb.run_scenario_mode(c_scen, "SELECTIVE", 0)
    assert none_r["observed"]["related_memories"] == 0
    assert sel_r["observed"]["related_memories"] > 0
    assert sel_r["observed"]["real_restore"] is True


def test_redundant_exploration_is_the_discriminator(scenarios):
    """On an A-axis scenario, structural metrics tie across modes; only
    redundant_exploration should separate SELECTIVE from FULL."""
    a_scen = next(sc for sc in scenarios if sc["axis"] == "A_interruption")
    sel = cb.run_scenario_mode(a_scen, "SELECTIVE", 0)["raw"]
    full = cb.run_scenario_mode(a_scen, "FULL", 0)["raw"]
    # Structural metrics identical (build_continuation runs before mode gate).
    assert sel["goal_preservation"] == full["goal_preservation"]
    assert sel["completed_preservation"] == full["completed_preservation"]
    assert sel["working_set_overlap"] == full["working_set_overlap"]
    # Redundant is where they diverge.
    assert sel["redundant_exploration"] >= full["redundant_exploration"]
