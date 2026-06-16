"""CI smoke for the Live continuity bench (v2) — flow + parsing only.

The Live bench calls a real LLM, so its NUMBERS cannot be CI-tested. What
*can* (and must) be guarded is the harness around the LLM:

  1. JSON parsing survives the messy shapes real models emit (fences,
     prose, trailing junk) — this is where Live silently breaks.
  2. The recovery-package renderer reflects the mode (NONE = amnesiac,
     SELECTIVE/FULL carry memories).
  3. The end-to-end dry-run flow (real restore_checkpoint + render + agg)
     runs without touching the API.

Kept API-free: every test here is offline.
"""

import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmark"
sys.path.insert(0, str(BENCH_DIR))

import continuity_bench_live as live  # noqa: E402
import continuity_bench as core  # noqa: E402


# --- JSON parsing robustness (the #1 Live failure mode) ---

@pytest.mark.parametrize("raw,expected", [
    ('["a", "b"]', ["a", "b"]),
    ('```json\n["a", "b"]\n```', ["a", "b"]),
    ('```\n["a"]\n```', ["a"]),
    ('Here are the actions:\n["x", "y"]', ["x", "y"]),
    ('{"verdicts": []}', {"verdicts": []}),
    ('```json\n{"verdicts": [{"action": "z", "redoes_forbidden": true}]}\n```',
     {"verdicts": [{"action": "z", "redoes_forbidden": True}]}),
])
def test_parse_json_loose_handles_real_shapes(raw, expected):
    assert live._parse_json_loose(raw) == expected


def test_parse_json_loose_returns_none_on_garbage():
    assert live._parse_json_loose("totally not json at all") is None


# --- Judge output normalization (GLM returned a bare list → crashed v1) ---

def test_normalize_judge_output_handles_object_shape():
    out = live.normalize_judge_output({"verdicts": [{"redoes_forbidden": True}], "reasoning": "r"})
    assert out["verdicts"] == [{"redoes_forbidden": True}]
    assert out["reasoning"] == "r"


def test_normalize_judge_output_handles_bare_list():
    """GLM-5.1 returns a bare verdict array instead of {"verdicts": [...]}.
    v1 crashed with 'list object has no attribute get' — guard it."""
    out = live.normalize_judge_output([{"redoes_forbidden": True}, {"redoes_forbidden": False}])
    assert isinstance(out["verdicts"], list)
    assert len(out["verdicts"]) == 2


def test_normalize_judge_output_handles_single_verdict_object():
    out = live.normalize_judge_output({"action": "x", "redoes_forbidden": True})
    assert out["verdicts"] == [{"action": "x", "redoes_forbidden": True}]


def test_normalize_judge_output_handles_none():
    out = live.normalize_judge_output(None, "raw garbage")
    assert out["verdicts"] == []
    assert "raw garbage" in out["reasoning"]


# --- Recovery package rendering reflects the mode ---

def test_none_package_is_amnesiac():
    cont = {"goal": "ship X", "completed": ["a"], "must_not_redo": [{"action": "drop db"}]}
    prompt = live.render_recovery_prompt(cont, [], [], "NONE")
    assert "none" in prompt.lower()
    # NONE must NOT leak completed work or must_not_redo into the package.
    assert "drop db" not in prompt
    assert "ship X" in prompt  # goal is the only thing a cold start knows


def test_selective_package_carries_memories_and_failures():
    cont = {"goal": "ship X", "completed": ["a"], "in_progress": ["b"],
            "preferred_next": ["c"], "must_not_redo": [{"action": "drop db"}],
            "must_preserve": ["keep prod"], "working_set": {"files": ["x.py"]}}
    mems = [{"content": "use redis"}]
    fails = [{"content": "dropping db lost data"}]
    prompt = live.render_recovery_prompt(cont, mems, fails, "SELECTIVE")
    assert "drop db" in prompt
    assert "use redis" in prompt
    assert "dropping db lost data" in prompt
    assert "keep prod" in prompt


# --- End-to-end dry-run flow (no API) ---

def test_dry_run_flow_completes_with_real_restore():
    scenarios = core.load_scenarios(["c1_retry_storm"])
    assert scenarios, "c1_retry_storm scenario must exist"
    sc = scenarios[0]

    none_r = live.run_live_one(sc, "NONE", 0, None, None, dry_run=True)
    sel_r = live.run_live_one(sc, "SELECTIVE", 0, None, None, dry_run=True)

    # Real restore really ran for SELECTIVE → memories recalled; NONE got none.
    assert none_r["observed"]["related_memories"] == 0
    assert sel_r["observed"]["related_memories"] > 0
    assert sel_r["observed"]["real_restore"] is True
    # Dry-run still produces a well-formed record.
    assert "redundant_exploration" in sel_r
    assert "constraint_violation" in sel_r


def test_judge_prompt_includes_ground_truth():
    p = live.build_judge_prompt(
        ["do thing"], ["forbidden thing"], ["preserve this"],
    )
    assert "forbidden thing" in p
    assert "preserve this" in p
    assert "do thing" in p
