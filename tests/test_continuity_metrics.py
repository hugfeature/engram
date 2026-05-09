"""Unit tests for Continuity Metrics (v0.13).

Covers all 6 metric computations + evaluate API + handler + tool registration.
"""

import pytest

from engram.continuity import (
    goal_retention,
    action_consistency,
    failure_recall,
    working_set_stability,
    replanning_rate,
    redundant_exploration,
    evaluate,
    evaluate_from_checkpoints,
    ContinuityScore,
)
from engram.db import MemoryDB
from engram.handlers import TOOL_HANDLERS, ARG_MAPPING, handle_evaluate_continuity
from engram.tools import TOOL_SCHEMAS
from engram import checkpoint

FAKE_EMBED = [0.1] * 768


@pytest.fixture
def db(tmp_path):
    return MemoryDB(str(tmp_path / "test.duckdb"), dim=768)


# --- Goal Retention ---

def test_goal_retention_identical():
    assert goal_retention({"goal": "refactor login"}, {"goal": "refactor login"}) == 1.0

def test_goal_retention_different():
    assert goal_retention({"goal": "refactor login"}, {"goal": "fix auth bug"}) == 0.0

def test_goal_retention_both_empty():
    assert goal_retention({}, {}) == 1.0

def test_goal_retention_one_empty():
    assert goal_retention({"goal": "something"}, {}) == 0.0


# --- Action Consistency ---

def test_action_consistency_identical():
    state = {"in_progress": ["step1", "step2"], "preferred_next": ["step3"]}
    assert action_consistency(state, state) == 1.0

def test_action_consistency_partial_overlap():
    before = {"in_progress": ["step1", "step2"], "preferred_next": ["step3"]}
    after = {"in_progress": ["step1", "step4"], "preferred_next": ["step3"]}
    score = action_consistency(before, after)
    assert 0.4 < score < 0.8  # 2/4 overlap = 0.5

def test_action_consistency_no_overlap():
    before = {"in_progress": ["a", "b"]}
    after = {"in_progress": ["c", "d"]}
    assert action_consistency(before, after) == 0.0

def test_action_consistency_both_empty():
    assert action_consistency({}, {}) == 1.0


# --- Failure Recall ---

def test_failure_recall_all_preserved():
    before = {"must_not_redo": [
        {"action": "deploy without tests", "reason": "failed_dont_retry"},
    ]}
    after = {"must_not_redo": [
        {"action": "deploy without tests", "reason": "failed_dont_retry"},
        {"action": "skip linting", "reason": "already_completed"},
    ]}
    assert failure_recall(before, after) == 1.0

def test_failure_recall_none_preserved():
    before = {"must_not_redo": [{"action": "deploy without tests", "reason": "x"}]}
    after = {"must_not_redo": [{"action": "something else", "reason": "y"}]}
    assert failure_recall(before, after) == 0.0

def test_failure_recall_no_before():
    assert failure_recall({}, {"must_not_redo": [{"action": "x"}]}) == 1.0


# --- Working Set Stability ---

def test_working_set_stability_identical():
    ws = {"working_set": {"files": ["a.py", "b.py"], "tools": ["grep"]}}
    assert working_set_stability(ws, ws) == 1.0

def test_working_set_stability_partial():
    before = {"working_set": {"files": ["a.py", "b.py"]}}
    after = {"working_set": {"files": ["a.py", "c.py"]}}
    score = working_set_stability(before, after)
    assert 0.2 < score < 0.5  # 1/3 overlap

def test_working_set_stability_both_empty():
    assert working_set_stability({}, {}) == 1.0


# --- Replanning Rate ---

def test_replanning_rate_no_checkpoints(db):
    assert replanning_rate(db, task_id=999) == 1.0

def test_replanning_rate_all_stable(db):
    task_id = db.create_task("test", goal="g", user_id="default")
    for _ in range(3):
        checkpoint.create_checkpoint(db, task_id, "AUTO_SAVE",
                                     {"goal": "g"}, user_id="default")
    assert replanning_rate(db, task_id) == 1.0

def test_replanning_rate_some_plan_changes(db):
    task_id = db.create_task("test", goal="g", user_id="default")
    checkpoint.create_checkpoint(db, task_id, "AUTO_SAVE", {"goal": "g"})
    checkpoint.create_checkpoint(db, task_id, "PLAN_UPDATE",
                                 {"goal": "g", "in_progress": ["new"]})
    rate = replanning_rate(db, task_id)
    assert rate == 0.5  # 1 of 2 is plan change


# --- Redundant Exploration ---

def test_redundant_exploration_no_violations():
    must_not = [{"action": "deploy without tests"}]
    actions = ["run tests", "deploy with tests"]
    assert redundant_exploration(must_not, actions) == 1.0

def test_redundant_exploration_one_violation():
    must_not = [{"action": "deploy without tests"}]
    actions = ["deploy without tests", "run tests"]
    assert redundant_exploration(must_not, actions) == 0.5

def test_redundant_exploration_no_actions():
    assert redundant_exploration([{"action": "x"}], []) == 1.0

def test_redundant_exploration_no_restrictions():
    assert redundant_exploration([], ["anything"]) == 1.0


# --- evaluate() full pipeline ---

def test_evaluate_perfect_continuity():
    state = {
        "goal": "refactor",
        "in_progress": ["step1"],
        "preferred_next": ["step2"],
        "must_not_redo": [{"action": "bad thing", "reason": "x"}],
        "working_set": {"files": ["a.py"]},
    }
    score = evaluate(state, state)
    assert score.goal_retention == 1.0
    assert score.action_consistency == 1.0
    assert score.failure_recall == 1.0
    assert score.working_set_stability == 1.0
    assert score.composite > 0.9

def test_evaluate_returns_continuity_score_type():
    score = evaluate({}, {})
    assert isinstance(score, ContinuityScore)
    d = score.to_dict()
    assert "composite" in d
    assert len(d) == 7


# --- evaluate_from_checkpoints ---

def test_evaluate_from_checkpoints_needs_two(db):
    task_id = db.create_task("test", goal="g")
    result = evaluate_from_checkpoints(db, task_id)
    assert result is None  # less than 2 checkpoints

def test_evaluate_from_checkpoints_works(db):
    task_id = db.create_task("test", goal="build feature")
    state1 = {"goal": "build feature", "in_progress": ["design"],
              "preferred_next": ["implement"]}
    state2 = {"goal": "build feature", "in_progress": ["implement"],
              "preferred_next": ["test"]}
    checkpoint.create_checkpoint(db, task_id, "AUTO_SAVE", state1)
    checkpoint.create_checkpoint(db, task_id, "PLAN_UPDATE", state2)
    score = evaluate_from_checkpoints(db, task_id)
    assert score is not None
    assert score.goal_retention == 1.0
    assert 0.0 <= score.action_consistency <= 1.0
    assert score.composite > 0.0


# --- Handler ---

def test_handle_evaluate_continuity_no_checkpoints(db, tmp_path):
    from engram.graph import MemoryGraph
    graph = MemoryGraph(str(tmp_path / "g.json"))
    task_id = db.create_task("test", goal="g")
    result = handle_evaluate_continuity(db, graph, task_id=task_id)
    assert result.get("ok") is False  # needs >= 2 checkpoints

def test_handle_evaluate_continuity_success(db, tmp_path):
    from engram.graph import MemoryGraph
    graph = MemoryGraph(str(tmp_path / "g.json"))
    task_id = db.create_task("test", goal="g")
    checkpoint.create_checkpoint(db, task_id, "AUTO_SAVE", {"goal": "g"})
    checkpoint.create_checkpoint(db, task_id, "AUTO_SAVE", {"goal": "g"})
    result = handle_evaluate_continuity(db, graph, task_id=task_id)
    assert result["ok"] is True
    assert "continuity_score" in result
    assert "composite" in result["continuity_score"]


# --- Tool registration ---

def test_evaluate_continuity_registered():
    assert "evaluate_continuity" in TOOL_HANDLERS
    assert "evaluate_continuity" in ARG_MAPPING
    tool_names = {t.name for t in TOOL_SCHEMAS}
    assert "evaluate_continuity" in tool_names
