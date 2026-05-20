"""Unit tests for checkpoint.py + Checkpoint v2 integration with handlers."""

import pytest

from engram.db import MemoryDB
from engram.graph import MemoryGraph
from engram import checkpoint as ckpt
from engram.checkpoint import (
    REASON_MANUAL_HANDOFF, REASON_PLAN_UPDATE, REASON_FAILURE,
    REASON_WORKING_SET_SHIFT, REASON_AUTO_SAVE,
    KIND_HANDOFF, KIND_AUTO,
    NEGATIVE_REASON_ALREADY_COMPLETED, NEGATIVE_REASON_SIDE_EFFECT_EMITTED,
    create_checkpoint, get_checkpoint, list_checkpoints,
    build_continuation, compute_shallow_diff, compute_confidence,
    detect_plan_changed, detect_working_set_shifted,
    should_create_auto_checkpoint,
    jaccard_similarity, derive_failure_signature, normalize_must_not_redo,
    recency_at,
)


@pytest.fixture
def db(tmp_path):
    return MemoryDB(str(tmp_path / "ckpt.duckdb"), dim=768)


@pytest.fixture
def task(db):
    """Create a default task and return its id."""
    return db.create_task(name="t", goal="goal-x")


# ============================================================
# Helpers
# ============================================================

class TestJaccard:
    def test_both_empty(self):
        assert jaccard_similarity(set(), set()) == 1.0

    def test_one_empty(self):
        assert jaccard_similarity({"a"}, set()) == 0.0
        assert jaccard_similarity(set(), {"a"}) == 0.0

    def test_identical(self):
        assert jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0

    def test_partial(self):
        # 交集 1，并集 3 → 1/3
        assert abs(jaccard_similarity({"a", "b"}, {"a", "c"}) - 1/3) < 1e-9


class TestFailureSignature:
    def test_basic(self):
        assert derive_failure_signature("foo", "TimeoutError") == "foo:timeouterror"

    def test_none_inputs(self):
        assert derive_failure_signature(None, None) == "unknown:unknown"

    def test_strip_and_lower(self):
        assert derive_failure_signature("  Bar  ", "  ConnErr ") == "bar:connerr"


class TestNormalizeMustNotRedo:
    def test_empty(self):
        assert normalize_must_not_redo(None) == []
        assert normalize_must_not_redo([]) == []

    def test_invalid_type(self):
        assert normalize_must_not_redo("not a list") == []

    def test_skip_missing_action(self):
        assert normalize_must_not_redo([{"reason": "already_completed"}]) == []

    def test_unknown_reason_falls_back(self):
        out = normalize_must_not_redo([{"action": "act", "reason": "made_up"}])
        assert len(out) == 1
        assert out[0]["reason"] == NEGATIVE_REASON_ALREADY_COMPLETED

    def test_full_object(self):
        out = normalize_must_not_redo([{
            "action": "create_pr",
            "reason": "side_effect_emitted",
            "scope": "session",
            "expires_at": "2026-12-31T00:00:00Z",
            "idempotency_key": "pr-42",
        }])
        assert out[0]["action"] == "create_pr"
        assert out[0]["reason"] == NEGATIVE_REASON_SIDE_EFFECT_EMITTED
        assert out[0]["scope"] == "session"
        assert out[0]["idempotency_key"] == "pr-42"


# ============================================================
# Event detection
# ============================================================

class TestPlanChanged:
    def test_first_checkpoint_with_plan(self):
        assert detect_plan_changed(None, {"in_progress": ["a"]}) is True

    def test_first_checkpoint_empty(self):
        assert detect_plan_changed(None, {"in_progress": []}) is False

    def test_unchanged(self):
        prev = {"in_progress": ["a", "b"]}
        new = {"in_progress": ["a", "b"]}
        assert detect_plan_changed(prev, new) is False

    def test_significant_change(self):
        prev = {"in_progress": ["a", "b"]}
        new = {"in_progress": ["c", "d"]}  # Jaccard = 0
        assert detect_plan_changed(prev, new) is True

    def test_minor_change_below_threshold(self):
        # 4 共同 / 5 并集 = 0.8 > 0.7 → 不算变化
        prev = {"in_progress": ["a", "b", "c", "d"]}
        new = {"in_progress": ["a", "b", "c", "d", "e"]}
        assert detect_plan_changed(prev, new) is False


class TestWorkingSetShifted:
    def test_first_with_files(self):
        assert detect_working_set_shifted(None, {"working_set": {"files": ["a.py"]}}) is True

    def test_first_empty(self):
        assert detect_working_set_shifted(None, {"working_set": {}}) is False

    def test_no_shift(self):
        prev = {"working_set": {"files": ["a.py", "b.py"]}}
        new = {"working_set": {"files": ["a.py", "b.py"]}}
        assert detect_working_set_shifted(prev, new) is False

    def test_significant_shift(self):
        prev = {"working_set": {"files": ["a.py"], "tools": ["grep"]}}
        new = {"working_set": {"files": ["x.py"], "tools": ["sed"]}}
        assert detect_working_set_shifted(prev, new) is True


# ============================================================
# Diff
# ============================================================

class TestShallowDiff:
    def test_empty_to_empty(self):
        d = compute_shallow_diff({}, {})
        assert d == {"changed_fields": {}}

    def test_field_added(self):
        d = compute_shallow_diff({}, {"goal": "g1"})
        assert "goal" in d["changed_fields"]
        assert d["changed_fields"]["goal"]["new"] == "g1"
        assert d["changed_fields"]["goal"]["old"] is None

    def test_field_changed(self):
        d = compute_shallow_diff({"goal": "old"}, {"goal": "new"})
        assert d["changed_fields"]["goal"] == {"old": "old", "new": "new"}

    def test_unchanged_skipped(self):
        d = compute_shallow_diff({"goal": "g"}, {"goal": "g"})
        assert d["changed_fields"] == {}


# ============================================================
# Confidence
# ============================================================

class TestConfidence:
    def test_empty_state_low_confidence(self, db, task):
        conf, breakdown = compute_confidence(db, task, {})
        assert breakdown["state_completeness"] == 0.0
        # recency 1.0 + verification 0.7 + drift 1.0 → 仍非零
        assert 0 < conf < 1

    def test_full_state_high_confidence(self, db, task):
        state = {
            "goal": "g", "completed": ["a"], "in_progress": ["b"],
            "preferred_next": ["c"], "must_not_redo": [{"action": "x"}],
        }
        conf, breakdown = compute_confidence(db, task, state)
        assert breakdown["state_completeness"] == 1.0
        assert conf > 0.85

    def test_drift_penalty(self, db, task):
        # 写 5 个 PLAN_UPDATE checkpoint，drift_signals 应被压低
        for i in range(5):
            create_checkpoint(db, task, REASON_PLAN_UPDATE,
                              {"in_progress": [f"step-{i}"]})
        _, breakdown = compute_confidence(db, task, {"goal": "g"})
        # 5 个里 5 个是 drift → 1 - 5/5 * 0.5 = 0.5
        assert breakdown["drift_signals"] == 0.5


class TestRecencyAt:
    def test_now(self):
        from datetime import datetime, timezone
        assert recency_at(datetime.now(timezone.utc)) == 1.0

    def test_24h_half_life(self):
        from datetime import datetime, timezone, timedelta
        ago = datetime.now(timezone.utc) - timedelta(hours=24)
        # 24h 后 recency ≈ 0.5
        assert 0.45 < recency_at(ago) < 0.55

    def test_none(self):
        assert recency_at(None) == 0.0


# ============================================================
# create / get / list
# ============================================================

class TestCreateCheckpoint:
    def test_first_version_is_1(self, db, task):
        r = create_checkpoint(db, task, REASON_MANUAL_HANDOFF, {"goal": "g"})
        assert r["version"] == 1
        assert r["kind"] == KIND_HANDOFF
        assert r["reason"] == REASON_MANUAL_HANDOFF
        assert 0 < r["continuation_confidence"] <= 1.0

    def test_version_monotonic(self, db, task):
        v1 = create_checkpoint(db, task, REASON_MANUAL_HANDOFF, {"goal": "g"})["version"]
        v2 = create_checkpoint(db, task, REASON_AUTO_SAVE, {"goal": "g"})["version"]
        v3 = create_checkpoint(db, task, REASON_FAILURE, {"goal": "g"})["version"]
        assert (v1, v2, v3) == (1, 2, 3)

    def test_kind_is_handoff_only_for_manual(self, db, task):
        r1 = create_checkpoint(db, task, REASON_MANUAL_HANDOFF, {})
        r2 = create_checkpoint(db, task, REASON_AUTO_SAVE, {})
        r3 = create_checkpoint(db, task, REASON_FAILURE, {})
        assert r1["kind"] == KIND_HANDOFF
        assert r2["kind"] == KIND_AUTO
        assert r3["kind"] == KIND_AUTO

    def test_invalid_reason_rejected(self, db, task):
        with pytest.raises(ValueError):
            create_checkpoint(db, task, "BOGUS", {})

    def test_must_not_redo_normalized(self, db, task):
        create_checkpoint(db, task, REASON_MANUAL_HANDOFF, {
            "must_not_redo": [
                {"action": "scan", "reason": "already_completed"},
                {"reason": "missing_action"},  # 应被丢弃
                "not a dict",  # 应被丢弃
            ],
        })
        c = get_checkpoint(db, task)
        assert len(c["state"]["must_not_redo"]) == 1
        assert c["state"]["must_not_redo"][0]["action"] == "scan"

    def test_state_diff_recorded(self, db, task):
        create_checkpoint(db, task, REASON_MANUAL_HANDOFF, {"goal": "g1"})
        create_checkpoint(db, task, REASON_AUTO_SAVE, {"goal": "g2"})
        c = get_checkpoint(db, task)
        diff = c["state_diff"]
        assert "goal" in diff["changed_fields"]
        assert diff["changed_fields"]["goal"] == {"old": "g1", "new": "g2"}

    def test_failure_signature_persisted(self, db, task):
        create_checkpoint(
            db, task, REASON_FAILURE, {"goal": "g"},
            failure_signature="db:timeout",
        )
        c = get_checkpoint(db, task)
        assert c["failure_signature"] == "db:timeout"

    def test_tasks_cache_updated(self, db, task):
        create_checkpoint(db, task, REASON_MANUAL_HANDOFF, {})
        create_checkpoint(db, task, REASON_AUTO_SAVE, {})
        # v0.18: checkpoint cache now lives in SQLite Tier 2
        if db._state_store:
            ver, count = db._state_store.get_task_checkpoint_cache(task)
            assert (ver, count) == (2, 2)
        else:
            row = db.conn.execute(
                "SELECT latest_checkpoint_version, checkpoint_count FROM tasks WHERE id = ?",
                [task],
            ).fetchone()
            assert row == (2, 2)

    def test_unique_per_task(self, db):
        # 不同 task 各自 version 独立
        t1 = db.create_task(name="t1")
        t2 = db.create_task(name="t2")
        r1 = create_checkpoint(db, t1, REASON_MANUAL_HANDOFF, {})
        r2 = create_checkpoint(db, t2, REASON_MANUAL_HANDOFF, {})
        assert r1["version"] == 1
        assert r2["version"] == 1


class TestGetCheckpoint:
    def test_no_checkpoint_returns_none(self, db, task):
        assert get_checkpoint(db, task) is None

    def test_get_latest(self, db, task):
        create_checkpoint(db, task, REASON_MANUAL_HANDOFF, {"goal": "v1"})
        create_checkpoint(db, task, REASON_AUTO_SAVE, {"goal": "v2"})
        c = get_checkpoint(db, task)
        assert c["version"] == 2
        assert c["state"]["goal"] == "v2"

    def test_get_specific_version(self, db, task):
        create_checkpoint(db, task, REASON_MANUAL_HANDOFF, {"goal": "v1"})
        create_checkpoint(db, task, REASON_AUTO_SAVE, {"goal": "v2"})
        c = get_checkpoint(db, task, version=1)
        assert c["version"] == 1
        assert c["state"]["goal"] == "v1"

    def test_user_isolation(self, db, task):
        create_checkpoint(db, task, REASON_MANUAL_HANDOFF, {}, user_id="alice")
        # bob 看不到 alice 的 checkpoint
        assert get_checkpoint(db, task, user_id="bob") is None
        assert get_checkpoint(db, task, user_id="alice") is not None


class TestListCheckpoints:
    def test_empty(self, db, task):
        assert list_checkpoints(db, task) == []

    def test_order_desc(self, db, task):
        for i, r in enumerate([REASON_MANUAL_HANDOFF, REASON_AUTO_SAVE, REASON_FAILURE]):
            create_checkpoint(db, task, r, {"goal": f"g{i}"})
        rows = list_checkpoints(db, task)
        versions = [r["version"] for r in rows]
        assert versions == [3, 2, 1]
        # 不返回完整 state（避免响应过大）
        assert "state" not in rows[0]

    def test_limit(self, db, task):
        for _ in range(5):
            create_checkpoint(db, task, REASON_AUTO_SAVE, {})
        rows = list_checkpoints(db, task, limit=3)
        assert len(rows) == 3


# ============================================================
# build_continuation
# ============================================================

class TestBuildContinuation:
    def test_basic_fields(self, db, task):
        create_checkpoint(db, task, REASON_MANUAL_HANDOFF, {
            "goal": "g",
            "completed": ["a"],
            "in_progress": ["b"],
            "blocked": ["c"],
            "preferred_next": ["d"],
            "must_not_redo": [{"action": "x", "reason": "side_effect_emitted"}],
            "must_preserve": ["no main branch"],
            "working_set": {"files": ["a.py"]},
        })
        c = get_checkpoint(db, task)
        cont = build_continuation(c)
        for k in ("goal", "completed", "in_progress", "blocked",
                  "preferred_next", "must_not_redo", "must_preserve",
                  "working_set", "continuation_confidence",
                  "confidence_breakdown"):
            assert k in cont
        assert cont["goal"] == "g"
        assert cont["must_not_redo"][0]["action"] == "x"

    def test_recency_recomputed(self, db, task):
        # 入库时 recency=1.0；恢复时立刻调用，时间差 ~0 → 仍约 1.0
        create_checkpoint(db, task, REASON_MANUAL_HANDOFF, {"goal": "g"})
        c = get_checkpoint(db, task)
        cont = build_continuation(c)
        assert 0.99 <= cont["confidence_breakdown"]["recency"] <= 1.0


# ============================================================
# should_create_auto_checkpoint (event-first)
# ============================================================

class TestShouldCreateAutoCheckpoint:
    def test_failure_always_triggers(self, db, task):
        # 先造一个 FAILURE，再来一个 failure 事件 → 仍触发（不受 debounce）
        create_checkpoint(db, task, REASON_FAILURE, {})
        ok, reason = should_create_auto_checkpoint(db, task, "failure", {})
        assert ok is True
        assert reason == REASON_FAILURE

    def test_progress_triggers_plan_update(self, db, task):
        create_checkpoint(db, task, REASON_MANUAL_HANDOFF,
                          {"in_progress": ["old-plan"]})
        ok, reason = should_create_auto_checkpoint(
            db, task, "progress", {"in_progress": ["new-plan"]},
        )
        assert ok is True
        assert reason == REASON_PLAN_UPDATE

    def test_progress_triggers_working_set_shift(self, db, task):
        create_checkpoint(db, task, REASON_MANUAL_HANDOFF, {
            "in_progress": ["same"],
            "working_set": {"files": ["a.py"]},
        })
        ok, reason = should_create_auto_checkpoint(db, task, "progress", {
            "in_progress": ["same"],
            "working_set": {"files": ["x.py", "y.py"]},
        })
        assert ok is True
        assert reason == REASON_WORKING_SET_SHIFT

    def test_progress_no_change_no_trigger(self, db, task):
        # 创建一个最近的 checkpoint → 5min 兜底不生效，且 plan/ws 都没变
        create_checkpoint(db, task, REASON_MANUAL_HANDOFF, {
            "in_progress": ["same"],
            "working_set": {"files": ["a.py"]},
        })
        ok, reason = should_create_auto_checkpoint(db, task, "progress", {
            "in_progress": ["same"],
            "working_set": {"files": ["a.py"]},
        })
        assert ok is False
        assert reason is None

    def test_first_event_triggers_auto_save(self, db, task):
        # 完全没有 checkpoint 时，任何 tick 都触发 AUTO_SAVE
        ok, reason = should_create_auto_checkpoint(db, task, "tick", {})
        assert ok is True
        assert reason == REASON_AUTO_SAVE

    def test_debounce_blocks_repeated_plan_update(self, db, task):
        # 先用 progress 触发 PLAN_UPDATE
        create_checkpoint(db, task, REASON_MANUAL_HANDOFF,
                          {"in_progress": ["a"]})
        ok1, r1 = should_create_auto_checkpoint(
            db, task, "progress", {"in_progress": ["b"]},
        )
        assert ok1 and r1 == REASON_PLAN_UPDATE
        # 模拟 handler 立即写入这个 checkpoint
        create_checkpoint(db, task, REASON_PLAN_UPDATE,
                          {"in_progress": ["b"]})
        # 立刻再来一次 plan 变化 → 60s debounce 应阻止
        ok2, r2 = should_create_auto_checkpoint(
            db, task, "progress", {"in_progress": ["c"]},
        )
        assert ok2 is False
        assert r2 is None


# ============================================================
# D3: handle_session_handoff 接入 checkpoint 的集成测试
# ============================================================

@pytest.fixture
def env(tmp_path, monkeypatch):
    """与 test_handlers.py 一致的环境：fake embedding + 真 DB/Graph。"""
    db = MemoryDB(str(tmp_path / "h.duckdb"), dim=768)
    graph = MemoryGraph(str(tmp_path / "h.json"))
    monkeypatch.setattr("engram.handlers.embed", lambda t: [0.1] * 768)
    monkeypatch.setattr("engram.retrieve.embed", lambda t: [0.1] * 768)
    return db, graph


class TestHandoffCheckpointIntegration:
    def test_handoff_without_task_id_unchanged(self, env):
        """不传 task_id：老行为完全一致，返回值无 checkpoint 字段。"""
        from engram.handlers import handle_session_handoff
        db, graph = env
        result = handle_session_handoff(
            db, graph,
            summary="did stuff",
            completed=["a"],
            next_steps=["b"],
        )
        assert "memory_id" in result
        assert "checkpoint_id" not in result
        assert "checkpoint_version" not in result
        # 数据库中没有 checkpoint
        cnt = db.conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        assert cnt == 0

    def test_handoff_with_task_id_creates_checkpoint(self, env):
        """传 task_id：自动创建 MANUAL_HANDOFF checkpoint。"""
        from engram.handlers import handle_session_handoff
        db, graph = env
        tid = db.create_task(name="t1", goal="ship feature X")
        result = handle_session_handoff(
            db, graph,
            summary="phase 1 done",
            completed=["plan", "scaffold"],
            in_progress=["impl"],
            blocked=["api-key"],
            next_steps=["wire endpoints", "tests"],
            task_id=tid,
        )
        assert "checkpoint_id" in result
        assert result["checkpoint_version"] == 1
        assert 0 < result["continuation_confidence"] <= 1.0

        # 验证 checkpoint 内容
        c = ckpt.get_checkpoint(db, tid)
        assert c is not None
        assert c["reason"] == ckpt.REASON_MANUAL_HANDOFF
        assert c["kind"] == ckpt.KIND_HANDOFF
        assert c["state"]["goal"] == "ship feature X"
        assert c["state"]["completed"] == ["plan", "scaffold"]
        assert c["state"]["in_progress"] == ["impl"]
        assert c["state"]["blocked"] == ["api-key"]
        assert c["state"]["preferred_next"] == ["wire endpoints", "tests"]
        assert c["source_memory_id"] == result["memory_id"]
        assert c["triggered_by_event"] == "session_handoff"

    def test_handoff_with_must_not_redo_persisted(self, env):
        from engram.handlers import handle_session_handoff
        db, graph = env
        tid = db.create_task(name="t2")
        result = handle_session_handoff(
            db, graph,
            summary="created PR",
            task_id=tid,
            must_not_redo=[
                {
                    "action": "create_pr",
                    "reason": "side_effect_emitted",
                    "idempotency_key": "pr-42",
                }
            ],
            must_preserve=["never push to main"],
            working_set={"files": ["api.py", "models.py"]},
            session_id="sess_abc",
        )
        assert "checkpoint_id" in result
        c = ckpt.get_checkpoint(db, tid)
        assert len(c["state"]["must_not_redo"]) == 1
        assert c["state"]["must_not_redo"][0]["action"] == "create_pr"
        assert c["state"]["must_not_redo"][0]["idempotency_key"] == "pr-42"
        assert c["state"]["must_preserve"] == ["never push to main"]
        assert c["state"]["working_set"] == {"files": ["api.py", "models.py"]}
        assert c["source_session_id"] == "sess_abc"

    def test_handoff_with_invalid_task_id_no_crash(self, env):
        """checkpoint 创建失败时 handoff 主流程仍成功（task_id 不存在 → goal 取空）。

        即使 get_task 返回 None，create_checkpoint 仍能完成（goal=''），
        但 tasks 表的缓存字段 UPDATE 0 行，不影响 handoff。
        """
        from engram.handlers import handle_session_handoff
        db, graph = env
        result = handle_session_handoff(
            db, graph,
            summary="ghost task",
            task_id=999999,
        )
        # handoff 主流程必须成功
        assert "memory_id" in result
        assert "recorded" in result["result"]
        # checkpoint 仍写入了（task_id 是外键软关联）
        c = ckpt.get_checkpoint(db, 999999)
        assert c is not None
        assert c["state"]["goal"] == ""

    def test_handoff_repeated_increments_version(self, env):
        from engram.handlers import handle_session_handoff
        db, graph = env
        tid = db.create_task(name="t3")
        r1 = handle_session_handoff(db, graph, summary="round 1", task_id=tid)
        r2 = handle_session_handoff(db, graph, summary="round 2", task_id=tid)
        r3 = handle_session_handoff(db, graph, summary="round 3", task_id=tid)
        assert r1["checkpoint_version"] == 1
        assert r2["checkpoint_version"] == 2
        assert r3["checkpoint_version"] == 3
        # tasks 缓存字段同步更新 (v0.18: data in SQLite Tier 2)
        if db._state_store:
            ver, count = db._state_store.get_task_checkpoint_cache(tid)
            assert (ver, count) == (3, 3)
        else:
            row = db.conn.execute(
                "SELECT latest_checkpoint_version, checkpoint_count FROM tasks WHERE id = ?",
                [tid],
            ).fetchone()
            assert row == (3, 3)


# ============================================================
# D4: track_failure / track_progress 接入 event-first checkpoint
# ============================================================

class TestTrackFailureCheckpoint:
    def test_failure_without_task_id_unchanged(self, env):
        from engram.handlers import handle_track_failure
        db, graph = env
        result = handle_track_failure(
            db, graph, error="boom", component="api",
        )
        assert "memory_id" in result
        assert "checkpoint_version" not in result
        cnt = db.conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        assert cnt == 0

    def test_failure_with_task_id_creates_failure_checkpoint(self, env):
        from engram.handlers import handle_track_failure
        db, graph = env
        tid = db.create_task(name="t", goal="ship")
        result = handle_track_failure(
            db, graph,
            error="connection timeout",
            component="db",
            root_cause="pool exhausted",
            severity="critical",
            task_id=tid,
        )
        assert result["checkpoint_version"] == 1
        assert result["checkpoint_reason"] == ckpt.REASON_FAILURE

        c = ckpt.get_checkpoint(db, tid)
        assert c["reason"] == ckpt.REASON_FAILURE
        assert c["kind"] == ckpt.KIND_AUTO
        assert c["state"]["goal"] == "ship"
        assert "connection timeout" in c["state"]["blocked"]
        assert "pool exhausted" in c["state"]["blocked"]
        assert c["state"]["working_set"] == {"tools": ["db"]}
        assert c["failure_signature"] == "db:critical"
        assert c["triggered_by_event"] == "track_failure"
        assert c["source_memory_id"] == result["memory_id"]

    def test_failure_bypasses_debounce(self, env):
        """两次 FAILURE 都应该触发 checkpoint（不受 60s debounce 限制）。"""
        from engram.handlers import handle_track_failure
        db, graph = env
        tid = db.create_task(name="t")
        r1 = handle_track_failure(db, graph, error="e1", component="x", task_id=tid)
        r2 = handle_track_failure(db, graph, error="e2", component="x", task_id=tid)
        assert r1["checkpoint_version"] == 1
        assert r2["checkpoint_version"] == 2


class TestTrackProgressCheckpoint:
    def test_progress_without_task_id_unchanged(self, env):
        from engram.handlers import handle_track_progress
        db, graph = env
        result = handle_track_progress(
            db, graph, feature="x", status="in_progress",
        )
        assert "memory_id" in result
        assert "checkpoint_version" not in result
        cnt = db.conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        assert cnt == 0

    def test_progress_first_triggers_auto_save(self, env):
        """没有任何 checkpoint 时，progress 触发 AUTO_SAVE 兜底（首个 in_progress 也算 plan_changed，二者择一）。"""
        from engram.handlers import handle_track_progress
        db, graph = env
        tid = db.create_task(name="t")
        result = handle_track_progress(
            db, graph, feature="impl", status="in_progress", task_id=tid,
        )
        assert "checkpoint_version" in result
        # 首次写入：可能是 PLAN_UPDATE（in_progress 从空变为 1 项 → Jaccard=0 < 0.7）
        # 实际看 should_create_auto_checkpoint：先检测 plan_changed → True → PLAN_UPDATE
        assert result["checkpoint_reason"] == ckpt.REASON_PLAN_UPDATE

    def test_progress_no_change_skips_checkpoint(self, env):
        """连续两次 progress 但内容相同 → 第二次 debounce 阻止。"""
        from engram.handlers import handle_track_progress
        db, graph = env
        tid = db.create_task(name="t")
        r1 = handle_track_progress(
            db, graph, feature="impl", status="in_progress", task_id=tid,
        )
        r2 = handle_track_progress(
            db, graph, feature="impl", status="in_progress", task_id=tid,
        )
        # 第一次触发了 PLAN_UPDATE
        assert "checkpoint_version" in r1
        # 第二次：plan/working_set 都没变，且 5min 内 → 不触发
        assert "checkpoint_version" not in r2

    def test_progress_plan_change_triggers_plan_update(self, env):
        """切换 feature → in_progress 列表 Jaccard=0 → PLAN_UPDATE。"""
        from engram.handlers import handle_track_progress
        from engram.checkpoint import create_checkpoint, REASON_MANUAL_HANDOFF
        db, graph = env
        tid = db.create_task(name="t")
        # 用 MANUAL_HANDOFF 落一个基线 checkpoint，避开"首个 progress 触发 PLAN_UPDATE"的边界
        create_checkpoint(db, tid, REASON_MANUAL_HANDOFF, {
            "in_progress": ["feat-A [in_progress]"],
        })
        # 现在切到 feat-B
        result = handle_track_progress(
            db, graph, feature="feat-B", status="in_progress", task_id=tid,
        )
        assert result["checkpoint_reason"] == ckpt.REASON_PLAN_UPDATE

    def test_done_progress_writes_to_completed(self, env):
        """status=done 时，feature 进 completed 而不是 in_progress。"""
        from engram.handlers import handle_track_progress
        from engram.checkpoint import create_checkpoint, REASON_MANUAL_HANDOFF
        db, graph = env
        tid = db.create_task(name="t")
        # 基线
        create_checkpoint(db, tid, REASON_MANUAL_HANDOFF, {
            "in_progress": ["feat-A [in_progress]"],
        })
        result = handle_track_progress(
            db, graph, feature="feat-A", status="done",
            completion=100, task_id=tid,
        )
        # done → in_progress 从 [feat-A] 变 [] → Jaccard=0 → PLAN_UPDATE
        assert result.get("checkpoint_reason") == ckpt.REASON_PLAN_UPDATE
        c = ckpt.get_checkpoint(db, tid)
        assert "feat-A [done]" in c["state"]["completed"]
        assert c["state"]["in_progress"] == []


# ============================================================
# D5: restore_checkpoint / list_checkpoints handler 集成测试
# ============================================================

class TestRestoreCheckpointHandler:
    def test_invalid_task_id(self, env):
        from engram.handlers import handle_restore_checkpoint
        db, graph = env
        result = handle_restore_checkpoint(db, graph, task_id="abc")
        assert result.get("ok") is False
        assert "task_id" in result["error"]

    def test_no_checkpoint_returns_fallback(self, env):
        from engram.handlers import handle_restore_checkpoint
        db, graph = env
        tid = db.create_task(name="t", goal="g")
        result = handle_restore_checkpoint(db, graph, task_id=tid)
        assert result["ok"] is False
        assert result["error"] == "no_checkpoint"
        assert result["fallback"]["task"]["id"] == tid
        assert "hint" in result["fallback"]

    def test_task_not_found(self, env):
        from engram.handlers import handle_restore_checkpoint
        db, graph = env
        result = handle_restore_checkpoint(db, graph, task_id=999999)
        assert result.get("ok") is False
        assert "not found" in result["error"]

    def test_basic_restore(self, env):
        from engram.handlers import handle_session_handoff, handle_restore_checkpoint
        db, graph = env
        tid = db.create_task(name="t", goal="ship")
        handle_session_handoff(
            db, graph, summary="phase 1",
            completed=["a"], next_steps=["b"], task_id=tid,
        )
        result = handle_restore_checkpoint(db, graph, task_id=tid)
        assert result["task_id"] == tid
        assert result["version"] == 1
        assert result["checkpoint_reason"] == ckpt.REASON_MANUAL_HANDOFF
        cont = result["continuation"]
        assert cont["goal"] == "ship"
        assert cont["completed"] == ["a"]
        assert cont["preferred_next"] == ["b"]
        assert "continuation_confidence" in cont

    def test_restore_specific_version(self, env):
        from engram.handlers import handle_session_handoff, handle_restore_checkpoint
        db, graph = env
        tid = db.create_task(name="t", goal="g")
        handle_session_handoff(db, graph, summary="r1", completed=["a"], task_id=tid)
        handle_session_handoff(db, graph, summary="r2", completed=["a", "b"], task_id=tid)
        # 默认取最新（v2）
        latest = handle_restore_checkpoint(db, graph, task_id=tid)
        assert latest["version"] == 2
        assert latest["continuation"]["completed"] == ["a", "b"]
        # 显式取 v1
        v1 = handle_restore_checkpoint(db, graph, task_id=tid, version=1)
        assert v1["version"] == 1
        assert v1["continuation"]["completed"] == ["a"]

    def test_memory_restore_mode_none(self, env):
        from engram.handlers import handle_session_handoff, handle_restore_checkpoint
        db, graph = env
        tid = db.create_task(name="t")
        handle_session_handoff(db, graph, summary="s", task_id=tid)
        result = handle_restore_checkpoint(
            db, graph, task_id=tid, memory_restore_mode="NONE",
        )
        assert result["memory_restore_mode"] == "NONE"
        assert "related_memories" not in result
        assert "related_failures" not in result

    def test_memory_restore_mode_selective_filters(self, env):
        from engram.handlers import (
            handle_session_handoff, handle_track_progress,
            handle_restore_checkpoint, handle_store,
        )
        db, graph = env
        tid = db.create_task(name="t")
        # importance>=0.5 的内容（handoff 0.9，progress 0.8 in_progress）应被保留
        handle_session_handoff(db, graph, summary="s", task_id=tid)
        handle_track_progress(db, graph, feature="x", status="in_progress", task_id=tid)
        # 直接挂一个 importance<0.5 的 memory 到 task → 应被 SELECTIVE 过滤掉
        from engram.embedding import embed
        mid_low = db.insert(
            "low importance note", [0.1] * 768, 0.2, "fact", "default",
            metadata={"task_id": tid, "type": "fact"},
        )
        result = handle_restore_checkpoint(
            db, graph, task_id=tid, memory_restore_mode="SELECTIVE",
        )
        assert result["memory_restore_mode"] == "SELECTIVE"
        ids = [m["memory_id"] for m in result["related_memories"]]
        assert mid_low not in ids

    def test_memory_restore_mode_full_keeps_low_importance(self, env):
        from engram.handlers import (
            handle_session_handoff, handle_restore_checkpoint,
        )
        db, graph = env
        tid = db.create_task(name="t")
        handle_session_handoff(db, graph, summary="s", task_id=tid)
        mid_low = db.insert(
            "low importance note", [0.1] * 768, 0.2, "fact", "default",
            metadata={"task_id": tid, "type": "fact"},
        )
        result = handle_restore_checkpoint(
            db, graph, task_id=tid, memory_restore_mode="FULL",
        )
        ids = [m["memory_id"] for m in result["related_memories"]]
        assert mid_low in ids

    def test_invalid_mode_falls_back_to_selective(self, env):
        from engram.handlers import handle_session_handoff, handle_restore_checkpoint
        db, graph = env
        tid = db.create_task(name="t")
        handle_session_handoff(db, graph, summary="s", task_id=tid)
        result = handle_restore_checkpoint(
            db, graph, task_id=tid, memory_restore_mode="BOGUS",
        )
        assert result["memory_restore_mode"] == "SELECTIVE"

    def test_failure_signature_in_restore(self, env):
        from engram.handlers import handle_track_failure, handle_restore_checkpoint
        db, graph = env
        tid = db.create_task(name="t")
        handle_track_failure(
            db, graph, error="boom", component="db", severity="critical", task_id=tid,
        )
        result = handle_restore_checkpoint(db, graph, task_id=tid)
        assert result["failure_signature"] == "db:critical"


class TestListCheckpointsHandler:
    def test_invalid_task_id(self, env):
        from engram.handlers import handle_list_checkpoints
        db, graph = env
        result = handle_list_checkpoints(db, graph, task_id="abc")
        assert result.get("ok") is False

    def test_empty(self, env):
        from engram.handlers import handle_list_checkpoints
        db, graph = env
        tid = db.create_task(name="t")
        result = handle_list_checkpoints(db, graph, task_id=tid)
        assert result["total"] == 0
        assert result["checkpoints"] == []

    def test_returns_metadata_only(self, env):
        from engram.handlers import handle_session_handoff, handle_list_checkpoints
        db, graph = env
        tid = db.create_task(name="t")
        handle_session_handoff(db, graph, summary="r1", task_id=tid)
        handle_session_handoff(db, graph, summary="r2", task_id=tid)
        result = handle_list_checkpoints(db, graph, task_id=tid)
        assert result["total"] == 2
        # latest first
        assert result["checkpoints"][0]["version"] == 2
        # 不返回完整 state
        assert "state" not in result["checkpoints"][0]
        assert "continuation_confidence" in result["checkpoints"][0]

    def test_limit(self, env):
        from engram.handlers import handle_session_handoff, handle_list_checkpoints
        db, graph = env
        tid = db.create_task(name="t")
        for _ in range(5):
            handle_session_handoff(db, graph, summary="s", task_id=tid)
        result = handle_list_checkpoints(db, graph, task_id=tid, limit=3)
        assert result["total"] == 3


class TestGetTaskInjectsLatestCheckpoint:
    """D6: get_task 返回值注入 latest_checkpoint 字段。"""

    def test_no_checkpoint_no_field(self, env):
        from engram.handlers import handle_get_task
        db, graph = env
        tid = db.create_task(name="t", goal="g")
        result = handle_get_task(db, graph, task_id=tid)
        assert "latest_checkpoint" not in result

    def test_handoff_creates_latest_checkpoint(self, env):
        from engram.handlers import handle_session_handoff, handle_get_task
        db, graph = env
        tid = db.create_task(name="t", goal="ship")
        handle_session_handoff(
            db, graph, summary="s", completed=["a"], next_steps=["b"], task_id=tid,
        )
        result = handle_get_task(db, graph, task_id=tid)
        assert "latest_checkpoint" in result
        latest = result["latest_checkpoint"]
        assert latest["version"] == 1
        assert latest["checkpoint_reason"] == ckpt.REASON_MANUAL_HANDOFF
        cont = latest["continuation"]
        assert cont["goal"] == "ship"
        assert cont["completed"] == ["a"]
        assert cont["preferred_next"] == ["b"]
        assert "continuation_confidence" in cont
        # 不携带 related_memories（避免响应膨胀）
        assert "related_memories" not in latest
        assert "related_failures" not in latest

    def test_latest_reflects_newest_version(self, env):
        from engram.handlers import handle_session_handoff, handle_get_task
        db, graph = env
        tid = db.create_task(name="t")
        handle_session_handoff(db, graph, summary="r1", completed=["a"], task_id=tid)
        handle_session_handoff(db, graph, summary="r2", completed=["a", "b"], task_id=tid)
        result = handle_get_task(db, graph, task_id=tid)
        assert result["latest_checkpoint"]["version"] == 2
        assert result["latest_checkpoint"]["continuation"]["completed"] == ["a", "b"]

    def test_failure_signature_preserved(self, env):
        from engram.handlers import handle_track_failure, handle_get_task
        db, graph = env
        tid = db.create_task(name="t")
        handle_track_failure(
            db, graph, error="boom", component="db", severity="critical", task_id=tid,
        )
        result = handle_get_task(db, graph, task_id=tid)
        assert result["latest_checkpoint"]["failure_signature"] == "db:critical"


class TestFullContinuityLoop:
    """Agent A → 中断 → Agent B restore → 继续 → 完成 的端到端验证。"""

    def test_full_loop(self, env):
        from engram.handlers import (
            handle_create_task, handle_track_progress, handle_track_failure,
            handle_session_handoff, handle_restore_checkpoint,
            handle_list_checkpoints,
        )
        db, graph = env

        # —— Session A：开任务，做事，记录失败，主动 handoff
        tid = handle_create_task(db, graph, name="ship-feature-x", goal="ship X")["task_id"]
        handle_track_progress(
            db, graph, feature="impl-core", status="in_progress",
            task_id=tid,
        )
        handle_track_failure(
            db, graph, error="auth timeout", component="auth-service",
            severity="major", task_id=tid,
        )
        handoff_a = handle_session_handoff(
            db, graph,
            summary="paused mid-impl due to auth timeout",
            completed=["scaffolding"],
            in_progress=["impl-core"],
            blocked=["auth timeout"],
            next_steps=["check auth config", "retry impl-core"],
            must_not_redo=[{
                "action": "create_pr",
                "reason": "side_effect_emitted",
                "idempotency_key": "pr-x-init",
            }],
            must_preserve=["never push to main"],
            working_set={"files": ["api.py"], "tools": ["auth-service"]},
            task_id=tid,
        )
        assert "checkpoint_id" in handoff_a
        version_at_handoff = handoff_a["checkpoint_version"]

        # —— Session B：新 Agent 接手，调 restore_checkpoint
        restored = handle_restore_checkpoint(db, graph, task_id=tid)
        assert restored["version"] == version_at_handoff
        cont = restored["continuation"]

        # 关键 continuity 不变量
        assert cont["goal"] == "ship X"
        assert "scaffolding" in cont["completed"]
        assert "impl-core" in cont["in_progress"]
        assert "auth timeout" in cont["blocked"]
        assert "check auth config" in cont["preferred_next"]
        # negative memory 必须保留（防重复 PR）
        assert any(
            x["action"] == "create_pr" and x["idempotency_key"] == "pr-x-init"
            for x in cont["must_not_redo"]
        )
        # invariant 必须保留
        assert "never push to main" in cont["must_preserve"]
        # working_set 必须保留
        assert "api.py" in cont["working_set"].get("files", [])
        # 历史 failure 上下文应被自动注入（auth-service 命中）
        failure_components = {f["component"] for f in restored.get("related_failures", [])}
        assert "auth-service" in failure_components

        # —— Session B 继续工作 → 写新 progress
        handle_track_progress(
            db, graph, feature="impl-core", status="done",
            completion=100, task_id=tid,
        )
        # —— Session B 主动 handoff 收尾
        handoff_b = handle_session_handoff(
            db, graph, summary="done", completed=["impl-core"],
            next_steps=[], task_id=tid,
        )

        # —— 验证 checkpoint 链单调递增
        ckpts = handle_list_checkpoints(db, graph, task_id=tid)
        versions = [c["version"] for c in ckpts["checkpoints"]]
        assert versions == sorted(versions, reverse=True)
        # latest > version_at_handoff
        assert ckpts["checkpoints"][0]["version"] > version_at_handoff
        assert handoff_b["checkpoint_version"] == ckpts["checkpoints"][0]["version"]
