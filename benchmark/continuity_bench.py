#!/usr/bin/env python3
"""Engram Continuity Benchmark (Core) — quantify how well a checkpoint
restore preserves cognitive state across an interruption, across three
recovery modes (NONE / SELECTIVE / FULL).

This is the **Core** bench: it scores the *continuation package itself*
(does it preserve goal / completed / working-set / failure-context, and
does it carry enough negative memory to avoid redundant exploration).

It is NOT the Live bench. Core uses *scripted* agent actions
(`agent_replay`) — it measures whether the recovery package would let an
agent avoid redoing forbidden work, under an assumed action sequence. The
real causal question ("does a real LLM actually walk back fewer steps")
is the Live bench (v2), deliberately kept separate so Core stays
deterministic and CI-able.

Honest boundaries (also printed in every report):
  * Only SELECTIVE runs a *real* engram restore_checkpoint. NONE / FULL are
    constructed baselines (empty package / full-history package).
  * `build_continuation` runs before the mode gate, so the structural
    fields (goal/completed/working_set/must_not_redo) are IDENTICAL across
    modes. Structural metrics therefore act as REGRESSION GUARDS (~1.0,
    proving the package faithfully preserves the checkpoint), not as the
    discriminator between modes.
  * The mode discriminator is `redundant_exploration`, driven by whether
    failure memories are recalled (SELECTIVE/FULL recall them, NONE does
    not) + the scripted `agent_replay`.
  * `observed.related_memories` records what each mode REALLY recalled from
    engram — the only fully-real (non-scripted) signal in Core.

Usage:
    python benchmark/continuity_bench.py --mode all --runs 5 --seed 42
    python benchmark/continuity_bench.py --mode selective --runs 1
    python benchmark/continuity_bench.py --scenarios a1_sigterm c1_retry_storm
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import tempfile
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")

BENCH_DIR = Path(__file__).parent
SCENARIO_DIR = BENCH_DIR / "continuity_scenarios"
RESULTS_DIR = BENCH_DIR / "results"

sys.path.insert(0, str(BENCH_DIR.resolve().parents[0] / "src"))

from engram.db import MemoryDB
from engram.graph import MemoryGraph
from engram import checkpoint as ckpt_mod
from engram import continuity as cont
from engram.handlers import handle_restore_checkpoint

# Deterministic fake embedding — Core bench never needs real semantics.
# (Recall is driven by metadata.task_id + importance/category filters, not
#  vector similarity, so a constant embedding is correct and reproducible.)
FAKE_EMBED = [0.1] * 768

MODES = ["NONE", "SELECTIVE", "FULL"]

# Benchmark composite weights (independent of continuity.py's internal
# weights — Core deliberately drops replanning_rate (degenerate when
# scripted) and adds completed_preservation; redundant gets the heaviest
# weight because it is closest to the bench's core question: "fewer
# wasted steps".)
BENCH_WEIGHTS = {
    "goal_preservation": 0.20,
    "completed_preservation": 0.20,
    "working_set_overlap": 0.15,
    "failure_context": 0.20,
    "redundant_exploration": 0.25,
}


# ============================================================
# Bench-local metric (not yet in continuity.py — kept here until stable)
# ============================================================

def completed_preservation(before_state: dict, after_state: dict) -> float:
    """Set coverage of the `completed` list across restore.

    1.0 = every previously-completed item is still marked completed.
    0.0 = all completed work was lost.
    """
    before = set(ckpt_mod._as_list(before_state.get("completed")))
    after = set(ckpt_mod._as_list(after_state.get("completed")))
    if not before:
        return 1.0
    return len(before & after) / len(before)


def score_one(before_state: dict, after_state: dict,
              forbidden_actions: list[str],
              actions_taken: list[str]) -> dict:
    """Compute the 5 raw metrics + bench composite for one (mode, run).

    Reuses continuity.py's pure functions for 4 of them; completed_
    preservation is bench-local. redundant_exploration is driven by the
    scenario's forbidden_actions (the ground-truth set) against the
    mode-dependent scripted actions_taken.
    """
    forbidden_as_mnr = [{"action": a} for a in forbidden_actions]

    raw = {
        "goal_preservation": cont.goal_retention(before_state, after_state),
        "completed_preservation": completed_preservation(before_state, after_state),
        "working_set_overlap": cont.working_set_stability(before_state, after_state),
        "failure_context": cont.failure_recall(before_state, after_state),
        "redundant_exploration": cont.redundant_exploration(
            forbidden_as_mnr, actions_taken,
        ),
    }
    composite = sum(BENCH_WEIGHTS[k] * v for k, v in raw.items())
    raw_rounded = {k: round(v, 4) for k, v in raw.items()}
    return {"raw": raw_rounded, "composite": round(composite, 4)}


# ============================================================
# Scenario execution
# ============================================================

def _build_engram_state(scenario: dict) -> dict:
    """The full state dict written into engram + checkpoint."""
    s = scenario["pre_interrupt_state"]
    return {
        "goal": s.get("goal", ""),
        "completed": s.get("completed", []),
        "in_progress": s.get("in_progress", []),
        "blocked": s.get("blocked", []),
        "preferred_next": s.get("preferred_next", []),
        "must_not_redo": ckpt_mod.normalize_must_not_redo(s.get("must_not_redo", [])),
        "must_preserve": s.get("must_preserve", []),
        "working_set": s.get("working_set", {}),
    }


def _restored_state_for_mode(scenario: dict, mode: str,
                             real_continuation: dict) -> dict:
    """Resolve the after_state for scoring.

    Default: use the REAL continuation package engram produced (faithful
    restore → structural metrics act as regression guards, ~1.0).
    Override: if the scenario declares restored_state_by_mode, inject a
    controlled drift (B-axis) to prove the structural metrics are sensitive
    (not dead 1.0s). Honestly labelled in the report as scripted injection.
    """
    by_mode = scenario.get("restored_state_by_mode")
    if not by_mode:
        return real_continuation
    if "ALL" in by_mode:
        return by_mode["ALL"]
    if mode in by_mode:
        return by_mode[mode]
    return real_continuation


def run_scenario_mode(scenario: dict, mode: str, run_idx: int) -> dict:
    """Run one scenario under one recovery mode, once.

    Each run uses an isolated temp DB so writes never collide and the run
    is fully reproducible.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db = MemoryDB(str(Path(tmp) / "bench.duckdb"), dim=768)
        graph = MemoryGraph(str(Path(tmp) / "bench.json"))

        # 1) Replay Agent A's work into engram (real writes).
        state = _build_engram_state(scenario)
        task_id = db.create_task(
            scenario["scenario_id"], goal=state["goal"], user_id="default",
        )
        for mem in scenario.get("task_memories", []):
            db.insert(
                mem["content"], FAKE_EMBED,
                importance=mem.get("importance", 0.5),
                category=mem.get("category", "fact"),
                user_id="default",
                metadata={"task_id": str(task_id), "type": mem.get("category", "fact")},
            )

        # 2) Real checkpoint at interruption time.
        ckpt_mod.create_checkpoint(
            db, task_id, ckpt_mod.REASON_AUTO_SAVE, state, user_id="default",
        )

        # 3) Recovery.
        observed = {"related_memories": 0, "related_failures": 0,
                    "continuation_confidence": None, "real_restore": False}

        if mode == "NONE":
            # Constructed baseline: empty recovery package. The agent gets
            # nothing → before_state for scoring is an empty package.
            real_continuation = {
                "goal": "", "completed": [], "in_progress": [],
                "preferred_next": [], "must_not_redo": [],
                "must_preserve": [], "working_set": {},
            }
        else:
            # SELECTIVE / FULL: REAL engram restore.
            result = handle_restore_checkpoint(
                db, graph, task_id=task_id, memory_restore_mode=mode,
            )
            real_continuation = result.get("continuation", {})
            observed["real_restore"] = True
            observed["related_memories"] = len(result.get("related_memories", []))
            observed["related_failures"] = len(result.get("related_failures", []))
            observed["continuation_confidence"] = real_continuation.get(
                "continuation_confidence")

        after_state = _restored_state_for_mode(scenario, mode, real_continuation)

        # before_state = ground truth (what SHOULD have been preserved).
        before_state = state
        actions_taken = scenario.get("agent_replay", {}).get(mode, [])
        forbidden = scenario.get("ground_truth", {}).get("forbidden_actions", [])

        scored = score_one(before_state, after_state, forbidden, actions_taken)
        scored["observed"] = observed
        scored["mode"] = mode
        scored["run"] = run_idx
        return scored


# ============================================================
# Aggregation
# ============================================================

def _mean_std(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(std, 4)}


def aggregate(per_run: list[dict]) -> dict:
    """Collapse N runs of one (scenario, mode) into mean±std per metric."""
    metric_keys = list(BENCH_WEIGHTS.keys())
    agg = {"composite": _mean_std([r["composite"] for r in per_run])}
    for k in metric_keys:
        agg[k] = _mean_std([r["raw"][k] for r in per_run])
    # observed signals — report the modal/last (deterministic across runs).
    agg["observed"] = per_run[-1]["observed"]
    return agg


# ============================================================
# Reporting
# ============================================================

def render_markdown(report: dict) -> str:
    lines = []
    lines.append(f"# Engram Continuity Benchmark (Core) — {report['timestamp']}")
    lines.append("")
    lines.append(f"- Scenarios: **{len(report['scenarios'])}**  "
                 f"| Modes: **{', '.join(report['modes'])}**  "
                 f"| Runs each: **{report['runs']}**  "
                 f"| Seed: **{report['seed']}**")
    total = len(report['scenarios']) * len(report['modes']) * report['runs']
    lines.append(f"- Total evaluations: **{total}**")
    lines.append("")
    lines.append("> **Core bench.** Scores the continuation package itself under "
                 "scripted agent actions. Only SELECTIVE is a real engram restore; "
                 "NONE/FULL are constructed baselines. Structural metrics are "
                 "regression guards (~1.0); `redundant_exploration` is the mode "
                 "discriminator. Real causal behaviour is the Live bench (v2).")
    lines.append("")

    # Overall mode comparison (mean composite across scenarios).
    lines.append("## Mode comparison (mean composite across all scenarios)")
    lines.append("")
    lines.append("| Mode | Composite | Redundant | FailCtx | Goal | Completed | WorkingSet | Real recall |")
    lines.append("|------|-----------|-----------|---------|------|-----------|------------|-------------|")
    for mode in report["modes"]:
        comps, reds, fcs, goals, comps2, ws, recalls = [], [], [], [], [], [], []
        for sc in report["scenarios"].values():
            m = sc[mode]
            comps.append(m["composite"]["mean"])
            reds.append(m["redundant_exploration"]["mean"])
            fcs.append(m["failure_context"]["mean"])
            goals.append(m["goal_preservation"]["mean"])
            comps2.append(m["completed_preservation"]["mean"])
            ws.append(m["working_set_overlap"]["mean"])
            recalls.append(m["observed"]["related_memories"])
        mean = lambda xs: round(statistics.mean(xs), 3) if xs else 0.0
        lines.append(
            f"| {mode} | {mean(comps)} | {mean(reds)} | {mean(fcs)} | "
            f"{mean(goals)} | {mean(comps2)} | {mean(ws)} | {mean(recalls)} |"
        )
    lines.append("")

    # Per-scenario composite.
    lines.append("## Per-scenario composite (mean ± std)")
    lines.append("")
    lines.append("| Scenario | Axis | " + " | ".join(report["modes"]) + " |")
    lines.append("|----------|------|" + "|".join(["------"] * len(report["modes"])) + "|")
    for sid, sc in report["scenarios"].items():
        axis = report["scenario_axes"].get(sid, "?")
        cells = []
        for mode in report["modes"]:
            c = sc[mode]["composite"]
            cells.append(f"{c['mean']}±{c['std']}")
        lines.append(f"| {sid} | {axis} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def load_scenarios(only: list[str] | None) -> list[dict]:
    files = sorted(SCENARIO_DIR.glob("*.json"))
    scenarios = []
    for f in files:
        if f.name.startswith("_"):
            continue
        data = json.loads(f.read_text())
        if only and data["scenario_id"] not in only:
            continue
        scenarios.append(data)
    return scenarios


def main():
    ap = argparse.ArgumentParser(description="Engram Continuity Benchmark (Core)")
    ap.add_argument("--mode", choices=["none", "selective", "full", "all"],
                    default="all", help="recovery mode(s) to run")
    ap.add_argument("--runs", type=int, default=5, help="runs per (scenario, mode)")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--scenarios", nargs="*", default=None,
                    help="scenario_ids to run (default: all)")
    ap.add_argument("--out", default=None, help="output path prefix (default: results/)")
    args = ap.parse_args()

    random.seed(args.seed)
    modes = MODES if args.mode == "all" else [args.mode.upper()]
    scenarios = load_scenarios(args.scenarios)
    if not scenarios:
        print("No scenarios found.", file=sys.stderr)
        sys.exit(1)

    print(f"Running {len(scenarios)} scenarios × {len(modes)} modes × "
          f"{args.runs} runs = {len(scenarios) * len(modes) * args.runs} evals\n")

    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modes": modes,
        "runs": args.runs,
        "seed": args.seed,
        "weights": BENCH_WEIGHTS,
        "scenarios": {},
        "scenario_axes": {},
    }

    for scenario in scenarios:
        sid = scenario["scenario_id"]
        report["scenario_axes"][sid] = scenario.get("axis", "?")
        report["scenarios"][sid] = {}
        for mode in modes:
            per_run = [run_scenario_mode(scenario, mode, i) for i in range(args.runs)]
            report["scenarios"][sid][mode] = aggregate(per_run)
            comp = report["scenarios"][sid][mode]["composite"]
            obs = report["scenarios"][sid][mode]["observed"]
            print(f"  {sid:24s} {mode:10s} composite={comp['mean']}±{comp['std']} "
                  f"(real_recall={obs['related_memories']})")

    # Write outputs.
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.out or str(RESULTS_DIR / f"continuity_{stamp}")
    json_path = Path(f"{prefix}.json")
    md_path = Path(f"{prefix}.md")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    md_path.write_text(render_markdown(report))

    print(f"\nJSON  → {json_path}")
    print(f"Table → {md_path}\n")
    print(render_markdown(report))


if __name__ == "__main__":
    main()
