#!/usr/bin/env python3
"""Engram Continuity Benchmark — LIVE (v2).

Where the Core bench scores the *continuation package itself* under
scripted agent actions, the Live bench answers the harder, causal
question:

    Given a real LLM that takes over an interrupted task with ONLY the
    recovery package (and no ground-truth list of what's forbidden), does
    it actually walk back fewer wasted steps under SELECTIVE restore than
    under FULL — i.e. is the context-pollution effect REAL, not scripted?

Design (executor / judge separation — the methodology backbone):
  * ACTOR LLM  — receives only the rendered recovery package. Asked for the
    next 3-5 concrete actions. Does NOT see forbidden_actions.
  * JUDGE LLM  — receives the actor's actions + the scenario ground truth
    (forbidden_actions + must_preserve). Labels each action as a redo of a
    forbidden path / a constraint violation / clean.
  Same model, different system prompt (v2 starter config).

The actor's actions are then fed through the SAME metric the Core bench
uses (redundant_exploration), plus a Live-only constraint_violation rate.

This bench is NON-deterministic (real LLM). It reports mean ± std over
--runs. It is NOT in CI. It reuses the Core scenario dataset verbatim, so
Core and Live are directly comparable.

Usage:
    export OPENAI_API_KEY=...
    python benchmark/continuity_bench_live.py \
        --llm DeepSeek-V3.2 --base-url https://api.example.com/v1 \
        --scenarios a1_sigterm b1_goal_mutation c1_retry_storm \
        --runs 3

    # dry-run: render prompts + exercise flow without calling the API
    python benchmark/continuity_bench_live.py --dry-run \
        --scenarios a1_sigterm b1_goal_mutation c1_retry_storm
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")

BENCH_DIR = Path(__file__).parent
RESULTS_DIR = BENCH_DIR / "results"
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(BENCH_DIR.resolve().parents[0] / "src"))

import continuity_bench as core  # reuse setup, modes, metrics, weights
from engram.db import MemoryDB
from engram.graph import MemoryGraph
from engram import checkpoint as ckpt_mod
from engram.handlers import handle_restore_checkpoint

# Default small-sample scenario set (one representative per axis) — keeps
# the first Live run cheap while still testing the core causal claim.
DEFAULT_LIVE_SCENARIOS = ["a1_sigterm", "b1_goal_mutation", "c1_retry_storm"]


# ============================================================
# LLM clients (OpenAI-compatible, mirrors locomo_eval.make_llm_fn)
# ============================================================

def make_chat_fn(model: str, base_url: str, api_key: str):
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key)

    def call(system: str, user: str, max_tokens: int = 600) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()

    return call


# ============================================================
# Recovery package → actor prompt
# ============================================================

ACTOR_SYSTEM = (
    "You are an autonomous coding agent taking over a task that was "
    "interrupted mid-flight. You have ONLY the recovery package below — no "
    "memory of the prior session beyond it. Decide what to do next.\n\n"
    "Output ONLY a JSON array of 3 to 5 short action strings (imperative, "
    "concrete, one clause each). No prose, no markdown fences. Example:\n"
    '["wire the refresh token rotation", "add token expiry tests"]'
)


def render_recovery_prompt(continuation: dict, related_memories: list[dict],
                           related_failures: list[dict], mode: str) -> str:
    """Render what each mode actually hands the actor.

    NONE     → almost nothing (amnesiac).
    SELECTIVE→ continuation + filtered memories/failures.
    FULL     → continuation + ALL task memories (the pollution baseline).
    """
    lines = []
    if mode == "NONE":
        lines.append("RECOVERY PACKAGE: (none — the previous session left no "
                     "usable checkpoint. You are starting cold.)")
        lines.append(f"\nTask goal (all you know): {continuation.get('goal', '(unknown)')}")
        return "\n".join(lines)

    lines.append("RECOVERY PACKAGE")
    lines.append(f"Goal: {continuation.get('goal', '')}")
    lines.append(f"Completed: {continuation.get('completed', [])}")
    lines.append(f"In progress: {continuation.get('in_progress', [])}")
    lines.append(f"Preferred next: {continuation.get('preferred_next', [])}")
    mnr = [m.get("action", "") for m in continuation.get("must_not_redo", []) if isinstance(m, dict)]
    lines.append(f"Must NOT redo: {mnr}")
    lines.append(f"Must preserve: {continuation.get('must_preserve', [])}")
    lines.append(f"Working set: {continuation.get('working_set', {})}")

    mems = [m.get("content", "") if isinstance(m, dict) else str(m) for m in related_memories]
    fails = [m.get("content", "") if isinstance(m, dict) else str(m) for m in related_failures]
    if mems:
        lines.append(f"\nRelated memories ({len(mems)}):")
        lines.extend(f"  - {m}" for m in mems)
    if fails:
        lines.append(f"\nRelated failures ({len(fails)}):")
        lines.extend(f"  - {m}" for m in fails)
    return "\n".join(lines)


# ============================================================
# Judge
# ============================================================

JUDGE_SYSTEM = (
    "You are a strict evaluator of an AI agent's recovery behaviour. You are "
    "given (a) the agent's proposed next actions, and (b) the GROUND TRUTH "
    "of what it must NOT do (forbidden actions) and what it must preserve "
    "(constraints). For each proposed action, decide if it REDOES a "
    "forbidden action or VIOLATES a constraint.\n\n"
    "Match on intent, not exact wording. Be strict: if an action clearly "
    "re-does forbidden work or breaks a constraint, flag it.\n\n"
    "Output ONLY JSON: {\"verdicts\": [{\"action\": str, \"redoes_forbidden\": "
    "bool, \"violates_constraint\": bool}], \"reasoning\": str}. No prose "
    "outside the JSON."
)


def build_judge_prompt(actions: list[str], forbidden: list[str],
                       must_preserve: list[str]) -> str:
    return (
        f"Agent's proposed actions:\n{json.dumps(actions, ensure_ascii=False)}\n\n"
        f"GROUND TRUTH — forbidden actions (must NOT redo):\n"
        f"{json.dumps(forbidden, ensure_ascii=False)}\n\n"
        f"GROUND TRUTH — constraints (must preserve):\n"
        f"{json.dumps(must_preserve, ensure_ascii=False)}\n\n"
        "Evaluate each proposed action."
    )


def _parse_json_loose(text: str):
    """Strip markdown fences and parse JSON; return None on failure."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if "```" in t[3:] else t[3:]
        if t.startswith("json"):
            t = t[4:]
        t = t.strip().rstrip("`").strip()
    try:
        return json.loads(t)
    except Exception:
        # last resort: find first [ or { ... matching tail
        for opener, closer in (("[", "]"), ("{", "}")):
            i, j = t.find(opener), t.rfind(closer)
            if i != -1 and j != -1 and j > i:
                try:
                    return json.loads(t[i:j + 1])
                except Exception:
                    pass
    return None


def normalize_judge_output(parsed, raw: str = "") -> dict:
    """Coerce whatever the judge emitted into {"verdicts": [...], "reasoning": str}.

    Different models return different shapes for the same prompt:
      * {"verdicts": [...], "reasoning": ...}   — the requested shape
      * [{...}, {...}]                          — a bare verdict array (GLM does this)
      * {"action": ..., "redoes_forbidden": ...} — a single verdict object
      * None / garbage                          — parse failed
    Always returns a dict with a "verdicts" list so callers never crash.
    """
    if isinstance(parsed, dict):
        if "verdicts" in parsed:
            v = parsed["verdicts"]
            parsed["verdicts"] = v if isinstance(v, list) else [v]
            return parsed
        # A single bare verdict object.
        if "redoes_forbidden" in parsed or "violates_constraint" in parsed:
            return {"verdicts": [parsed], "reasoning": parsed.get("reasoning", "")}
        return {"verdicts": [], "reasoning": str(parsed)[:200]}
    if isinstance(parsed, list):
        return {"verdicts": parsed, "reasoning": ""}
    return {"verdicts": [], "reasoning": (raw or "")[:200]}


# ============================================================
# One (scenario, mode, run)
# ============================================================

def run_live_one(scenario: dict, mode: str, run_idx: int,
                 actor_fn, judge_fn, dry_run: bool) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        db = MemoryDB(str(Path(tmp) / "live.duckdb"), dim=768)
        graph = MemoryGraph(str(Path(tmp) / "live.json"))

        # 1) Replay Agent A's work into engram (identical to Core).
        state = core._build_engram_state(scenario)
        task_id = db.create_task(scenario["scenario_id"], goal=state["goal"], user_id="default")
        for mem in scenario.get("task_memories", []):
            db.insert(mem["content"], core.FAKE_EMBED,
                      importance=mem.get("importance", 0.5),
                      category=mem.get("category", "fact"), user_id="default",
                      metadata={"task_id": str(task_id), "type": mem.get("category", "fact")})
        ckpt_mod.create_checkpoint(db, task_id, ckpt_mod.REASON_AUTO_SAVE, state, user_id="default")

        # 2) Recovery package per mode.
        related, failures, continuation = [], [], {"goal": state["goal"]}
        observed = {"related_memories": 0, "related_failures": 0, "real_restore": False}
        if mode != "NONE":
            result = handle_restore_checkpoint(db, graph, task_id=task_id, memory_restore_mode=mode)
            continuation = result.get("continuation", {})
            related = result.get("related_memories", [])
            failures = result.get("related_failures", [])
            observed.update(real_restore=True, related_memories=len(related),
                            related_failures=len(failures))
        else:
            continuation = {"goal": state["goal"], "completed": [], "in_progress": [],
                            "preferred_next": [], "must_not_redo": [], "must_preserve": [],
                            "working_set": {}}

        actor_prompt = render_recovery_prompt(continuation, related, failures, mode)

        # 3) Actor proposes actions.
        if dry_run:
            actions = [f"<dry-run action for {mode}>"]
            judged = {"verdicts": [], "reasoning": "dry-run"}
            redundant, violation = 1.0, 0.0
        else:
            raw = actor_fn(ACTOR_SYSTEM, actor_prompt)
            actions = _parse_json_loose(raw)
            if not isinstance(actions, list):
                actions = [raw]  # judge will see raw text as one action
            actions = [str(a) for a in actions][:5]

            # 4) Judge labels against ground truth.
            gt = scenario.get("ground_truth", {})
            forbidden = gt.get("forbidden_actions", [])
            preserve = gt.get("must_preserve", [])
            jraw = judge_fn(JUDGE_SYSTEM, build_judge_prompt(actions, forbidden, preserve))
            judged = normalize_judge_output(_parse_json_loose(jraw), jraw)
            verdicts = judged.get("verdicts", [])

            n = len(actions) or 1
            redo = sum(1 for v in verdicts if isinstance(v, dict) and v.get("redoes_forbidden"))
            viol = sum(1 for v in verdicts if isinstance(v, dict) and v.get("violates_constraint"))
            redundant = 1.0 - (redo / n)          # higher = fewer redos (matches Core)
            violation = viol / n                  # Live-only: constraint break rate

        return {
            "mode": mode, "run": run_idx,
            "actions": actions,
            "redundant_exploration": round(redundant, 4),
            "constraint_violation": round(violation, 4),
            "observed": observed,
            "judge_reasoning": judged.get("reasoning", "")[:300],
        }


# ============================================================
# Aggregation + report
# ============================================================

def _mean_std(vals):
    if not vals:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": round(statistics.mean(vals), 4),
            "std": round(statistics.pstdev(vals) if len(vals) > 1 else 0.0, 4)}


def render_markdown(report: dict) -> str:
    L = []
    L.append(f"# Engram Continuity Benchmark — LIVE (v2) — {report['timestamp']}")
    L.append("")
    L.append(f"- Scenarios: **{len(report['scenarios'])}** | Modes: "
             f"**{', '.join(report['modes'])}** | Runs each: **{report['runs']}** "
             f"| Model: **{report['model']}**")
    L.append(f"- Total LLM calls: ~**{report['total_calls']}** "
             f"(actor + judge), non-deterministic")
    L.append("")
    L.append("> **Live bench.** Real LLM takes over each interrupted task with "
             "ONLY the recovery package (actor never sees the forbidden list). "
             "An independent judge labels its actions against ground truth. "
             "This measures *real* recovery behaviour — the causal counterpart "
             "to the Core bench's scripted scores.")
    L.append("")
    L.append("## Mode comparison (mean across scenarios)")
    L.append("")
    L.append("| Mode | Redundant↑ | ConstraintViol↓ | Real recall |")
    L.append("|------|-----------|-----------------|-------------|")
    for mode in report["modes"]:
        reds, viols, recs = [], [], []
        for sc in report["scenarios"].values():
            reds.append(sc[mode]["redundant_exploration"]["mean"])
            viols.append(sc[mode]["constraint_violation"]["mean"])
            recs.append(sc[mode]["observed"]["related_memories"])
        mean = lambda xs: round(statistics.mean(xs), 3) if xs else 0.0
        L.append(f"| {mode} | {mean(reds)} | {mean(viols)} | {mean(recs)} |")
    L.append("")
    L.append("## Per-scenario redundant_exploration (mean ± std, higher = fewer redos)")
    L.append("")
    L.append("| Scenario | Axis | " + " | ".join(report["modes"]) + " |")
    L.append("|----------|------|" + "|".join(["---"] * len(report["modes"])) + "|")
    for sid, sc in report["scenarios"].items():
        axis = report["scenario_axes"].get(sid, "?")
        cells = [f"{sc[m]['redundant_exploration']['mean']}±{sc[m]['redundant_exploration']['std']}"
                 for m in report["modes"]]
        L.append(f"| {sid} | {axis} | " + " | ".join(cells) + " |")
    L.append("")
    return "\n".join(L)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Engram Continuity Benchmark — LIVE (v2)")
    ap.add_argument("--llm", type=str, default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--base-url", type=str, default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--scenarios", nargs="*", default=DEFAULT_LIVE_SCENARIOS)
    ap.add_argument("--mode", choices=["none", "selective", "full", "all"], default="all")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true", help="render + flow, no API calls")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    modes = core.MODES if args.mode == "all" else [args.mode.upper()]
    scenarios = core.load_scenarios(args.scenarios)
    if not scenarios:
        print("No scenarios found.", file=sys.stderr); sys.exit(1)

    actor_fn = judge_fn = None
    if not args.dry_run:
        if not args.api_key:
            print("Error: --api-key or $OPENAI_API_KEY required (or use --dry-run)", file=sys.stderr)
            sys.exit(1)
        actor_fn = make_chat_fn(args.llm, args.base_url, args.api_key)
        judge_fn = make_chat_fn(args.llm, args.base_url, args.api_key)  # same model, diff prompt

    total_calls = len(scenarios) * len(modes) * args.runs * 2  # actor + judge
    print(f"Live: {len(scenarios)} scenarios × {len(modes)} modes × {args.runs} runs "
          f"= {len(scenarios)*len(modes)*args.runs} actor calls (~{total_calls} total)")
    print(f"Model: {args.llm} @ {args.base_url}  {'[DRY-RUN]' if args.dry_run else ''}\n")

    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modes": modes, "runs": args.runs, "model": args.llm,
        "base_url": args.base_url, "dry_run": args.dry_run,
        "total_calls": total_calls, "scenarios": {}, "scenario_axes": {},
    }

    for sc in scenarios:
        sid = sc["scenario_id"]
        report["scenario_axes"][sid] = sc.get("axis", "?")
        report["scenarios"][sid] = {}
        for mode in modes:
            per_run = [run_live_one(sc, mode, i, actor_fn, judge_fn, args.dry_run)
                       for i in range(args.runs)]
            agg = {
                "redundant_exploration": _mean_std([r["redundant_exploration"] for r in per_run]),
                "constraint_violation": _mean_std([r["constraint_violation"] for r in per_run]),
                "observed": per_run[-1]["observed"],
                "samples": per_run,
            }
            report["scenarios"][sid][mode] = agg
            r = agg["redundant_exploration"]
            print(f"  {sid:24s} {mode:10s} redundant={r['mean']}±{r['std']} "
                  f"(recall={agg['observed']['related_memories']})")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.out or str(RESULTS_DIR / f"continuity_live_{stamp}")
    Path(f"{prefix}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    Path(f"{prefix}.md").write_text(render_markdown(report))
    print(f"\nJSON  → {prefix}.json\nTable → {prefix}.md\n")
    print(render_markdown(report))


if __name__ == "__main__":
    main()
