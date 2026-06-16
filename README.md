# Engram

**Runtime continuity and interruption recovery for AI coding agents**

[![PyPI](https://img.shields.io/pypi/v/mcp-engram)](https://pypi.org/project/mcp-engram/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-545%20passed-brightgreen)](https://github.com/hugfeature/engram)
[![engram MCP server](https://glama.ai/mcp/servers/hugfeature/engram/badges/card.svg)](https://glama.ai/mcp/servers/hugfeature/engram)

> Engram gives Claude Code, Cursor, and OpenHands **resumable execution** — recover agent working state after context collapse, interruption, or session termination. Agents continue from where they stopped, not from zero.

[中文文档](./README_CN.md) · [PyPI](https://pypi.org/project/mcp-engram/) · [Glama](https://glama.ai/mcp/servers/hugfeature/engram)

---

## The Problem

Every AI agent session is an island:

| Situation                       | Without Engram                       | With Engram                                       |
| ------------------------------- | ------------------------------------ | ------------------------------------------------- |
| Context window full             | Agent loses history, starts guessing | Checkpoint restored, constraints preserved        |
| Process killed / server restart | All working state lost               | Auto-checkpoint on SIGTERM, resume on next recall |
| Switch agents (Claude → Cursor) | Start from scratch                   | Structured handoff with task state                |
| Same bug hits twice             | No memory of previous fix            | Failure record surfaces automatically             |
| 3-session task                  | No one knows overall progress        | Task state + continuity metrics available         |

---

## Why Engram

Most memory systems focus on conversation recall.

Engram focuses on runtime continuity:

- **What the agent was doing** — task state, working set, modified files
- **Where execution stopped** — checkpoint with goal, progress, blockers
- **How to resume safely** — negative memory (must_not_redo) + invariants (must_preserve)

---

## Quick Start

```bash
pip install mcp-engram
engram-setup          # downloads embedding model + initializes DB (~500MB, one-time)
```

Add to your Claude Code / Cursor config:

```json
{
  "mcpServers": {
    "engram": {
      "command": "engram",
      "env": { "HF_ENDPOINT": "https://huggingface.co" }
    }
  }
}
```

That's it. Engram runs fully locally — no cloud, no API keys, no telemetry.

---

## Interruption Recovery

When a session ends unexpectedly — context overflow, process kill, IDE crash — Engram automatically captures a structured working state on shutdown:

```
SIGTERM / context collapse
        ↓
Engram auto-checkpoint
  goal · completed · in_progress · blocked
  modified_files (git diff) · last_tool_called · last_failure
        ↓
Agent restarts → recall_memory()
        ↓
interrupt_recovery injected into response
  → "Call restore_checkpoint(task_id=X) to resume"
```

No manual intervention. The next agent session picks up the hint automatically on first `recall_memory` call.

**Generate a CLAUDE.md snippet from current state:**

```bash
engram-prompt
```

Output (paste into your project's `CLAUDE.md`):

```markdown
## Engram Runtime State

**Active task:** 10 — Fix the auth bug in login.py
**Checkpoint:** v3 (confidence: 0.71)

**Already completed (do not redo):**

- Reproduced the bug in test_auth.py

**In progress:**

- Tracing the JWT validation path

**Files modified in last session:**

- `src/auth/validator.py`
- `tests/test_auth.py`

## Engram Session Rules

- Session start: recall_memory(query) — interrupt state auto-pinned
- Resume task: restore_checkpoint(task_id=10)
- Context filling up: report_interruption(reason="overflow") then session_handoff(...)
```

---

## How It Works

Engram is a local [MCP](https://modelcontextprotocol.io/) server providing **18 tools** across three areas:

### Memory

Hybrid semantic search (BM25 + vector + graph) with Ebbinghaus decay so important memories stay longer.

| Tool                 | What it does                                         |
| -------------------- | ---------------------------------------------------- |
| `store_memory`       | Save a fact, decision, or lesson (auto-deduplicates) |
| `recall_memory`      | Hybrid search — call at the start of every task      |
| `update_memory`      | Correct a specific memory by ID                      |
| `consolidate_memory` | Merge similar memories, prune weak ones              |
| `memory_stats`       | Health overview: counts, strength, last maintenance  |
| `get_runtime_health` | Read-only backend health report                      |

### Task Tracking

Tasks are first-class entities that survive session boundaries.

| Tool             | What it does                                    |
| ---------------- | ----------------------------------------------- |
| `create_task`    | Start a tracked multi-session task              |
| `get_task`       | Full task context + latest checkpoint           |
| `update_task`    | Update status, goal, or metadata                |
| `list_tasks`     | See all tasks, filter by status                 |
| `track_progress` | Snapshot feature/task progress                  |
| `track_failure`  | Record structured failure with root cause + fix |

### Continuity

Checkpoint-based recovery so a new agent can pick up exactly where the previous one left off.

| Tool                  | What it does                                          |
| --------------------- | ----------------------------------------------------- |
| `session_handoff`     | End-of-session structured summary                     |
| `session_outcome`     | Mark session success/failure (adjusts memory weights) |
| `restore_checkpoint`  | Constrained continuation package for takeover         |
| `list_checkpoints`    | Checkpoint history for a task                         |
| `report_interruption` | Signal overflow/rate-limit before exit                |
| `evaluate_continuity` | Score recovery quality across checkpoint versions     |

---

## Checkpoint Recovery Flow

```
Agent A (Claude Code)                  Agent B (Cursor)
  │                                        │
  ├─ create_task(name, goal)               │
  ├─ track_progress / track_failure        │
  ├─ ⚡ Interrupted (SIGTERM / overflow)   │
  ├─ [auto] interrupt checkpoint saved     │
  │        │                               │
  │   ┌────▼────────────────┐              │
  │   │  Engram Checkpoint  │              │
  │   │  goal               │              │
  │   │  completed          │              │
  │   │  must_not_redo  ────┼──────────▶   │
  │   │  must_preserve      │             ├─ recall_memory() → interrupt_recovery pinned
  │   │  modified_files     │             ├─ restore_checkpoint(task_id)
  │   │  working_set        │             ├─ continue, not from zero
  │   └─────────────────────┘             └─ session_handoff → next agent...
```

**Continuation package fields**: `goal` · `completed` · `in_progress` · `blocked` · `preferred_next` · `must_not_redo` (negative memory) · `must_preserve` (invariants) · `working_set` · `continuation_confidence`

---

## Recommended CLAUDE.md Instructions

Run `engram-prompt` to auto-generate this from your current state, or paste manually:

```markdown
## Engram Session Rules

- Task start: create_task(name, goal) → save the task_id
- Session begin: recall_memory(query) — latest handoff + interrupt state auto-pinned
- Task takeover: restore_checkpoint(task_id, memory_restore_mode="SELECTIVE")
- Progress update: track_progress(feature, status, task_id=<id>)
- On error: track_failure(error, component, root_cause, task_id=<id>)
- Context filling up: report_interruption(reason="overflow") then session_handoff(...)
- Session end: session_handoff(summary, completed, in_progress, blocked, next_steps, task_id=<id>)
```

---

## Architecture

```
Tier 1 — Event Journal (immutable, append-only)
  source of truth · snapshot compaction · incremental replay
  → ~/.engram/events/*.jsonl  (fsync, gzip-rotated)
  → periodic snapshots for fast startup

Tier 2 — Runtime State Store (operationally durable)
  tasks · checkpoints · executions · sessions
  → SQLite WAL  ~/.engram/*.state.sqlite
  → concurrent reads, fast restore, recovery-critical
  → controlled by ENGRAM_SQLITE_TIER2=1 (fallback: DuckDB)

Tier 3 — Runtime Intelligence Cache (rebuildable)
  memories · embeddings · FTS · vector index
  future: drift vectors · recovery metrics · tool stats · continuity history
  → DuckDB (can be dropped and rebuilt without data loss)
  → rebuild with: engram-setup rebuild-cache
```

**Two Laws:**

1. The event log is the only durability primitive.
2. If it cannot be replayed, it is not critical state.

DB corruption enters **readonly degraded mode** — never silently resets. Recover with `engram-setup recover`.

---

## Benchmark

Evaluated on [LoCoMo](https://github.com/snap-research/locomo) (Snap Research long-term conversation memory benchmark):

| System     | Overall F1 | Hit@5     | LLM           | Deployment |
| ---------- | ---------- | --------- | ------------- | ---------- |
| MemMachine | 0.8487     | —         | GPT-4o-mini   | Cloud      |
| Memobase   | 0.7578     | —         | GPT-4o-mini   | Cloud      |
| Zep        | 0.7514     | —         | GPT-4o-mini   | Cloud      |
| Mem0       | 0.6688     | —         | GPT-4o-mini   | Cloud      |
| **Engram** | **0.4383** | **77.7%** | DeepSeek-V3.2 | **Local**  |

Zero cloud dependency. Four optimization rounds: **F1 +50.3%**, **Hit@5 +26.2pp**.
LoCoMo measures *retrieval* quality. But Engram's real job is **recovery** — surviving an interruption and handing a clean state to the next agent. That needs its own benchmark.

### Continuity Benchmark (Core)

LoCoMo tells you if memory is *findable*. It does not tell you if a **restore** is any good. There is no standard benchmark for cross-session continuity, so Engram ships its own.

**The question it answers:** given an interruption, does the continuation package let the next agent resume *without redoing forbidden work* — and is selective restore actually better than dumping the full history back in?

```bash
python benchmark/continuity_bench.py --mode all --runs 5 --seed 42
```

**20 scenarios × 3 recovery modes × 5 runs = 300 evaluations.** Fully deterministic (std = 0).

| Recovery mode | Composite | Redundant exploration | What it is |
| ------------- | --------- | --------------------- | ---------- |
| `NONE`        | 0.282     | 0.20                  | Empty package (amnesiac baseline) |
| **`SELECTIVE`** | **0.912** | **1.00**            | Real Engram restore (importance ≥ 0.5 + failures) |
| `FULL`        | 0.791     | 0.52                  | Full task history (context-pollution baseline) |

**`SELECTIVE` beats `FULL`** — independently reproducing the context-pollution finding from Letta's [Recovery-Bench](https://www.letta.com/blog/recovery-bench) (whose `--message-mode full/summary/none` mirrors Engram's `memory_restore_mode`). The two real-restore modes preserve identical *structural* state; the gap is entirely in **redundant exploration** — `FULL` re-feeds stale failed paths and the agent re-walks them.

Scenarios span three axes: **A — interruption type** (SIGTERM, crash, context overflow, tool timeout, network failure, session restart, long idle, multi-day pause), **B — state drift** (goal mutation, branch switch, dependency-upgrade scope creep, tool-permission change, workspace wipe), **C — failure recall** (retry storm, human handoff, memory corruption, planner restart).
#### For evaluators — honest boundaries

This is the **Core** bench. It scores the *continuation package itself*, not live agent behaviour:

- **Scripted actions.** Agent actions after restore are declared per-scenario (`agent_replay`), not produced by a live LLM. Core measures whether the package *would* let an agent avoid redoing forbidden work, under an assumed action sequence.
- **Structural metrics are regression guards.** `build_continuation` runs before the memory-mode gate, so goal / completed / working-set are identical across modes (~1.0). They prove the package faithfully preserves the checkpoint; they are not the mode discriminator.
- **`redundant_exploration` is the discriminator**, driven by whether failure memories are recalled (real, mode-dependent) + scripted actions.
- **Only `SELECTIVE` is a real `restore_checkpoint`.** `NONE`/`FULL` are constructed baselines. The `observed.related_memories` column records what each mode *really* recalled — the one fully-real signal in Core.
- The real causal question (*does a live agent actually walk back fewer steps?*) is the **Live bench (v2)**, deliberately separate so Core stays deterministic and CI-able.

The invariant `SELECTIVE > FULL > NONE` is guarded in CI (`tests/test_continuity_bench.py`) — if an Engram change ever degrades recovery quality, the build fails.

Bench composite weights (independent of the in-tool metric): Goal 0.20 · Completed 0.20 · Working-set 0.15 · Failure-context 0.20 · **Redundant 0.25** (heaviest — closest to "fewer wasted steps"). The live `evaluate_continuity` tool exposes a related 6-dimension score: Goal Retention · Action Consistency · Failure Recall · Working Set Stability · Replanning Rate · Redundant Exploration.

### Continuity Benchmark — Live (v2)

Core scores the package under *scripted* actions. **Live** closes the causal loop: a real LLM takes over each interrupted task with **only the recovery package** (it never sees the forbidden list), proposes its next actions, and an **independent judge** labels them against ground truth.

- **Actor / judge separation.** The actor only gets the rendered package; the judge gets the actor's actions + ground truth (`forbidden_actions`, `must_preserve`) and flags every redo / constraint violation. Same model, different system prompt.
- **The hypothesis under test:** does a real LLM handed the `FULL` history get *lured into re-walking* stale failed paths, while `SELECTIVE` lets it resume clean?
- Non-deterministic by design: reports mean ± std over `--runs`, **not** in CI. Reuses the Core scenario dataset verbatim, so Core and Live are directly comparable.
- `--dry-run` exercises the full flow (real `restore_checkpoint`, prompt rendering, judging pipeline) without spending API calls — used to validate the harness offline.

#### Results (3 axis-representative scenarios × 3 modes × 3 runs)

`redundant_exploration` (higher = fewer redos of forbidden work):

| Model        | NONE  | SELECTIVE | FULL  |
| ------------ | ----- | --------- | ----- |
| DeepSeek-V3.2 | 0.956 | 1.000     | 1.000 |
| GLM-5.1      | 1.000 | 1.000     | 1.000 |

Real recall per mode was correct throughout (NONE = 0, SELECTIVE ≈ 2.7, FULL = 3).

**Honest finding — Live does *not* reproduce Core's `SELECTIVE > FULL` gap, and that is the result.** Two readings:

1. **A weak-but-real causal signal.** The only score below 1.0 was DeepSeek on the *retry-storm* scenario under `NONE` (0.956): with no recovery package, the model did occasionally propose an action close to a known-bad path — something the scripted Core bench cannot demonstrate. With a package, it never did.
2. **Strong models resist context pollution.** Core's `SELECTIVE > FULL` (0.912 vs 0.791) comes from *scripted* agents that re-walk whatever the package surfaces. Real frontier models filter the `FULL` history themselves, so the pollution gap collapses. The effect Recovery-Bench reported needs **weaker models, longer histories, or more adversarial forbidden-action design** to surface — which is *why* Recovery-Bench uses weak models to manufacture failures.

This delimits the two benches cleanly: **Core measures package quality (model-independent); Live measures real agent behaviour (confounded by model capability).** A negative Live result does not weaken Core — it bounds the claim each can make.

> The Live harness also surfaced a real bug: GLM-5.1's judge returns a bare JSON array instead of `{"verdicts": [...]}`, which crashed the first run. Fixed (`normalize_judge_output` tolerates four output shapes) and regression-guarded in `tests/test_continuity_bench_live.py`.

Engram also tracks **runtime continuity metrics** via `evaluate_continuity`: Goal Retention · Action Consistency · Failure Recall · Working Set Stability · Replanning Rate · Redundant Exploration.

<details>
<summary>Category scores + memory algorithm details</summary>

### Category Scores

| Category    | Count   | F1         | Hit@5     |
| ----------- | ------- | ---------- | --------- |
| Single-Hop  | 114     | 0.5121     | 76.3%     |
| Temporal    | 63      | 0.4501     | 95.2%     |
| Multi-Hop   | 43      | 0.3181     | 60.5%     |
| Open-Domain | 13      | 0.1324     | 61.5%     |
| **Overall** | **233** | **0.4383** | **77.7%** |

### Memory Algorithm

- **Ebbinghaus Decay**: `strength = importance × e^(−λ × days) × (1 + recall_count × 0.2)`
  - `failure`: λ=0.35, ~11d half-life · `strategy`: λ=0.10, ~38d half-life
- **Deduplication**: ≥0.85 → reinforce · 0.65–0.84 → merge/replace contradiction · <0.65 → new
- **Hybrid Retrieval**: `0.3 × BM25 + 0.7 × (semantic × decay strength) + graph boost`
- **Auto-maintenance**: consolidate every 12h + prune (strength < 0.05) + FTS rebuild

### Importance Guide

| Range   | Use for                                    |
| ------- | ------------------------------------------ |
| 0.9–1.0 | Core identity, permanent facts             |
| 0.7–0.8 | Architecture decisions, strong preferences |
| 0.5     | Regular project facts                      |
| 0.2–0.3 | Transient session context                  |

### Key Environment Variables

| Variable                            | Default                 | Description                                                           |
| ----------------------------------- | ----------------------- | --------------------------------------------------------------------- |
| `HF_ENDPOINT`                       | `https://hf-mirror.com` | HuggingFace mirror (change to `https://huggingface.co` outside China) |
| `ENGRAM_MODEL`                      | `all-mpnet-base-v2`     | Embedding model                                                       |
| `ENGRAM_DEDUP_THRESHOLD`            | `0.65`                  | Dedup similarity lower bound                                          |
| `ENGRAM_REINFORCE_THRESHOLD`        | `0.85`                  | Reinforce similarity threshold                                        |
| `ENGRAM_W_BM25` / `ENGRAM_W_VECTOR` | `0.30` / `0.70`         | Retrieval weights                                                     |
| `ENGRAM_PRUNE_THRESHOLD`            | `0.05`                  | Prune strength threshold                                              |
| `ENGRAM_SQLITE_TIER2`               | _(disabled)_            | Set to `1` to enable SQLite WAL Runtime State Store (Tier 2)          |

Full variable list: `src/engram/config.py`

</details>

---

## What Engram Is Not

- ❌ Guaranteed identical LLM behavior after recovery (LLM non-determinism is a physical constraint)
- ❌ Custom agent loop or prompt orchestration (handled by the MCP client)
- ❌ Multi-agent coordination or shared team memory (single-user, local-first)

---

## Requirements

- macOS / Linux / WSL2
- Python 3.11+
- ~500MB disk for embedding model cache (one-time download)

---

## Contributing

```bash
git clone https://github.com/hugfeature/engram.git
cd engram
pip install -e ".[dev]"
pytest tests/ -v
```

Issues and PRs welcome.

## License

[MIT](https://opensource.org/licenses/MIT) · Maintained by [@hugfeature](https://github.com/hugfeature)

---

> _Engram restores an agent's working state, not just its memories._
