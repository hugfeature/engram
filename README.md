# Engram

**Runtime continuity and interruption recovery for AI coding agents**

[![PyPI](https://img.shields.io/pypi/v/mcp-engram)](https://pypi.org/project/mcp-engram/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-456%20passed-brightgreen)](https://github.com/hugfeature/engram)

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
Tier 1 — Runtime Continuity (Source of Truth, replay-recoverable)
  tasks · checkpoints · session lifecycle · handoff events
  → append-only event log  ~/.engram/events/*.jsonl  (fsync, gzip-rotated)

Tier 2 — Semantic Recall (Degradable, readonly-recoverable)
  memories · metadata · semantic graph
  → DuckDB projection from event log

Tier 3 — Retrieval Cache (Disposable, rebuildable)
  embeddings · FTS · vector index
  → rebuilt on demand, never in recovery path
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
