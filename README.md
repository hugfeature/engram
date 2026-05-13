# Engram

**Durable Agent Runtime — cross-session task continuity for MCP-aware coding agents**

> Engram lets agents recover **task execution state and working context** after interruptions, restarts, and session boundaries.
> Not another vector DB / long-term memory store — the primary axis is **runtime durability + execution continuity**.
> Positioned as the continuity layer for Claude Code / Cursor / OpenHands / Devin-class runtimes.

[![PyPI](https://img.shields.io/pypi/v/mcp-engram)](https://pypi.org/project/mcp-engram/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

[中文文档](./README_CN.md)

---

## Two Laws

> **Rule 1.** Event log is the only durability primitive.
> **Rule 2.** If it cannot be replayed, it is not critical state.

Any data that claims "must not be lost" must first be written to `~/.engram/events/*.jsonl` (append-only, fsync).
DuckDB is merely its projection layer.

## Tiered Architecture

```
Tier 1 — Runtime Continuity Layer  (Source of Truth, must never be lost)
  tasks · checkpoints · session lifecycle · handoff events
  → append-only event log (~/.engram/events/) + replay-recoverable

Tier 2 — Semantic Recall Layer    (Degradable, readonly-recoverable)
  memories.content · metadata · summaries · semantic graph
  → DuckDB projection from event log

Tier 3 — Derived Retrieval Cache  (Disposable, rebuildable)
  embeddings · FTS · vector index · rerank cache
  → never participates in recovery; rebuilt on demand
```

DB corruption will **not** silently reset: Engram enters **readonly degraded mode**, and you explicitly rebuild with `engram-setup recover`.

---

## Why Memory + Continuity

Every AI agent session is an island:

- **Switch agents?** Start from scratch.
- **Context window full?** Drop history, keep guessing.
- **Yesterday's lessons?** Gone — repeat the same mistakes.
- **Task spans three sessions?** Nobody knows the overall progress.

Root cause: **Agents lack two layers of infrastructure simultaneously** —
| Layer | Problem Solved | Engram's Implementation |
| --- | --- | --- |
| **Memory** | What to "know" across sessions | Hybrid retrieval + Ebbinghaus decay + dedup/contradiction resolution |
| **Continuity** | What to "continue" across interruptions | Task state + structured handoff + behavioral verification |

Engram is a locally-running [MCP](https://modelcontextprotocol.io/) Server that delivers both layers to existing clients like Claude Code and Cursor.

It does **not** do:

- ❌ General agent runtime / workflow orchestration (that's LangGraph, Temporal)
- ❌ Custom agent loop / prompt orchestration (let MCP clients handle that)
- ❌ Guarantee identical LLM behavior after recovery (LLM non-determinism is a physical constraint — we do **constrained continuation**: structured state narrows the action space)

It **specifically** does:

- ✅ Make tasks recoverable after session interruption
- ✅ Make context handoff-able after agent switch
- ✅ Preserve engineering state across long tasks (failures, progress, constraints)
- ✅ Reverse-verify state correctness via subsequent behavior (**Behavioral Handoff Verification**)

---

## Continuity Flow

**The core experience of AI Agent Continuity: checkpoint recovery.**

```
Agent A (Claude Code)                      Agent B (Cursor)
  │                                          │
  ├─ Create task, start execution            │
  ├─ Record progress + failure lessons       │
  ├─ ━━━━━━━━━━━━━━━━━━━                    │
  │   ⚡ Session Interrupted                 │
  │   ━━━━━━━━━━━━━━━━━━━                    │
  ├─ session_handoff(handoff summary)        │
  │        │                                 │
  │        ▼                                 │
  │   ┌─────────────────┐                    │
  │   │   Engram        │                    │
  │   │   Checkpoint    │                    │
  │   │   ┌───────────┐ │                    │
  │   │   │ Task State │ │                    │
  │   │   │ Progress   │ │                    │
  │   │   │ Failures   │ │                    │
  │   │   │ Next Steps │ │                    │
  │   │   └───────────┘ │                    │
  │   └────────┬────────┘                    │
  │            │                             │
  │            ▼                             │
  │     Restore State ───────────────────▶   │
  │                                          ├─ recall_memory(query)
  │                                          │    └─ handoff auto-pinned
  │                                          │    └─ historical failure context
  │                                          │    └─ next_steps execution verify
  │                                          ├─ Continue, not from zero
  │                                          └─ session_handoff() ──────────▶ ...
```

> **No matter how many agents you switch or sessions you cross, task state persists.**

---

## 18 MCP Tools

Engram provides **18 MCP tools** covering the full Cognitive Continuity lifecycle:

### Memory

| Tool | Purpose |
| --- | --- |
| `store_memory` | Store new memory (auto-dedup/merge/replace) |
| `recall_memory` | Semantic search (hybrid: BM25 + vector + graph boost) |
| `update_memory` | Update existing memory by ID |
| `consolidate_memory` | Merge similar memories, reduce bloat |
| `memory_stats` | Count, category distribution, avg strength, last maintenance |
| `get_runtime_health` | Read-only runtime health report (DB mode, event-log summary, backups, residue) |

### Task

| Tool | Purpose |
| --- | --- |
| `create_task` | Create tracked task (first-class entity) |
| `get_task` | Get task + all associated memories + latest checkpoint |
| `update_task` | Update status / goal / metadata |
| `list_tasks` | List tasks, optionally filtered by status |
| `track_progress` | Record feature/task progress snapshot |
| `track_failure` | Record structured failure event (bug, test failure, etc.) |

### Continuity

| Tool | Purpose |
| --- | --- |
| `session_handoff` | Record structured end-of-session state |
| `session_outcome` | Mark session success/failure (adjusts memory importance) |
| `restore_checkpoint` | Restore constrained continuation package from checkpoint |
| `list_checkpoints` | List checkpoint history (latest first) |
| `report_interruption` | Report imminent interruption reason for recovery routing |
| `evaluate_continuity` | Evaluate continuity quality between checkpoint versions |

**Interruption Playbook (recommended default flow)**

1. **Before interruption**: call `report_interruption(reason, context)` when you detect overflow / rate_limit / repeated tool failures.
2. **On takeover**: call `restore_checkpoint(task_id, memory_restore_mode="SELECTIVE")` to get constrained continuation + targeted memories.
3. **After resuming work**: call `evaluate_continuity(task_id, before_version, after_version)` to quantify recovery quality.

---

## Checkpoint v2 — Constrained Continuation

Elevates `session_handoff` to **versioned cognitive checkpoints**: instead of forcing the new agent to replay identical actions, it provides a set of **constraints** that narrow the action space.

**Continuation Package Fields**

| Field | Purpose |
| --- | --- |
| `goal` / `completed` / `in_progress` / `blocked` / `preferred_next` | Task state body |
| `must_not_redo` | Negative memory — actions already done or with side effects, must not redo |
| `must_preserve` | User-stated invariants (e.g. "don't touch the main branch") |
| `working_set` | Working set at interruption (files / tools / artifacts) |
| `continuation_confidence` | System self-assessed recovery reliability (0–1) |

**Event-first Triggers** (saved on cognitive events, not time periods; 60s debounce per reason)

| Reason | Trigger Condition |
| --- | --- |
| `MANUAL_HANDOFF` | `session_handoff` called |
| `FAILURE` | `track_failure` called (forced, bypasses debounce) |
| `PLAN_UPDATE` | `in_progress` Jaccard < 0.7 |
| `WORKING_SET_SHIFT` | Working set Jaccard < 0.5 |
| `AUTO_SAVE` | 5 minutes with no checkpoint (fallback) |

**Interface**

```python
# One-stop recovery (recommended): get_task includes latest_checkpoint
get_task(task_id=42)["latest_checkpoint"]["continuation"]

# Full recovery: with related memories + historical failure context
restore_checkpoint(task_id=42, memory_restore_mode="SELECTIVE")
# memory_restore_mode: FULL / SELECTIVE (default, importance≥0.5 or failure) / NONE

# Checkpoint history
list_checkpoints(task_id=42, limit=10)
```

**Backward Compatible**: Existing tool signatures unchanged; new fields appended. When old tasks have no checkpoint, `restore_checkpoint` falls back gracefully.

---

## Installation

```bash
pip install mcp-engram
engram-setup          # Download embedding model + initialize DuckDB
```

MCP client configuration (Claude Code / Cursor):

```json
{
  "mcpServers": {
    "engram": {
      "command": "engram",
      "env": { "HF_ENDPOINT": "https://hf-mirror.com" }
    }
  }
}
```

Data directory `~/.engram/`: `memories.duckdb` (single-file DB) + `graph.json` (semantic graph) + `model_cache/` (model).

### Recommended CLAUDE.md Agent Instructions

```markdown
## Memory Rules
- Task start: create_task(name, goal)
- Session begin: recall_memory(query) — handoff auto-pinned + historical failures
- Task takeover: get_task(task_id) — includes latest_checkpoint
- Progress update: track_progress(feature, status, task_id=X)
- On error: track_failure(error, component, root_cause, task_id=X)
- Session end: session_handoff(summary, completed, in_progress, blocked,
            next_steps, must_not_redo=[...], must_preserve=[...],
            working_set={...}, task_id=X)
```

Supports macOS / Linux / WSL2, Python 3.11+, ~500MB model cache.

If your MCP host has strict startup timeouts, Engram boots in a fast path by
default (no eager embedding-model dimension probe). To force eager dimension
detection during boot, set `ENGRAM_EAGER_DIM_DETECT=1`.

---

## Benchmark

Evaluated on [LoCoMo](https://github.com/snap-research/locomo) (Snap Research long-term conversation memory benchmark):

| System     | Overall F1 | LLM           | Deployment |
| ---------- | ---------- | ------------- | ---------- |
| MemMachine | 0.8487     | GPT-4o-mini   | Cloud      |
| Memobase   | 0.7578     | GPT-4o-mini   | Cloud      |
| Zep        | 0.7514     | GPT-4o-mini   | Cloud      |
| Mem0       | 0.6688     | GPT-4o-mini   | Cloud      |
| **Engram** | **0.4383** | DeepSeek-V3.2 | **Local**  |

> Zero cloud dependency, local deployment. Four optimization rounds yielded **F1 +50.3%**, **Hit@5 +26.2pp**.

For runtime durability, also track **continuity metrics** (`evaluate_continuity`): Goal Retention, Action Consistency, Failure Recall, Working Set Stability, Replanning Rate, Redundant Exploration, and a weighted composite score.

<details>
<summary>Category scores + memory mechanism details</summary>

### Category Scores

| Category    |   Count |         F1 |     Hit@5 |
| ----------- | ------: | ---------: | --------: |
| Single-Hop  |     114 |     0.5121 |     76.3% |
| Temporal    |      63 |     0.4501 |     95.2% |
| Multi-Hop   |      43 |     0.3181 |     60.5% |
| Open-Domain |      13 |     0.1324 |     61.5% |
| **Overall** | **233** | **0.4383** | **77.7%** |

### Memory Mechanisms (Key Algorithm Summary)

- **Ebbinghaus Decay**: `strength = importance × e^(-λ × days) × (1 + recall_count × 0.2)`, `failure` half-life ~11 days, `strategy` ~38 days
- **Deduplication**: similarity ≥0.85 reinforces, 0.65–0.84 detects contradiction then merges/overwrites, <0.65 creates new
- **Hybrid Retrieval**: `0.3 × BM25 + 0.7 × (semantic similarity × decay strength) + graph boost`, HNSW + DuckDB FTS
- **Recall Enhancement**: handoff auto-pinned + associated failure context + dynamic `quality_score`
- **Auto Maintenance**: consolidate every 12h (≥0.70 cluster merge) + prune (strength<0.05) + FTS rebuild

### Importance Reference

`0.9–1.0` core identity / permanent facts · `0.7–0.8` architecture decisions / strong preferences · `0.5` regular facts · `0.2–0.3` transient context

### Environment Variables (High-Frequency)

| Variable | Default | Description |
| --- | --- | --- |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace mirror |
| `ENGRAM_MODEL` | `all-mpnet-base-v2` | Embedding model |
| `ENGRAM_DEDUP_THRESHOLD` | `0.65` | Dedup similarity lower bound |
| `ENGRAM_REINFORCE_THRESHOLD` | `0.85` | Reinforce similarity threshold |
| `ENGRAM_W_BM25` / `ENGRAM_W_VECTOR` | `0.30` / `0.70` | Retrieval weights |
| `ENGRAM_PRUNE_THRESHOLD` | `0.05` | Prune strength threshold |
| `ENGRAM_CONSOLIDATE_THRESHOLD` | `0.70` | Consolidate cluster threshold |

Full variable list in `src/engram/config.py`.

</details>

---

## Roadmap

Focus principle: **only build Memory + Continuity dual layers**. Anything sliding toward "general agent runtime / workflow orchestration" is deferred to avoid overlap with LangGraph / Temporal.

### Shipped

- [x] ~~Error-aware Memory~~ — attach historical failure context by component ✅
- [x] ~~Handoff Validation~~ — next_steps execution status detection ✅
- [x] ~~Task Context~~ — Task as first-class entity, cross-session task panorama ✅
- [x] ~~Memory Quality Score~~ — dynamic scoring based on importance + recall + outcome ✅
- [x] ~~Session Lifecycle~~ — auto heartbeat, interruption detection, atexit fallback ✅

### Shipped (Cognitive Continuation Layer 1)

- [x] ~~**Checkpoint v2**~~ — versioned cognitive checkpoint, event-first trigger (6 reason types), `restore_checkpoint` / `list_checkpoints` live, supports constrained continuation (must_not_redo as negative memory / must_preserve / preferred_next / working_set / continuation_confidence) ✅

### Shipped (Interruption Taxonomy)

- [x] ~~**Interruption Taxonomy**~~ — 6 interruption categories (overflow / user_away / tool_failure / crash / rate_limit / unknown), route recovery strategy by type. New `report_interruption` MCP tool for LLM proactive reporting; `cleanup_stale_sessions` auto heuristic classification; recall `interrupted_sessions` hint provides targeted recovery advice by interruption type ✅

### Shipped (Chaos Continuity Test + Continuity Metrics)

- [x] ~~**Continuity Metrics**~~ — 6-dimension metrics engine (`continuity.py`): Goal Retention / Action Consistency / Failure Recall / Working Set Stability / Replanning Rate / Redundant Exploration. `restore_checkpoint` auto-attaches `continuity_score`, new `evaluate_continuity` MCP tool for proactive assessment ✅
- [x] ~~**Chaos Continuity Test**~~ — 5 major interruption scenario automated tests: Normal Handoff (baseline) / SIGTERM / kill -9 crash / Failure Mid-Session / Working Set Drift, all passing with quantified recovery quality ✅

### Shipped (P1-6 Event Log Gzip Rotate)

- [x] ~~**Event Log Gzip Rotate**~~ — non-today `events-YYYYMMDD.jsonl` auto-gzipped at boot to `.jsonl.gz`, saving disk space. recover / iter_events transparently read `.gz` files. Line count verified before compression, zero data loss ✅

### In Progress (Cognitive Continuation Hardening)

- [ ] **Behavioral Verification persistence** — `handoff_verifications` table, as a differentiating capability

### Deferred

- [ ] ~~Multi-Agent Coordination — multi-agent parallel task assignment & sync~~ → **Deferred**: falls under general orchestration, conflicts with project positioning; let upstream frameworks (LangGraph / AutoGen) handle it.
- [ ] ~~Cross-model next_steps intermediate representation~~ → **Deferred**: mainstream MCP clients use same-tier models, natural language next_steps is sufficient; over-engineering yields low ROI.
- [ ] Coding Agent deep integration — IDE-native Task panel (retained, depends on Checkpoint v2 completion)

---

## Changelog

### v0.13.1 — P1-6 Event Log Gzip Rotate

Theme: **Keep event logs forever without blowing up disk.**

**New**

- ✨ **Event log gzip rotate** (`event_log.py`): `rotate_old_files()` compresses non-today `.jsonl` files to `.jsonl.gz`. Line count verified before and after compression — zero data loss.
- ✨ **Transparent `.jsonl.gz` reading** (`event_log.py`): `_sorted_event_files()` recognizes both `.jsonl` and `.jsonl.gz`; `_iter_file()` auto-selects `open` or `gzip.open`. When both formats exist for the same date, `.jsonl` takes priority.
- ✨ **Boot auto-rotate** (`maintenance.py`): `schedule_startup_maintenance()` calls `rotate_event_logs()` in daemon thread, non-blocking.
- ✨ **Recover transparent compatibility**: `recover()` benefits via `iter_events()` — no changes needed to read from `.gz` files.

**Upgrade**

```bash
pip install -U mcp-engram      # install command unchanged
```

- Fully forward-compatible with v0.13.0: old `.jsonl` files read normally; first boot auto-compresses historical files

**Regression**

- 456 tests passed (v0.13.0's 441 + 15 new: `test_event_log_rotate`)
- 0 lint errors

### v0.13.0 — Chaos Continuity Test + Continuity Metrics

Theme: **Quantify the cognitive quality of agent cross-interruption recovery** — answer the core question "is checkpoint restore actually good enough?"

**New**

- ✨ **6-dimension Continuity Metrics engine** (`continuity.py`): auto-compute 6-dimension scores + weighted composite on every checkpoint restore. Dimensions: Goal Retention / Action Consistency / Failure Recall / Working Set Stability / Replanning Rate / Redundant Exploration.
- ✨ **MCP tool `evaluate_continuity`** (`tools.py` / `handlers.py`): LLM can proactively evaluate continuity score between any two checkpoint versions. Supports `actions_taken_after_restore` for redundant exploration measurement.
- ✨ **`restore_checkpoint` auto-attaches `continuity_score`** (`handlers.py`): auto-compares against parent_version on restore, embeds 6-dimension scores in response. LLM can judge "is this recovery quality good enough, or do I need compensation?"
- ✨ **Chaos Continuity Test suite** (`test_chaos_continuity.py`): 5 major scenario automated verification — S1: Normal Handoff (baseline) / S2: SIGTERM (atexit fires) / S3: kill -9 Crash / S4: Failure Mid-Session / S5: Working Set Drift.

**Upgrade**

```bash
pip install -U mcp-engram      # install command unchanged
```

- Fully forward-compatible with v0.12: `evaluate_continuity` is a new tool, no impact on existing clients
- `restore_checkpoint`'s `continuity_score` is optional output, old clients can ignore

**Regression**

- 441 tests passed (v0.12's 404 + 37 new: `test_continuity_metrics` 28 + `test_chaos_continuity` 9)
- 0 lint errors

### v0.12.0 — Interruption Taxonomy

Theme: Let the next agent **know how the previous agent was interrupted**, and choose the optimal recovery strategy accordingly — instead of a generic "session ended unexpectedly".

**New**

- ✨ **6 interruption categories** (`db.py`): `overflow` / `user_away` / `tool_failure` / `crash` / `rate_limit` / `unknown`, each mapped to a recovery strategy (restore_checkpoint + memory_restore_mode + hint).
- ✨ **MCP tool `report_interruption`** (`tools.py` / `handlers.py`): LLM calls this when detecting imminent interruption (e.g. context window filling, API rate limiting), records interruption reason. The reason is written to `session_lifecycle` on process exit, so the next agent receives targeted recovery advice.
- ✨ **Stale session auto-classification** (`db.py`): `cleanup_stale_sessions` now auto-classifies interruption reasons via heuristic rules: session < 2min → `crash`; ≥ 2 failure memories → `tool_failure`; otherwise → `user_away`.
- ✨ **Taxonomy-aware recall hints** (`handlers.py`): `recall_memory`'s `interrupted_sessions` no longer gives generic hints — it provides targeted recovery strategies by interruption type (`recovery_strategy` / `memory_restore_mode` / `hint`).
- ✨ **atexit interruption-aware** (`shared.py`): `_on_exit` now checks if LLM pre-reported interruption via `report_interruption`; if so, writes to session_lifecycle, otherwise marks as normal `process_exit`.

**Schema Changes**

- `session_lifecycle` gains `interruption_reason VARCHAR` + `interruption_context JSON` columns
- Fully forward-compatible: old data with `interruption_reason = NULL` is treated as `unknown`; schema migration via `ALTER TABLE ADD COLUMN IF NOT EXISTS`

**New Event Fields**

- `session.end` event gains optional fields: `interruption_reason` / `interruption_context`
- `engram recover`'s `_replay_session_end` supports replaying these fields

**Upgrade**

```bash
pip install -U mcp-engram      # install command unchanged
engram-setup doctor            # session_lifecycle table auto-adds columns
```

- Fully forward-compatible with v0.11: old sessions' `interruption_reason` is NULL, recall hint falls back to `unknown` strategy
- New `report_interruption` tool is optional; not calling it yields identical behavior to v0.11

**Regression**

- 404 tests passed (v0.11's 387 + 17 new: `test_interruption_taxonomy`)
- 0 lint errors

### v0.11.0 — Operational Hardening

Theme: On top of v0.10's "Two Laws", fill in **operational visibility** and **catastrophic growth prevention**. Zero-config, enabled by default, fully forward-compatible with v0.10.

**New**

- ✨ **Periodic Snapshot + Replay Acceleration** (`snapshot.py`): Async snapshot DuckDB file to `~/.engram/snapshots/snapshot-seq{N}-{ts}.duckdb` every N events (default 1000) or H hours (default 1). `engram-setup recover` loads from latest snapshot and only replays `seq > snapshot_seq` events — long-running engram no longer slows down recovery due to event accumulation.
- ✨ **Backup Auto-Archive Policy** (`maintenance.py`): When managed files in `~/.engram/backups/` (`memories-pre-recover-*` / `memories-pre-duckdb-upgrade-*`) exceed `ENGRAM_BACKUP_RETAIN` (default 10), oldest are archived to `backups/archive/` (**move, not delete** — recoverable).
- ✨ **DuckDB Version Upgrade Auto-Backup**: Detects `duckdb_version` minor/major changes (e.g. `1.5 → 1.6`, `0.9 → 0.10`), copies current DB to `backups/memories-pre-duckdb-upgrade-<old>-to-<new>-<ts>.duckdb` before startup, and writes a `runtime.duckdb_upgrade` event to anchor the time.
- ✨ **MCP tool `get_runtime_health`**: LLM (Claude Code / Cursor) can proactively query engram health. Returns `advice` array (readable suggestions) + full `doctor()` fields; in degraded mode, LLM can prompt user to run `engram-setup recover`.
- ✨ **`engram-setup doctor` output enhanced**: New `backups` (`live_count` / `retain` / `archive_count` / `live_recent`) and `snapshots` (`count` / `latest_seq` / `latest_size_bytes`) sections; prints archive hint when retention exceeded.
- ✨ **Recover report enhanced**: New `snapshot_used` / `snapshot_seq` fields — clearly see which snapshot this recovery started from.

**New Event Kinds** (not involved in Tier 1 replay, for ops audit only)

```
snapshot.create            # {snapshot_path, seq, db_size_bytes}
runtime.duckdb_upgrade     # {old_version, new_version, backup_path}
maintenance.backup_pruned  # {archived: [...], kept, dir}
```

**New Env Vars**

| Variable | Default | Description |
|---|---|---|
| `ENGRAM_BACKUP_RETAIN` | 10 | Number of backups to retain in `backups/`; excess archived to `archive/` |
| `ENGRAM_SNAPSHOT_INTERVAL_EVENTS` | 1000 | Trigger snapshot after this many events written |
| `ENGRAM_SNAPSHOT_INTERVAL_HOURS` | 1.0 | Maximum hours between snapshots |
| `ENGRAM_SNAPSHOT_RETAIN` | 5 | Number of snapshots to retain (oldest deleted) |

**Upgrade**

```bash
pip install -U mcp-engram      # install command unchanged
engram-setup doctor            # see backups + snapshots new sections = upgrade successful
```

- Fully forward-compatible with v0.10: if no snapshot exists, recover falls back to full replay
- No MCP client config changes needed; `get_runtime_health` is a new tool, old clients unaffected
- Background maintenance thread only starts in the main runtime process (short-lived tool scripts like doctor / recover don't trigger it)

**Regression**

- 381 tests passed (v0.10's 348 + 33 new: `test_backup_pruner` / `test_duckdb_upgrade` / `test_runtime_health_tool` / `test_snapshot`)
- 0 lint errors

### v0.10.0 — Durable Agent Runtime (Architecture Refactor)

Positioning upgrade: from *AI Memory System* to **Durable Agent Runtime**.
Primary axis: `runtime durability + execution continuity`; vector recall demoted to auxiliary.

**Two Laws**

> Event log is the only durability primitive.
> If it cannot be replayed, it is not critical state.

**New**

- ✨ **Append-only Event Log**: `~/.engram/events/events-YYYYMMDD.jsonl`, fsync writes, daily rotation; Tier 1 (task / checkpoint / session) write path fully goes through the log.
- ✨ **Replay-based Recovery**: When DuckDB is missing/corrupted, Tier 1 can be fully rebuilt from event log.
- ✨ **CLI**: `engram-setup doctor` (health check), `engram-setup recover [--since YYYYMMDD] [--promote]` (dry-run rebuild).
- ✨ **`engram_meta` table**: Exposes `schema_version` / `engram_version` / `duckdb_version` / `embedding_model` / `embedding_dim` / `embedding_stale` / `last_boot_at` for MCP client version negotiation.
- ✨ **Readonly Degraded Mode**: When DB is unwritable, enters read-only mode; write ops throw `DegradedModeError`; HTTP returns 503 + `recover_command`, MCP returns `{ok: false, code: "degraded_mode", recover_command: "engram recover"}`.
- ✨ `tasks` table pre-reserved `parent_task_id` / `retry_of_task_id` columns (not implemented yet, avoiding future breaking migrations).
- ✨ `/health` adds `db_readonly` / `embedding_stale` / `residue_files` / `engram_meta` fields.

**⚠️ Behavior Changes (Breaking-ish)**

- **DB corruption no longer silently rebuilds empty DB**: The old `os.replace(db, db + ".corrupt")` + auto-create-empty logic is removed. On corruption, throws `DatabaseCorruptionError`, original file isolated as `<db>.corrupt.<timestamp>` to `~/.engram/backups/`, user explicitly runs `engram-setup recover`.
  - To keep old behavior: `ENGRAM_ALLOW_RESET=1 engram-server run`
- **Embedding model/dimension changes no longer auto-ALTER columns**: The old "clear all + ALTER COLUMN" is removed, replaced by marking `embedding_stale=true`; vector search auto-falls back to BM25/FTS, write path unblocked.
- **WAL startup recovery path improved**: First attempts `FORCE CHECKPOINT` to salvage data; on failure, isolates WAL as `<db>.wal-recovery.<timestamp>` (with timestamp, never overwrites).
- **Shutdown auto-CHECKPOINT**: HTTP server proactively flushes WAL on close, avoiding residual WAL on next startup.

**Upgrade**

```bash
pip install -U mcp-engram          # install command unchanged
engram-setup doctor                # recommended health check after upgrade
```

- Existing `~/.engram/memories.duckdb` reused directly; schema auto-`ALTER ... ADD COLUMN IF NOT EXISTS`.
- Event log starts accumulating from this point; pre-upgrade data still relies on the DB file itself (no event log to replay).
- MCP client config doesn't need changes.

**Regression**

- 348 tests passed (17 new: `test_event_log` / `test_recover` / `test_degraded_mode`).
- 0 lint errors.

### v0.9.x (Historical)

- Checkpoint v2 — versioned cognitive checkpoint, event-first trigger (6 reason types), `restore_checkpoint` / `list_checkpoints` live.
- Task as first-class entity; Session Lifecycle; Handoff Validation; Memory Quality Score; Error-aware Memory.

---

## Contributing

Contributions welcome:

1. **Issues** — Report bugs or suggest features
2. **Pull Requests** — Fork → new branch → submit PR

```bash
git clone https://github.com/hugfeature/engram.git
cd engram
pip install -e ".[dev]"
pytest tests/ -v       # make sure tests pass
```

## Maintainer

- [@hugfeature](https://github.com/hugfeature)

## License

[MIT](https://opensource.org/licenses/MIT)

---

> **Cognitive Continuation Layer — we restore an agent's cognition, not a machine's execution.**
