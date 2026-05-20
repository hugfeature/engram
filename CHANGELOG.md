# Changelog

## [0.19.0] — 2026-05-20

### Drift Nudge + Stats + Adaptive Checkpoint

Turn drift detection from passive observation into active correction, add runtime analytics, and make checkpoint frequency self-tuning.

### Added

- **Drift Nudge** — auto-injects a high-priority warning memory when drift exceeds threshold
  - Warning surfaces in next `recall_memory()`, nudging the agent back on track
  - Configurable: `ENGRAM_DRIFT_NUDGE` (on/off), `ENGRAM_DRIFT_NUDGE_THRESHOLD` (default 0.7)
- **Stats module** (`engram stats` / `engram report`) — runtime usage analytics from Event Journal
  - Sessions, checkpoints, drift, continuity, must-not-redo saves, memory activity
  - `engram-setup stats --days 7` for terminal table, `--json` for raw output
  - `engram-setup report --period weekly` for markdown report
- **Adaptive Checkpoint interval** — auto_save frequency adjusts based on restore rates
  - Low restore rate (< 10%) → double the interval (capped at 600s)
  - Prevents checkpoint noise when auto_save is never consumed
  - Configurable: `ENGRAM_ADAPTIVE_CHECKPOINT`, `ENGRAM_ADAPTIVE_LOW_RESTORE_RATE`, `ENGRAM_ADAPTIVE_MAX_INTERVAL`
- **Enhanced interrupt_recovery** — `recall_memory()` now returns structured recovery context
  - `completed`, `in_progress`, `must_not_redo_count` fields added to interrupt_recovery
  - `action_required` replaces `hint` with stronger resume instruction
- **New event types** for stats tracking: `checkpoint.restore`, `drift.detected`, `drift.nudge`, `continuity.evaluated`, `continuity.redundant_exploration`

### Changed

- `engram-prompt` CLAUDE.md snippet now includes mandatory `restore_checkpoint` call before any work
- Event log `RUNTIME_KINDS` expanded with 5 new event kinds

---

## [0.18.0] — 2026-05-20

### Durable Runtime Storage Architecture

Complete re-architecture of the storage layer into three well-defined tiers.

### Added

- **Tier 2 — Runtime State Store (SQLite WAL)**
  - Tasks, checkpoints, executions, sessions now stored in SQLite WAL
  - Concurrent readers + single writer — eliminates DuckDB lock contention
  - Auto-migration from DuckDB on first boot (zero manual steps)
  - Enabled by default; disable with `ENGRAM_SQLITE_TIER2=0`

- **Tier 3 — Runtime Intelligence Cache**
  - Formal definition: rebuildable layer for embeddings, FTS, drift vectors, recovery metrics
  - `engram-setup rebuild-cache` CLI command to drop & rebuild Tier 3
  - `rebuild_tier3_cache(reembed=True)` API for programmatic rebuild

- **Auto-recovery from snapshots**
  - DB corruption now triggers automatic recovery: load latest snapshot + replay incremental events
  - Agent never enters degraded mode if a snapshot exists — fully transparent

- **Snapshot Compaction** (scheduler)
  - Periodic DuckDB snapshots taken by daemon thread
  - `recover()` uses latest snapshot as base, only replays incremental events
  - Configurable: `ENGRAM_SNAPSHOT_INTERVAL_EVENTS`, `ENGRAM_SNAPSHOT_INTERVAL_HOURS`

### Changed

- `ENGRAM_SQLITE_TIER2` now defaults to **enabled** (was opt-in)
- Architecture docstring updated to reflect three-tier naming:
  - Tier 1: Event Journal (immutable, append-only)
  - Tier 2: Runtime State Store (operationally durable)
  - Tier 3: Runtime Intelligence Cache (rebuildable)

### Fixed

- Timestamp handling: `_safe_isoformat()` handles both SQLite strings and DuckDB datetime objects
- `_to_utc()` now parses ISO format strings (SQLite returns text, not datetime)
- Checkpoint confidence computation reads from SQLite when Tier 2 is active

---

## [0.17.0] — 2026-05-18

### Runtime Reliability Signals

- Interruption Intelligence: classification + severity + recoverability scoring
- Execution Drift Analysis: goal/tool/planning/constraint drift detection
- Semantic Continuity Scoring: checkpoint consistency + resume alignment
- Lightweight Recovery Heuristics: `recommend_recovery()` with hardcoded rules

---

## [0.16.1] — 2026-05-17

### Execution Continuity

- Execution Lineage: `execution_id` + `execution_sessions` table + retry chain
- Checkpoint Semantic Completeness: execution_position, blocked_reasons, open_subtasks
- Runtime Coordination: DuckDB writer isolation + single writer queue + readonly fallback
- Daemon auto-reload on package update

---

## [0.15.1] — 2026-05-10

### Stability & Recovery

- Event log rotation with gzip compression
- Snapshot scheduler for fast recovery
- Backup pruning with retention policy
- `engram-prompt` CLI for CLAUDE.md generation
