# Changelog

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
