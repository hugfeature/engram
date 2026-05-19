"""Shared MCP tool schemas — used by both stdio and HTTP servers."""

from __future__ import annotations

from mcp.types import Tool

TOOL_SCHEMAS: list[Tool] = [
    Tool(
        name="recall_memory",
        description=(
            "Retrieve the most relevant memories for a query using hybrid search "
            "(40% BM25 keyword + 60% vector similarity weighted by decay strength), "
            "extended by semantic-graph BFS expansion to depth 2. "
            "Call at the start of every task or whenever prior context may be relevant. "
            "Do NOT call for purely ephemeral one-shot questions with no historical context. "
            "Use memory_type to narrow results: 'handoff' for session continuity, "
            "'failure' for bug/error history, 'progress' for feature state. "
            "Each recall increments recall_count of returned memories, slowing their decay "
            "(reinforcement side effect). No data is written or deleted."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search string; use key concepts of the current task. More specific queries return more precise results.",
                },
                "user_id": {
                    "type": "string",
                    "description": "Memory partition key. Must match the value used in store_memory.",
                    "default": "default",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of memories to return (1–20, default 5). Increase for broad context; keep low for focused lookups.",
                    "default": 5,
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional session identifier. When provided, recalled memories are tracked for session_outcome reinforcement.",
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["all", "handoff", "failure", "progress"],
                    "description": (
                        "'all' (default): everything, with the latest handoff auto-pinned to top. "
                        "'handoff': session continuity snapshots only. "
                        "'failure': structured bug/error records only. "
                        "'progress': feature/task progress snapshots only."
                    ),
                    "default": "all",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="store_memory",
        description=(
            "Store a new memory with automatic deduplication (WRITE — has side effects). "
            "Similarity thresholds determine outcome: "
            "≥0.85 → REINFORCE existing memory (increments recall_count, no new row); "
            "0.65–0.84 → MERGE compatible memories or REPLACE contradictions; "
            "<0.65 → NEW record created. "
            "Call whenever the agent learns something that should persist across sessions: "
            "user preferences, architecture decisions, validated strategies, recurring failures. "
            "importance controls decay rate and retrieval ranking: "
            "0.9–1.0 = permanent facts; 0.7–0.8 = strong preferences; "
            "0.5 = ordinary project facts; 0.2–0.3 = temporary context. "
            "category controls half-life: strategy (~38d), fact (~24d), assumption (~19d), failure (~11d). "
            "Side effects: writes to DuckDB; may modify an existing record on MERGE/REPLACE; "
            "triggers graph edge recalculation."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Memory text (one sentence preferred). Vague content degrades retrieval quality.",
                },
                "importance": {
                    "type": "number",
                    "description": "Float in [0.0, 1.0]. Invalid values are clamped. Recommended 0.4–0.8 for most memories.",
                },
                "category": {
                    "type": "string",
                    "enum": ["fact", "assumption", "failure", "strategy"],
                    "description": "Controls decay rate. strategy: λ=0.10 (~38d half-life). fact: λ=0.16 (~24d). assumption: λ=0.20 (~19d). failure: λ=0.35 (~11d).",
                    "default": "fact",
                },
                "user_id": {
                    "type": "string",
                    "description": "Memory partition key.",
                    "default": "default",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional JSON-serialisable dict for extra structured fields (e.g. failure attribution, feature state).",
                },
            },
            "required": ["content", "importance"],
        },
    ),
    Tool(
        name="update_memory",
        description=(
            "Update the content and/or importance of an existing memory by ID (WRITE — has side effects). "
            "Use when you already know the memory_id (returned by store_memory or recall_memory) "
            "and want to correct or extend a specific record without triggering full deduplication. "
            "For content that contradicts an existing memory when you don't have the ID, "
            "use store_memory instead — contradiction detection handles replacement automatically. "
            "Passing an invalid or non-existent memory_id returns an error; no record is created. "
            "Side effects: overwrites content; recalculates embedding vector and graph edges. "
            "Does NOT reset recall_count or strength."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "integer",
                    "description": "ID of the memory to update. Must be a valid existing ID.",
                },
                "new_content": {
                    "type": "string",
                    "description": "Replacement text. Triggers embedding and graph edge recalculation.",
                },
                "importance": {
                    "type": "number",
                    "description": "New importance in [0.0, 1.0]. If omitted, existing importance is preserved.",
                },
            },
            "required": ["memory_id", "new_content"],
        },
    ),
    Tool(
        name="consolidate_memory",
        description=(
            "Manually trigger a full memory consolidation cycle (WRITE — has side effects). "
            "Clusters similar memories (cosine ≥0.70), merges redundant records, rebuilds graph edges, "
            "and prunes memories with strength <0.05 (unless graph-protected by strong neighbours). "
            "Runs automatically every 12 hours in the background. "
            "Call manually only for immediate clean-up: after large batch store_memory calls, "
            "or before querying memory_stats for accurate counts. "
            "Do NOT call during sensitive audits requiring byte-for-byte historical wording stability. "
            "Side effects: DELETES memories with strength <0.05 that are not graph-protected; "
            "MODIFIES content of merged memories; recalculates all embeddings and graph edges. "
            "This operation is irreversible."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Memory partition key to consolidate.",
                    "default": "default",
                },
            },
        },
    ),
    Tool(
        name="session_handoff",
        description=(
            "Record a structured end-of-session summary for cross-session continuity (WRITE — has side effects). "
            "Creates a searchable handoff snapshot and may trigger checkpointing. "
            "Call once near session end or before an agent switch whenever multi-session continuity matters. "
            "Avoid calling repeatedly during active implementation loops. "
            "Do NOT call for short one-off interactions. "
            "Stored as a high-importance 'strategy' memory (λ=0.10, ~38d half-life) "
            "so it survives long gaps between sessions. "
            "Side effects: writes one or more memory records; triggers deduplication against prior handoff summaries."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "One-paragraph summary of the session's work. Required.",
                },
                "completed": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Task/subtask strings finished in this session.",
                },
                "in_progress": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Items currently being worked on.",
                },
                "blocked": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Blocked items with reason for each block.",
                },
                "next_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Recommended actions for the next session.",
                },
                "user_id": {
                    "type": "string",
                    "description": "Memory partition key.",
                    "default": "default",
                },
                "task_id": {
                    "type": "integer",
                    "description": "Optional task ID to associate this handoff with a tracked task.",
                },
            },
            "required": ["summary"],
        },
    ),
    Tool(
        name="memory_stats",
        description=(
            "Return aggregate statistics for a user's memory partition (read-only, no side effects). "
            "Includes total record count, per-category breakdown, average strength, last maintenance time, "
            "failure trends by component and severity, and active feature progress snapshots. "
            "Use to get an overview of memory health, diagnose which components have the most failures, "
            "or surface all in-progress features before planning the next session. "
            "For accurate counts, call consolidate_memory first if a large batch was recently stored."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Memory partition key.",
                    "default": "default",
                },
            },
        },
    ),
    Tool(
        name="track_failure",
        description=(
            "Record a structured failure event — bug, test failure, or deployment incident (WRITE — has side effects). "
            "Enforces consistent schema so failure patterns can be queried by component or severity across sessions. "
            "Call whenever a non-trivial error is encountered during coding or deployment. "
            "Do NOT use for expected/handled exceptions that require no follow-up. "
            "severity maps to importance: critical→0.9, major→0.7, minor→0.5. "
            "Stored under category 'failure' (λ=0.35, ~11d half-life) so stale fixes expire naturally. "
            "Side effects: writes one memory record; increments failure counters in memory_stats."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "error": {
                    "type": "string",
                    "description": "Human-readable description of the error or failure.",
                },
                "component": {
                    "type": "string",
                    "description": "Logical component or module name (e.g. 'auth', 'payment', 'ci-pipeline'). Used for aggregated failure stats.",
                },
                "root_cause": {
                    "type": "string",
                    "description": "Diagnosed underlying cause (why it failed).",
                },
                "severity": {
                    "type": "string",
                    "enum": ["critical", "major", "minor"],
                    "description": "Impact severity. Maps to importance: critical→0.9, major→0.7, minor→0.5.",
                    "default": "major",
                },
                "fix": {
                    "type": "string",
                    "description": "Fix applied or recommended fix.",
                },
                "related_test_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "IDs or names of test cases that cover this failure.",
                },
                "user_id": {
                    "type": "string",
                    "description": "Memory partition key.",
                    "default": "default",
                },
                "task_id": {
                    "type": "integer",
                    "description": "Optional task ID to associate this failure with a tracked task.",
                },
            },
            "required": ["error", "component"],
        },
    ),
    Tool(
        name="track_progress",
        description=(
            "Snapshot the current status of a feature or task (WRITE — has side effects). "
            "Creates a searchable record so progress survives context resets and can be queried in future sessions. "
            "Call at meaningful milestones: start, significant progress, blocker encountered, completion. "
            "Repeated calls for the same feature name overwrite the previous snapshot via deduplication "
            "— only the latest state is kept. "
            "Use a stable slug for 'feature' across calls (e.g. 'login-flow-refactor'). "
            "status maps to importance: blocked→0.9, in_progress→0.8, planning→0.7, done→0.5. "
            "Stored under 'strategy' (λ=0.10, ~38d half-life). Completed features decay without manual cleanup. "
            "Side effects: writes or updates one memory record; aggregated in memory_stats under engineering.features."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "feature": {
                    "type": "string",
                    "description": "Stable slug identifying the feature across sessions (e.g. 'login-flow-refactor', 'engram-v0.4'). Used for deduplication.",
                },
                "status": {
                    "type": "string",
                    "enum": ["planning", "in_progress", "blocked", "review", "done"],
                    "description": "Current status. Maps to importance: blocked→0.9, in_progress→0.8, planning→0.7, review→0.6, done→0.5.",
                },
                "completion": {
                    "type": "number",
                    "description": "Completion percentage (0–100).",
                    "default": 0,
                },
                "blockers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Blocking issues or dependencies preventing progress.",
                },
                "quality_score": {
                    "type": "number",
                    "description": "Subjective quality assessment 0.0–1.0 (e.g. test coverage, code review outcome).",
                },
                "notes": {
                    "type": "string",
                    "description": "Free-form progress notes.",
                },
                "user_id": {
                    "type": "string",
                    "description": "Memory partition key.",
                    "default": "default",
                },
                "task_id": {
                    "type": "integer",
                    "description": "Optional task ID to associate this progress snapshot with a tracked task.",
                },
            },
            "required": ["feature", "status"],
        },
    ),
    Tool(
        name="session_outcome",
        description=(
            "Mark a session as successful or failed and adjust memory importance accordingly (WRITE — has side effects). "
            "When outcome is 'success', importance of memories recalled in the session is reinforced. "
            "When outcome is 'failure', notes are stored as a failure lesson and recalled memories are down-weighted. "
            "session_id must match the value passed to recall_memory during the session. "
            "Call once at the end of a session when a clear success/failure determination can be made. "
            "Side effects: modifies importance of recalled memories; may store a failure lesson memory."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session identifier — must match the session_id used in recall_memory calls during this session.",
                },
                "outcome": {
                    "type": "string",
                    "enum": ["success", "failure"],
                    "description": "'success': reinforces recalled memory importance. 'failure': down-weights recalled memories and stores notes as a failure lesson.",
                },
                "notes": {
                    "type": "string",
                    "description": "Optional outcome notes. When outcome is 'failure', stored as a failure lesson memory.",
                },
                "user_id": {
                    "type": "string",
                    "description": "Memory partition key.",
                    "default": "default",
                },
            },
            "required": ["session_id", "outcome"],
        },
    ),
    Tool(
        name="create_task",
        description=(
            "Create a new tracked task (WRITE — has side effects). "
            "Tasks are first-class entities that group handoffs, progress snapshots, and failure records "
            "for structured cross-session continuity. "
            "Use when starting a multi-session effort that benefits from checkpoint/restore support. "
            "Returns a task_id that can be passed to session_handoff, track_failure, track_progress, "
            "restore_checkpoint, and evaluate_continuity. "
            "Side effects: creates a task record; does not create any memory records."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Stable task slug (e.g. 'login-flow-refactor', 'engram-v0.8'). Used for identification across sessions.",
                },
                "goal": {
                    "type": "string",
                    "description": "Task goal or objective description.",
                    "default": "",
                },
                "status": {
                    "type": "string",
                    "enum": ["planning", "in_progress", "blocked", "review", "done", "cancelled"],
                    "description": "Initial task status.",
                    "default": "planning",
                },
                "user_id": {
                    "type": "string",
                    "description": "Memory partition key.",
                    "default": "default",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional JSON-serialisable structured metadata.",
                },
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="update_task",
        description=(
            "Update an existing task's status, goal, or metadata (WRITE — has side effects). "
            "Use when task status changes (e.g. planning→in_progress, in_progress→blocked) "
            "or when the goal needs correction. "
            "Passing an invalid task_id returns an error; no record is created. "
            "metadata update replaces the existing metadata entirely — include all fields you want to preserve. "
            "Side effects: modifies the task record only; does not affect associated memory records."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "ID of the task to update. Must be a valid existing task ID.",
                },
                "status": {
                    "type": "string",
                    "enum": ["planning", "in_progress", "blocked", "review", "done", "cancelled"],
                    "description": "New task status.",
                },
                "goal": {
                    "type": "string",
                    "description": "Updated goal description.",
                },
                "metadata": {
                    "type": "object",
                    "description": "Updated metadata. Replaces existing metadata entirely.",
                },
            },
            "required": ["task_id"],
        },
    ),
    Tool(
        name="get_task",
        description=(
            "Retrieve a task by ID with all associated memories: handoffs, failures, and progress snapshots "
            "(read-only, no side effects). "
            "Use this when a new agent takes over a task to understand full context before acting. "
            "Prefer restore_checkpoint when resuming an interrupted task — it returns a structured "
            "continuation package with negative memory (must_not_redo) and invariants (must_preserve). "
            "get_task is better for inspection and auditing without the filtering that restore_checkpoint applies."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "ID of the task to retrieve.",
                },
                "user_id": {
                    "type": "string",
                    "description": "Memory partition key.",
                    "default": "default",
                },
            },
            "required": ["task_id"],
        },
    ),
    Tool(
        name="list_tasks",
        description=(
            "List all tasks for a user, optionally filtered by status (read-only, no side effects). "
            "Use to get an overview of all tracked tasks and their current states before planning. "
            "Filter by status to surface only actionable tasks (e.g. status='blocked' or 'in_progress'). "
            "Does not return associated memories — use get_task or restore_checkpoint for full context."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Memory partition key.",
                    "default": "default",
                },
                "status": {
                    "type": "string",
                    "enum": ["planning", "in_progress", "blocked", "review", "done", "cancelled"],
                    "description": "Optional filter. Omit to return all tasks regardless of status.",
                },
            },
        },
    ),
    Tool(
        name="restore_checkpoint",
        description=(
            "Restore a structured continuation package from a task checkpoint (read-only, no side effects). "
            "Use when a new agent takes over an interrupted task. "
            "Returns: goal, completed/in_progress/blocked items, must_not_redo (negative memory of failed approaches), "
            "must_preserve (invariants that must not be broken), preferred_next, working_set, and continuation_confidence. "
            "memory_restore_mode controls context pollution risk: "
            "SELECTIVE (default): only high-importance or failure memories (cap 10) — recommended for most cases. "
            "FULL: all task memories (cap 20) — use when full context is needed. "
            "NONE: continuation package only, no memories — use when context window is near limit. "
            "Prefer this over get_task when resuming interrupted work."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "ID of the task to restore.",
                },
                "version": {
                    "type": "integer",
                    "description": "Specific checkpoint version to restore. Omit for the latest.",
                },
                "memory_restore_mode": {
                    "type": "string",
                    "enum": ["FULL", "SELECTIVE", "NONE"],
                    "description": "FULL: all task memories (cap 20). SELECTIVE (default): importance≥0.5 OR failure memories (cap 10). NONE: no memories, continuation package only.",
                    "default": "SELECTIVE",
                },
                "user_id": {
                    "type": "string",
                    "description": "Memory partition key.",
                    "default": "default",
                },
            },
            "required": ["task_id"],
        },
    ),
    Tool(
        name="list_checkpoints",
        description=(
            "List checkpoint history for a task, latest first (read-only, no side effects). "
            "Returns checkpoint metadata only (version, timestamp, summary) — not the full state. "
            "Use to inspect checkpoint history before deciding which version to restore, "
            "or to debug continuity issues between agent handoffs. "
            "Use restore_checkpoint to load a specific version's full continuation package."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "ID of the task whose checkpoints to list.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max checkpoints to return (1–100, default 10).",
                    "default": 10,
                },
                "user_id": {
                    "type": "string",
                    "description": "Memory partition key.",
                    "default": "default",
                },
            },
            "required": ["task_id"],
        },
    ),
    Tool(
        name="get_runtime_health",
        description=(
            "Return a read-only runtime health report for the engram backend (no side effects). "
            "Reports: DB status (readonly flag, embedding_stale flag), event log summary, "
            "backup inventory, residue files (corruption indicators), and engram_meta. "
            "Call proactively at session start to detect issues requiring operator action "
            "(e.g. residue files present → run 'engram-setup recover'). "
            "Call reactively when memory tools return 'degraded_mode' errors. "
            "Always safe to call; never modifies state."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
    Tool(
        name="report_interruption",
        description=(
            "Record an imminent interruption reason before session end (WRITE — has side effects). "
            "Call BEFORE the session ends when you detect: context window overflow, rate limiting, "
            "or repeated tool failures. "
            "The reason is stored and flushed to session_lifecycle on process exit, "
            "enabling the next agent to receive a targeted recovery strategy "
            "instead of a generic 'session ended unexpectedly' hint. "
            "Do NOT call for normal planned session ends — use session_handoff instead. "
            "If session_id is provided, the session is immediately closed with the given reason. "
            "Side effects: writes interruption record; triggers session lifecycle flush on exit."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "enum": ["overflow", "user_away", "tool_failure", "crash", "rate_limit", "unknown"],
                    "description": (
                        "'overflow': context window full (include token_count in context). "
                        "'rate_limit': API throttling (include error message in context). "
                        "'tool_failure': consecutive tool errors. "
                        "'user_away': user closed session or went inactive. "
                        "'crash': unexpected process death. "
                        "'unknown': cannot determine cause."
                    ),
                },
                "context": {
                    "type": "object",
                    "description": "Optional structured context. Examples: {'token_count': 195000, 'max_tokens': 200000} for overflow; {'error': '429 Too Many Requests'} for rate_limit.",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session identifier. If provided, immediately closes the session with the given reason.",
                },
                "user_id": {
                    "type": "string",
                    "description": "Memory partition key.",
                    "default": "default",
                },
            },
            "required": ["reason"],
        },
    ),
    Tool(
        name="evaluate_continuity",
        description=(
            "Evaluate how well cognitive state survived across checkpoint versions (read-only, no side effects). "
            "Returns 6 metrics: Goal Retention, Action Consistency, Failure Recall, "
            "Working Set Stability, Replanning Rate, Redundant Exploration, "
            "plus a weighted composite score. "
            "Call after restore_checkpoint to quantify recovery quality. "
            "Call with before_version and after_version to compare any two checkpoint versions. "
            "Pass actions_taken_after_restore to score Redundant Exploration "
            "(detects whether the new agent repeated already-failed approaches). "
            "Useful for diagnosing poor handoffs and tuning memory_restore_mode."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "ID of the task to evaluate.",
                },
                "before_version": {
                    "type": "integer",
                    "description": "Checkpoint version before interruption. Omit for second-to-last.",
                },
                "after_version": {
                    "type": "integer",
                    "description": "Checkpoint version after restore. Omit for latest.",
                },
                "actions_taken_after_restore": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Actions the new agent performed after restoring. Used for Redundant Exploration scoring.",
                },
                "user_id": {
                    "type": "string",
                    "description": "Memory partition key.",
                    "default": "default",
                },
            },
            "required": ["task_id"],
        },
    ),
    # ================================================================
    # Execution Lineage — v0.16 Durable Runtime Continuity
    # ================================================================
    Tool(
        name="start_execution",
        description=(
            "Start a new execution lineage — a continuous runtime intent that persists "
            "across interruptions, retries, and session boundaries (WRITE — has side effects). "
            "An execution is the durable entity; tasks within it are individual attempts. "
            "Call when beginning a non-trivial multi-step task that may survive interruptions. "
            "Returns execution_id (UUID) and the first task_id (attempt #1). "
            "If resuming from a previous checkpoint, pass origin_checkpoint to establish lineage."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "The high-level goal of this execution. Should be stable across retries.",
                },
                "user_id": {
                    "type": "string",
                    "description": "Memory partition key.",
                    "default": "default",
                },
                "origin_checkpoint": {
                    "type": "string",
                    "description": "Checkpoint ID this execution spawns from. Establishes resume lineage.",
                },
            },
            "required": ["goal"],
        },
    ),
    Tool(
        name="retry_task",
        description=(
            "Retry a failed/interrupted task within the same execution lineage "
            "(WRITE — has side effects). Creates a new task as the next attempt, "
            "preserving the retry chain for lineage tracing. "
            "The original task is marked cancelled. The new task inherits the same "
            "execution_id and goal. Call when an agent resumes after interruption "
            "or when a task fails and needs re-execution."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "ID of the task to retry (the failed/interrupted one).",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this retry is needed (e.g. 'context_overflow', 'tool_failure', 'user_correction').",
                },
                "user_id": {
                    "type": "string",
                    "description": "Memory partition key.",
                    "default": "default",
                },
            },
            "required": ["task_id"],
        },
    ),
    Tool(
        name="spawn_subtask",
        description=(
            "Spawn a subtask within an existing execution lineage "
            "(WRITE — has side effects). Creates a child task under the parent, "
            "sharing the same execution_id. Use when decomposing a complex task "
            "into smaller units that should be tracked as part of the same execution."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "parent_task_id": {
                    "type": "integer",
                    "description": "ID of the parent task.",
                },
                "name": {
                    "type": "string",
                    "description": "Short name for the subtask.",
                },
                "goal": {
                    "type": "string",
                    "description": "Goal of the subtask. Defaults to name if omitted.",
                },
                "user_id": {
                    "type": "string",
                    "description": "Memory partition key.",
                    "default": "default",
                },
                "checkpoint_id": {
                    "type": "string",
                    "description": "Optional checkpoint ID this subtask spawns from.",
                },
            },
            "required": ["parent_task_id", "name"],
        },
    ),
    Tool(
        name="trace_execution",
        description=(
            "Trace the full execution lineage — all attempts, retries, and spawned subtasks "
            "(read-only, no side effects). Returns the execution metadata, all task nodes "
            "with their relationships (retry_of, parent, previous), and identifies the "
            "current active task. Use to understand where execution is, what's been tried, "
            "and what the retry history looks like."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "execution_id": {
                    "type": "string",
                    "description": "UUID of the execution to trace.",
                },
                "user_id": {
                    "type": "string",
                    "description": "Memory partition key.",
                    "default": "default",
                },
            },
            "required": ["execution_id"],
        },
    ),
    Tool(
        name="end_execution",
        description=(
            "Mark an execution as completed, abandoned, or paused "
            "(WRITE — has side effects). Call when the execution's root goal is "
            "achieved, permanently given up, or deliberately suspended for later."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "execution_id": {
                    "type": "string",
                    "description": "UUID of the execution to end.",
                },
                "status": {
                    "type": "string",
                    "enum": ["completed", "abandoned", "paused"],
                    "description": "'completed': goal achieved. 'abandoned': permanently given up. 'paused': suspended for later.",
                    "default": "completed",
                },
                "user_id": {
                    "type": "string",
                    "description": "Memory partition key.",
                    "default": "default",
                },
            },
            "required": ["execution_id"],
        },
    ),
    # ---- v0.17: Runtime Reliability Signal tools ----
    Tool(
        name="detect_drift",
        description=(
            "Detect execution drift after a checkpoint restore or retry "
            "(READ — no side effects). Compares your current state against the "
            "checkpoint baseline to identify goal drift, tool drift, planning drift, "
            "and constraint drift. Call after restoring a checkpoint and performing "
            "some actions to check whether execution is still on track. "
            "Returns a composite drift score (0.0=on track, 1.0=fully diverged) "
            "and per-dimension signals."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "The task to check drift for.",
                },
                "current_goal": {
                    "type": "string",
                    "description": "Your current stated goal (what you are working on right now).",
                },
                "tools_used": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of tool names you have used since the last checkpoint restore.",
                },
                "actions_taken": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of actions performed since restore (used for constraint violation detection).",
                },
                "in_progress": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Items you are currently working on.",
                },
                "violated_constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Any active_constraints you know you have violated.",
                },
                "user_id": {
                    "type": "string",
                    "description": "Memory partition key.",
                    "default": "default",
                },
            },
            "required": ["task_id", "current_goal"],
        },
    ),
    Tool(
        name="score_recovery",
        description=(
            "Score the semantic continuity of a recovery "
            "(READ — no side effects). Call after restoring a checkpoint to measure "
            "how well you have re-oriented. Returns goal_alignment, constraint_retention, "
            "task_position_alignment, tool_behavior_stability, retry_degradation, "
            "and an overall recovery_confidence score."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "The task whose recovery to score.",
                },
                "goal": {
                    "type": "string",
                    "description": "Your current stated goal after restore.",
                },
                "completed": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Items you recognize as already completed.",
                },
                "in_progress": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Items you are currently working on.",
                },
                "must_not_redo": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Items you acknowledge you must NOT redo.",
                },
                "active_constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Constraints you are aware of and will respect.",
                },
                "tools_used": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tools you have used since restore.",
                },
                "user_id": {
                    "type": "string",
                    "description": "Memory partition key.",
                    "default": "default",
                },
            },
            "required": ["task_id", "goal"],
        },
    ),
    Tool(
        name="recommend_recovery",
        description=(
            "Get a lightweight recovery recommendation for a task "
            "(READ — no side effects). Returns a recommended action "
            "(restore_checkpoint / start_fresh / abandon / show_summary) "
            "based on interruption reason, retry depth, and checkpoint freshness. "
            "This is NOT a policy engine — just heuristics."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "The task to get recovery recommendation for.",
                },
                "interruption_reason": {
                    "type": "string",
                    "enum": ["overflow", "user_away", "tool_failure", "crash", "rate_limit", "unknown"],
                    "description": "Why the execution was interrupted.",
                },
                "retry_count": {
                    "type": "integer",
                    "description": "How many times this has already been retried.",
                    "default": 0,
                },
                "user_id": {
                    "type": "string",
                    "description": "Memory partition key.",
                    "default": "default",
                },
            },
            "required": ["task_id"],
        },
    ),
]