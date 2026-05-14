"""Shared MCP tool schemas — used by both stdio and HTTP servers."""

from __future__ import annotations

from mcp.types import Tool

TOOL_SCHEMAS: list[Tool] = [
    Tool(
        name="recall_memory",
        description="Search memories by semantic similarity. Call at the start of every task. "
                    "Use memory_type to filter by type (e.g. 'handoff' to find session handoffs).",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords or sentence to search for",
                },
                "user_id": {
                    "type": "string",
                    "description": "User identifier",
                    "default": "default",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5,
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional session identifier for tracking which memories were used",
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["all", "handoff", "failure", "progress"],
                    "description": "Filter results by memory type. 'handoff' returns session handoffs, "
                                   "'failure' returns failure records, 'progress' returns progress snapshots. "
                                   "Default 'all' returns everything with the latest handoff auto-pinned to top.",
                    "default": "all",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="store_memory",
        description="Store a new memory (WRITE with side effects). Automatically deduplicates against existing memories "
                    "and may reinforce/merge/replace an existing memory instead of creating a new row. "
                    "importance should be in [0.0, 1.0] (recommended 0.4-0.8; invalid values are clamped).",
        inputSchema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The memory content (one sentence preferred)",
                },
                "importance": {
                    "type": "number",
                    "description": "0.0-1.0, how important this memory is",
                },
                "category": {
                    "type": "string",
                    "enum": ["fact", "assumption", "failure", "strategy"],
                    "description": "Memory category (controls decay rate)",
                    "default": "fact",
                },
                "user_id": {
                    "type": "string",
                    "description": "User identifier",
                    "default": "default",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional structured metadata (e.g. failure attribution, feature state)",
                },
            },
            "required": ["content", "importance"],
        },
    ),
    Tool(
        name="update_memory",
        description="Update an existing memory by ID with new content.",
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "integer",
                    "description": "ID of the memory to update",
                },
                "new_content": {
                    "type": "string",
                    "description": "New content for the memory",
                },
                "importance": {
                    "type": "number",
                    "description": "New importance (0.0-1.0), optional",
                },
            },
            "required": ["memory_id", "new_content"],
        },
    ),
    Tool(
        name="consolidate_memory",
        description="Scan all memories and auto-merge similar ones (WRITE with side effects). "
                    "Reduces bloat but can rewrite existing memory content/importance after merge. "
                    "Do not call during sensitive audits that require byte-for-byte historical wording stability.",
        inputSchema={
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "User identifier",
                    "default": "default",
                },
            },
        },
    ),
    Tool(
        name="session_handoff",
        description="Record structured end-of-session state for cross-session continuity (WRITE with side effects). "
                    "Creates a searchable handoff snapshot and may trigger checkpointing. "
                    "Call near session end or before agent switch; avoid calling repeatedly during active implementation loops.",
        inputSchema={
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "One-paragraph summary of the session's work",
                },
                "completed": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of completed items",
                },
                "in_progress": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of in-progress items",
                },
                "blocked": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of blocked items with reasons",
                },
                "next_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Recommended next actions",
                },
                "user_id": {
                    "type": "string",
                    "description": "User identifier",
                    "default": "default",
                },
                "task_id": {
                    "type": "integer",
                    "description": "Optional task ID to associate this handoff with a tracked task",
                },
            },
            "required": ["summary"],
        },
    ),
    Tool(
        name="memory_stats",
        description="Get memory system statistics: total count, category distribution, average strength, last maintenance time.",
        inputSchema={
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "User identifier",
                    "default": "default",
                },
            },
        },
    ),
    Tool(
        name="track_failure",
        description="Record a structured failure event (bug, test failure, deployment issue). "
                    "Enforces consistent schema for pattern analysis across sessions.",
        inputSchema={
            "type": "object",
            "properties": {
                "error": {
                    "type": "string",
                    "description": "The error message or failure description",
                },
                "component": {
                    "type": "string",
                    "description": "Component/module where the failure occurred (e.g. 'auth', 'payment', 'ci-pipeline')",
                },
                "root_cause": {
                    "type": "string",
                    "description": "Root cause analysis (why it failed)",
                },
                "severity": {
                    "type": "string",
                    "enum": ["critical", "major", "minor"],
                    "description": "Impact severity",
                    "default": "major",
                },
                "fix": {
                    "type": "string",
                    "description": "How it was fixed, or proposed fix",
                },
                "related_test_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "IDs or names of related test cases",
                },
                "user_id": {
                    "type": "string",
                    "description": "User identifier",
                    "default": "default",
                },
                "task_id": {
                    "type": "integer",
                    "description": "Optional task ID to associate this failure with a tracked task",
                },
            },
            "required": ["error", "component"],
        },
    ),
    Tool(
        name="track_progress",
        description="Record a feature or task progress snapshot. "
                    "Creates a searchable record for cross-session continuity.",
        inputSchema={
            "type": "object",
            "properties": {
                "feature": {
                    "type": "string",
                    "description": "Feature or task name (e.g. 'login-flow-refactor', 'engram-v0.4')",
                },
                "status": {
                    "type": "string",
                    "enum": ["planning", "in_progress", "blocked", "review", "done"],
                    "description": "Current status",
                },
                "completion": {
                    "type": "number",
                    "description": "Completion percentage (0-100)",
                    "default": 0,
                },
                "blockers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Current blockers preventing progress",
                },
                "quality_score": {
                    "type": "number",
                    "description": "Quality assessment 0.0-1.0 (test pass rate, coverage, etc.)",
                },
                "notes": {
                    "type": "string",
                    "description": "Free-form progress notes",
                },
                "user_id": {
                    "type": "string",
                    "description": "User identifier",
                    "default": "default",
                },
                "task_id": {
                    "type": "integer",
                    "description": "Optional task ID to associate this progress snapshot with a tracked task",
                },
            },
            "required": ["feature", "status"],
        },
    ),
    Tool(
        name="session_outcome",
        description="Mark a session as successful or failed. Adjusts importance of memories recalled in the session based on outcome.",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "The session identifier (must match session_id used in recall_memory)",
                },
                "outcome": {
                    "type": "string",
                    "enum": ["success", "failure"],
                    "description": "Whether the session succeeded or failed",
                },
                "notes": {
                    "type": "string",
                    "description": "Optional notes about the outcome (used as failure lesson when outcome is failure)",
                },
                "user_id": {
                    "type": "string",
                    "description": "User identifier",
                    "default": "default",
                },
            },
            "required": ["session_id", "outcome"],
        },
    ),
    Tool(
        name="create_task",
        description="Create a new tracked task. Tasks are first-class entities that group handoffs, "
                    "progress snapshots, and failure records for cross-session continuity.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Task name (e.g. 'login-flow-refactor', 'engram-v0.8')",
                },
                "goal": {
                    "type": "string",
                    "description": "Task goal / objective description",
                    "default": "",
                },
                "status": {
                    "type": "string",
                    "enum": ["planning", "in_progress", "blocked", "review", "done", "cancelled"],
                    "description": "Initial task status",
                    "default": "planning",
                },
                "user_id": {
                    "type": "string",
                    "description": "User identifier",
                    "default": "default",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional structured metadata",
                },
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="update_task",
        description="Update an existing task's status, goal, or metadata.",
        inputSchema={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "ID of the task to update",
                },
                "status": {
                    "type": "string",
                    "enum": ["planning", "in_progress", "blocked", "review", "done", "cancelled"],
                    "description": "New task status",
                },
                "goal": {
                    "type": "string",
                    "description": "Updated goal description",
                },
                "metadata": {
                    "type": "object",
                    "description": "Updated metadata (replaces existing)",
                },
            },
            "required": ["task_id"],
        },
    ),
    Tool(
        name="get_task",
        description="Get a task by ID with all associated memories (handoffs, failures, progress snapshots). "
                    "Use this when a new Agent takes over to understand the full task context.",
        inputSchema={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "ID of the task to retrieve",
                },
                "user_id": {
                    "type": "string",
                    "description": "User identifier",
                    "default": "default",
                },
            },
            "required": ["task_id"],
        },
    ),
    Tool(
        name="list_tasks",
        description="List all tasks for a user, optionally filtered by status. "
                    "Use this to see all tracked tasks and their current states.",
        inputSchema={
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "User identifier",
                    "default": "default",
                },
                "status": {
                    "type": "string",
                    "enum": ["planning", "in_progress", "blocked", "review", "done", "cancelled"],
                    "description": "Optional filter by task status",
                },
            },
        },
    ),
    Tool(
        name="restore_checkpoint",
        description="Restore a constrained continuation package from a task checkpoint. "
                    "Use this when a new Agent takes over an interrupted task. Returns goal, "
                    "completed/in_progress/blocked items, must_not_redo (negative memory), "
                    "must_preserve (invariants), preferred_next, working_set, and "
                    "continuation_confidence. Memory recall is controlled by memory_restore_mode "
                    "to mitigate context pollution.",
        inputSchema={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "ID of the task to restore",
                },
                "version": {
                    "type": "integer",
                    "description": "Specific checkpoint version. Omit for the latest.",
                },
                "memory_restore_mode": {
                    "type": "string",
                    "enum": ["FULL", "SELECTIVE", "NONE"],
                    "description": "FULL: all task memories (cap 20). "
                                   "SELECTIVE (default): importance>=0.5 OR failure (cap 10). "
                                   "NONE: no related memories, just the continuation package.",
                    "default": "SELECTIVE",
                },
                "user_id": {
                    "type": "string",
                    "description": "User identifier",
                    "default": "default",
                },
            },
            "required": ["task_id"],
        },
    ),
    Tool(
        name="list_checkpoints",
        description="List checkpoint history for a task (latest first). "
                    "Returns checkpoint metadata only (no full state) for debugging "
                    "and visualization. Use restore_checkpoint to load a specific version.",
        inputSchema={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "ID of the task",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max checkpoints to return (1-100)",
                    "default": 10,
                },
                "user_id": {
                    "type": "string",
                    "description": "User identifier",
                    "default": "default",
                },
            },
            "required": ["task_id"],
        },
    ),
    Tool(
        name="get_runtime_health",
        description="Read-only runtime health report for the engram backend. "
                    "Returns DB status (readonly / embedding_stale), event log "
                    "summary (kinds + max seq), backups inventory, residue "
                    "files (corruption indicators), and engram_meta. Call this "
                    "when memory tools start returning 'degraded_mode' errors, "
                    "or proactively at session start to detect operator-action "
                    "needed (e.g. residue files present → suggest "
                    "`engram-setup recover`). Always safe to call; never "
                    "modifies state.",
        inputSchema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
    Tool(
        name="report_interruption",
        description="Report an imminent interruption reason so the session taxonomy is "
                    "recorded for the next Agent. Call this BEFORE the session ends when "
                    "you detect: context window overflow, rate limiting, or repeated tool "
                    "failures. The reason is stored and flushed to session_lifecycle on "
                    "process exit, enabling the next Agent to receive a targeted recovery "
                    "strategy instead of a generic 'session ended unexpectedly' hint.",
        inputSchema={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "enum": ["overflow", "user_away", "tool_failure",
                             "crash", "rate_limit", "unknown"],
                    "description": "Why the session is being interrupted. "
                                   "'overflow': context window full. "
                                   "'rate_limit': API throttling. "
                                   "'tool_failure': consecutive tool errors. "
                                   "'user_away': user closed/inactive. "
                                   "'crash': unexpected process death. "
                                   "'unknown': cannot determine.",
                },
                "context": {
                    "type": "object",
                    "description": "Optional structured context (e.g. "
                                   "{'token_count': 195000, 'max_tokens': 200000} "
                                   "for overflow, or {'error': '429 Too Many Requests'} "
                                   "for rate_limit).",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session identifier. If provided, the session "
                                   "is immediately closed with the given reason.",
                },
                "user_id": {
                    "type": "string",
                    "description": "User identifier",
                    "default": "default",
                },
            },
            "required": ["reason"],
        },
    ),
    Tool(
        name="evaluate_continuity",
        description="Evaluate how well cognitive state survived across checkpoint versions. "
                    "Returns 6 continuity metrics: Goal Retention, Action Consistency, "
                    "Failure Recall, Working Set Stability, Replanning Rate, Redundant "
                    "Exploration, plus a weighted composite score. Call after restoring a "
                    "checkpoint to quantify recovery quality, or to compare any two "
                    "checkpoint versions of the same task.",
        inputSchema={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "ID of the task to evaluate",
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
                    "description": "Actions the new Agent performed after restoring "
                                   "(for redundant_exploration scoring). Optional.",
                },
                "user_id": {
                    "type": "string",
                    "description": "User identifier",
                    "default": "default",
                },
            },
            "required": ["task_id"],
        },
    ),
]

# --- Tool Definition Quality enrichment (Glama scoring helpers) ---
# Add concise, structured usage and behavior notes to every tool description.
_TDQ_NOTES: dict[str, dict[str, str]] = {
    "recall_memory": {
        "when": "Call at session/task start or before a major decision.",
        "side_effects": "Read-only.",
    },
    "store_memory": {
        "when": "Call when a new durable fact/lesson should survive sessions.",
        "side_effects": "Writes memory; may dedup as reinforce/merge/replace.",
    },
    "update_memory": {
        "when": "Call only when a specific memory_id must be corrected.",
        "side_effects": "Writes memory content/importance.",
    },
    "consolidate_memory": {
        "when": "Call during maintenance windows to reduce memory bloat.",
        "side_effects": "Writes/rewrites memory rows via merges.",
    },
    "session_handoff": {
        "when": "Call near session end or before agent handoff.",
        "side_effects": "Writes handoff and may trigger checkpoint creation.",
    },
    "memory_stats": {"when": "Call for diagnostics and monitoring.", "side_effects": "Read-only."},
    "track_failure": {"when": "Call immediately after a failure is understood.", "side_effects": "Writes failure memory and checkpoint context."},
    "track_progress": {"when": "Call after meaningful progress/status change.", "side_effects": "Writes progress memory and checkpoint context."},
    "session_outcome": {"when": "Call once per session after success/failure is known.", "side_effects": "Writes outcome signals that affect memory quality weighting."},
    "create_task": {"when": "Call at the beginning of a multi-step objective.", "side_effects": "Writes a new task record."},
    "update_task": {"when": "Call when task status/goal/metadata changes.", "side_effects": "Writes task fields."},
    "get_task": {"when": "Call when taking over an existing task.", "side_effects": "Read-only."},
    "list_tasks": {"when": "Call to discover active/planning/blocked tasks.", "side_effects": "Read-only."},
    "restore_checkpoint": {"when": "Call on takeover after interruption.", "side_effects": "Read-only restore payload; does not mutate task state."},
    "list_checkpoints": {"when": "Call for timeline/debugging of checkpoint versions.", "side_effects": "Read-only."},
    "get_runtime_health": {"when": "Call when backend appears degraded or before risky operations.", "side_effects": "Read-only."},
    "report_interruption": {"when": "Call before exit when interruption reason is known.", "side_effects": "Writes interruption taxonomy for next-session recovery."},
    "evaluate_continuity": {"when": "Call after recovery to score continuity quality.", "side_effects": "Read-only scoring output."},
}

for _tool in TOOL_SCHEMAS:
    _notes = _TDQ_NOTES.get(_tool.name)
    if not _notes:
        continue
    _suffix = (
        f" Usage: {_notes['when']} "
        f"Behavior: {_notes['side_effects']}"
    )
    if _suffix not in _tool.description:
        _tool.description = f"{_tool.description} {_suffix}"
