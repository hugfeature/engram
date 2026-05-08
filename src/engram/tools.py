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
        description="Store a new memory. Automatically deduplicates against existing memories.",
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
        description="Scan all memories and auto-merge similar ones. Reduces bloat and keeps knowledge clean.",
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
        description="Record structured end-of-session state for cross-session continuity. Creates a searchable handoff snapshot.",
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
]
