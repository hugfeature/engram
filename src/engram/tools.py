"""Shared MCP tool schemas — used by both stdio and HTTP servers."""

from __future__ import annotations

from mcp.types import Tool

TOOL_SCHEMAS: list[Tool] = [
    Tool(
        name="recall_memory",
        description="Search memories by semantic similarity. Call at the start of every task.",
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
            },
            "required": ["feature", "status"],
        },
    ),
]
