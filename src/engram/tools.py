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
]
