"""Engram runtime — session lifecycle hooks (startup / resume / reflection).

This package provides deterministic memory orchestration for MCP clients
(Claude Code, Cursor, etc.). It is invoked by client-side hooks and is
responsible for converting raw recall results into structured "active context"
that gets injected into the agent's system prompt.

Entry points:
- engram-session-hook -> engram.runtime.startup:main
"""
