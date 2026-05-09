"""Engram — Durable Agent Runtime.

Cross-session task continuity, agent handoff, and constrained continuation
for MCP-aware coding agents (Claude Code, Cursor, ...).

Two laws:
    1. Event log is the only durability primitive.
    2. If it cannot be replayed, it is not critical state.
"""

__version__ = "0.13.1"
