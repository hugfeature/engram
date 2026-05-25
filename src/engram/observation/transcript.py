"""Layer 0: Raw Transcript — immutable structured representation.

Loads Claude Code JSONL transcripts and indexes by turn_id + block_index.
No inference, no transformation. Pure structural access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class ContentBlock:
    """A single content block within a turn (text, tool_use, tool_result, etc.)."""

    block_index: int
    block_type: str  # "text", "tool_use", "tool_result", "thinking", etc.
    raw: dict  # Original JSON dict — never modified

    @property
    def text(self) -> str:
        """Extract displayable text from this block."""
        if self.block_type == "text":
            return self.raw.get("text", "")
        if self.block_type == "tool_use":
            return f"[tool_use: {self.raw.get('name', '?')}]"
        if self.block_type == "tool_result":
            content = self.raw.get("content", "")
            if isinstance(content, list):
                return " ".join(
                    item.get("text", "") for item in content if isinstance(item, dict)
                )
            return str(content)
        if self.block_type == "thinking":
            return self.raw.get("thinking", "")
        return json.dumps(self.raw, ensure_ascii=False)[:200]


@dataclass(frozen=True)
class Turn:
    """A single turn in the transcript (one assistant or user message)."""

    turn_id: str
    role: str  # "assistant", "user", "system"
    blocks: tuple[ContentBlock, ...]
    raw: dict  # Original JSON line — never modified
    line_number: int  # 0-based line index in the JSONL file

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    def get_block(self, block_index: int) -> ContentBlock | None:
        """Get a content block by index."""
        if 0 <= block_index < len(self.blocks):
            return self.blocks[block_index]
        return None

    @property
    def tool_uses(self) -> list[ContentBlock]:
        """All tool_use blocks in this turn."""
        return [b for b in self.blocks if b.block_type == "tool_use"]

    @property
    def tool_results(self) -> list[ContentBlock]:
        """All tool_result blocks in this turn."""
        return [b for b in self.blocks if b.block_type == "tool_result"]


class Transcript:
    """Indexed access to a Claude Code JSONL transcript.

    Provides O(1) lookup by turn_id and sequential iteration.
    The raw JSONL is never modified — this is a read-only index.
    """

    def __init__(self, turns: list[Turn], source_path: Path | None = None):
        self._turns: list[Turn] = turns
        self._index: dict[str, Turn] = {t.turn_id: t for t in turns}
        self.source_path = source_path

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "Transcript":
        """Load a transcript from a JSONL file.

        Supports two formats:
        1. Claude Code native: top-level {type, uuid, message: {role, content}, ...}
        2. Generic: {id, role, content, ...}
        """
        path = Path(path)
        turns: list[Turn] = []

        with path.open("r", encoding="utf-8") as file_handle:
            for line_number, line in enumerate(file_handle):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                turn = cls._parse_turn(data, line_number)
                if turn is not None:
                    turns.append(turn)

        return cls(turns, source_path=path)

    @classmethod
    def from_json_array(cls, path: str | Path) -> "Transcript":
        """Load from a JSON array file (alternative format)."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)

        if not isinstance(data, list):
            data = [data]

        turns = []
        for line_number, item in enumerate(data):
            turn = cls._parse_turn(item, line_number)
            if turn is not None:
                turns.append(turn)

        return cls(turns, source_path=path)

    @classmethod
    def _parse_turn(cls, data: dict, line_number: int) -> Turn | None:
        """Parse a single JSON object into a Turn.

        Handles Claude Code native format:
          {type: "user"|"assistant", uuid: "...", message: {role, content}, timestamp, ...}
        And generic format:
          {id: "...", role: "...", content: [...]}

        Non-message lines (permission-mode, file-history-snapshot, etc.) are skipped.
        """
        if not isinstance(data, dict):
            return None

        top_type = data.get("type", "")

        # Claude Code native format: type is "user"/"assistant"/"system"
        if top_type in ("user", "assistant", "system") and "uuid" in data:
            turn_id = data["uuid"]
            role = top_type
            message = data.get("message", {})

            if isinstance(message, dict):
                raw_content = message.get("content", [])
            else:
                raw_content = []

            blocks = cls._parse_content_blocks(raw_content)
            return Turn(
                turn_id=turn_id,
                role=role,
                blocks=tuple(blocks),
                raw=data,
                line_number=line_number,
            )

        # Skip non-message lines (attachment, file-history-snapshot, etc.)
        if top_type in ("attachment", "file-history-snapshot", "permission-mode",
                        "queue-operation", "custom-title", "agent-name", "last-prompt"):
            return None

        # Generic format fallback
        turn_id = data.get("id") or data.get("uuid") or f"line_{line_number}"
        role = data.get("role", "unknown")
        raw_content = data.get("content", [])
        blocks = cls._parse_content_blocks(raw_content)

        if not blocks:
            return None

        return Turn(
            turn_id=turn_id,
            role=role,
            blocks=tuple(blocks),
            raw=data,
            line_number=line_number,
        )

    @classmethod
    def _parse_content_blocks(cls, raw_content) -> list[ContentBlock]:
        """Parse content into ContentBlock list."""
        blocks: list[ContentBlock] = []

        if isinstance(raw_content, str):
            if raw_content.strip():
                blocks.append(ContentBlock(
                    block_index=0,
                    block_type="text",
                    raw={"text": raw_content},
                ))
        elif isinstance(raw_content, list):
            for block_idx, block_data in enumerate(raw_content):
                if isinstance(block_data, dict):
                    block_type = block_data.get("type", "unknown")
                    blocks.append(ContentBlock(
                        block_index=block_idx,
                        block_type=block_type,
                        raw=block_data,
                    ))
                elif isinstance(block_data, str):
                    blocks.append(ContentBlock(
                        block_index=block_idx,
                        block_type="text",
                        raw={"text": block_data},
                    ))

        return blocks

    def __len__(self) -> int:
        return len(self._turns)

    def __iter__(self) -> Iterator[Turn]:
        return iter(self._turns)

    def __getitem__(self, index: int) -> Turn:
        return self._turns[index]

    def get_turn(self, turn_id: str) -> Turn | None:
        """O(1) lookup by turn_id."""
        return self._index.get(turn_id)

    def get_span(self, turn_id: str, block_index: int) -> ContentBlock | None:
        """Get a specific content block by anchor coordinates."""
        turn = self._index.get(turn_id)
        if turn is None:
            return None
        return turn.get_block(block_index)

    def get_raw_text(self, turn_id: str, block_index: int) -> str | None:
        """Get raw text for a specific anchor — used for reversibility check."""
        block = self.get_span(turn_id, block_index)
        if block is None:
            return None
        return block.text

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def assistant_turns(self) -> list[Turn]:
        return [t for t in self._turns if t.role == "assistant"]

    @property
    def user_turns(self) -> list[Turn]:
        return [t for t in self._turns if t.role == "user"]
