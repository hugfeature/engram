"""SessionStart hook entry point.

Reads the Claude Code SessionStart hook payload from stdin (optional), assembles
an Active Context block via `injector`, and emits the official hook response on
stdout:

    {"hookSpecificOutput": {"hookEventName": "SessionStart",
                             "additionalContext": "..."}}

Hard guarantees:
- Exits 0 even on internal failure (never blocks the host session).
- All diagnostic logging goes to stderr / `~/.engram/runtime/hook.log`.
- Total wall time is bounded by `ENGRAM_HOOK_TIMEOUT` (default 2s) per call,
  ~3 calls in practice — soft cap ~6s. The host session will not hang.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .injector import (
    DEFAULT_DAEMON_URL,
    DEFAULT_TIMEOUT,
    DEFAULT_USER_ID,
    RUNTIME_DIR,
    fetch_active_context,
    probe_project,
    render_active_context,
    write_debug_artifacts,
)

log = logging.getLogger("engram.runtime.startup")


def _setup_logging() -> None:
    try:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            RUNTIME_DIR / "hook.log",
            maxBytes=256 * 1024,
            backupCount=2,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s"))
        root = logging.getLogger("engram.runtime")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    except OSError:
        # Fall back to stderr only — never abort the hook.
        logging.basicConfig(level=logging.INFO, stream=sys.stderr)


def _read_hook_input() -> dict:
    """Claude Code passes a JSON blob on stdin. Empty/invalid input is fine."""
    if sys.stdin is None or sys.stdin.isatty():
        return {}
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return {}
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("could not parse hook stdin: %s", exc)
        return {}


def _emit(additional_context: str) -> None:
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": additional_context,
        }
    }
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    sys.stdout.write("\n")
    sys.stdout.flush()


def main() -> int:
    _setup_logging()
    try:
        hook_input = _read_hook_input()
        cwd = hook_input.get("cwd") or os.getcwd()
        project = probe_project(cwd)
        log.info(
            "SessionStart hook fired: cwd=%s repo=%s branch=%s source=%s",
            project.cwd, project.repo, project.branch,
            hook_input.get("source", "unknown"),
        )
        daemon = fetch_active_context(
            project,
            user_id=DEFAULT_USER_ID,
            daemon_url=DEFAULT_DAEMON_URL,
            timeout=DEFAULT_TIMEOUT,
        )
        rendered = render_active_context(project, daemon)
        write_debug_artifacts(project, daemon, rendered)
        _emit(rendered)
        return 0
    except Exception as exc:  # pragma: no cover - defensive last resort
        # Absolutely must not crash the hook; emit a minimal notice instead.
        log.exception("SessionStart hook crashed: %s", exc)
        try:
            _emit(
                "# Active Context\n"
                f"> ⚠️ Engram runtime hook crashed: {exc!r}. "
                "See `~/.engram/runtime/hook.log`.\n"
            )
        except Exception:
            pass
        return 0


if __name__ == "__main__":
    sys.exit(main())
