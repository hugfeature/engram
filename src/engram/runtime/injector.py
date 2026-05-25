"""Active Context injector — talks to the local Engram daemon and renders a
structured Markdown block suitable for SessionStart hook injection.

Design principles:
- Deterministic over clever: fixed sections, fixed weights, no LLM in the loop.
- Executable state, not historical narrative: "what's in flight" rather than
  "what happened on 2026-05-01".
- Graceful degradation: any failure is logged and yields an empty/notice block,
  never blocks the host session.
- Bounded budget: total injection is capped (default 2000 chars) so the agent's
  context isn't polluted.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("engram.runtime.injector")

DEFAULT_DAEMON_URL = os.environ.get("ENGRAM_DAEMON_URL", "http://127.0.0.1:8900")
DEFAULT_TIMEOUT = float(os.environ.get("ENGRAM_HOOK_TIMEOUT", "4.0"))
DEFAULT_USER_ID = os.environ.get("ENGRAM_USER_ID", "default")
RUNTIME_DIR = Path(os.environ.get("ENGRAM_HOME", str(Path.home() / ".engram"))) / "runtime"
MAX_INJECTION_CHARS = int(os.environ.get("ENGRAM_INJECT_BUDGET", "2000"))


# ---------- Environment probe ----------

@dataclass
class ProjectContext:
    cwd: str
    repo: str | None = None
    branch: str | None = None

    def as_lines(self) -> list[str]:
        lines = [f"- cwd: `{self.cwd}`"]
        if self.repo:
            lines.append(f"- repo: `{self.repo}`")
        if self.branch:
            lines.append(f"- branch: `{self.branch}`")
        return lines

    def query_keyword(self) -> str:
        """Best-effort short query string for semantic recall."""
        if self.repo:
            # repo may be "owner/name" or just "name"
            return self.repo.split("/")[-1]
        return Path(self.cwd).name


def _git(args: list[str], cwd: str) -> str | None:
    import shutil
    
    # Security: Use absolute path to prevent PATH hijacking
    git_path = shutil.which("git")
    if not git_path:
        return None
    
    try:
        out = subprocess.run(
            [git_path, *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=1.5,
            shell=False,  # Explicitly disable shell
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def probe_project(cwd: str | None = None) -> ProjectContext:
    cwd = cwd or os.getcwd()
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    remote = _git(["config", "--get", "remote.origin.url"], cwd)
    repo: str | None = None
    if remote:
        # Normalize "git@github.com:owner/name.git" or "https://.../owner/name.git"
        slug = remote
        for prefix in ("git@github.com:", "https://github.com/"):
            if slug.startswith(prefix):
                slug = slug[len(prefix):]
                break
        if slug.endswith(".git"):
            slug = slug[:-4]
        repo = slug or None
    return ProjectContext(cwd=cwd, repo=repo, branch=branch)


# ---------- Daemon client ----------

@dataclass
class DaemonResult:
    available: bool
    payloads: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _post_json(url: str, body: dict, timeout: float) -> dict | None:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        log.warning("POST %s failed: %s", url, exc)
        return None


def fetch_active_context(
    project: ProjectContext,
    user_id: str = DEFAULT_USER_ID,
    daemon_url: str = DEFAULT_DAEMON_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> DaemonResult:
    """Issue 4 sequential recalls. Each individual failure is non-fatal.

    Why sequential, not parallel: the embedding model (sentence-transformers
    mpnet on ARM CPU) is not safe to invoke concurrently — multiple threads
    calling encode() at once can SIGBUS the daemon. See embedding.py header
    notes. With daemon-side warmup, single-call latency is ~80ms hot, so
    4 calls × 80ms = ~320ms total, well under the 8s hook budget.
    """
    result = DaemonResult(available=True)

    # 1. Latest handoff (memory_type=handoff, query is project name)
    handoff_q = project.query_keyword() or "session"
    handoff = _post_json(
        f"{daemon_url}/v1/recall",
        {"query": handoff_q, "user_id": user_id, "top_k": 1, "memory_type": "handoff"},
        timeout,
    )
    if handoff is None:
        result.errors.append("handoff recall failed")
    else:
        result.payloads["handoff"] = handoff

    # 2. In-progress tasks (list_tasks status=in_progress)
    tasks = _post_json(
        f"{daemon_url}/v1/tasks/list",
        {"user_id": user_id, "status": "in_progress"},
        timeout,
    )
    if tasks is None:
        result.errors.append("list_tasks failed")
    else:
        result.payloads["tasks"] = tasks

    # 3. Recent failures: try project-keyword first, fall back to broad query
    # so we still surface failures even when the project name is rare in the corpus.
    failures = _post_json(
        f"{daemon_url}/v1/recall",
        {"query": handoff_q, "user_id": user_id, "top_k": 5, "memory_type": "failure"},
        timeout,
    )
    if not _memories_of(failures):
        # Fall back: broad query — recall by category alone.
        failures = _post_json(
            f"{daemon_url}/v1/recall",
            {"query": "failure error bug", "user_id": user_id, "top_k": 5, "memory_type": "failure"},
            timeout,
        )
    if failures is None:
        result.errors.append("failure recall failed")
    else:
        result.payloads["failures"] = failures

    # 4. Project-relevant general recall
    general = _post_json(
        f"{daemon_url}/v1/recall",
        {"query": handoff_q, "user_id": user_id, "top_k": 5, "memory_type": "all"},
        timeout,
    )
    if general is None:
        result.errors.append("general recall failed")
    else:
        result.payloads["general"] = general

    # If every single call failed, the daemon is effectively unavailable.
    if not result.payloads:
        result.available = False

    return result


# ---------- Renderers ----------

def _memories_of(payload: Any) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    mems = payload.get("memories")
    if not isinstance(mems, list):
        return []
    return [m for m in mems if isinstance(m, dict)]


def _truncate(text: str, limit: int) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _render_handoff(payload: Any) -> list[str]:
    mems = _memories_of(payload)
    if not mems:
        return []
    top = mems[0]
    content = _truncate(str(top.get("content", "")), 320)
    meta = top.get("metadata") if isinstance(top.get("metadata"), dict) else {}
    next_steps = meta.get("next_steps") if isinstance(meta.get("next_steps"), list) else []
    in_progress = meta.get("in_progress") if isinstance(meta.get("in_progress"), list) else []

    lines: list[str] = [content]
    if in_progress:
        lines.append("- in-progress: " + "; ".join(_truncate(str(x), 80) for x in in_progress[:3]))
    if next_steps:
        lines.append("- next: " + "; ".join(_truncate(str(x), 80) for x in next_steps[:3]))
    return lines


def _render_tasks(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        return []
    out: list[str] = []
    for t in tasks[:5]:
        if not isinstance(t, dict):
            continue
        tid = t.get("id") or t.get("task_id")
        name = t.get("name") or "(unnamed)"
        goal = t.get("goal") or ""
        head = f"- [task#{tid}] **{name}**" if tid is not None else f"- **{name}**"
        if goal:
            head += f" — {_truncate(str(goal), 120)}"
        out.append(head)
    return out


def _render_failures(payload: Any) -> list[str]:
    mems = _memories_of(payload)
    if not mems:
        return []
    out: list[str] = []
    for m in mems[:3]:
        content = _truncate(str(m.get("content", "")), 160)
        meta = m.get("metadata") if isinstance(m.get("metadata"), dict) else {}
        component = meta.get("component")
        prefix = f"[{component}] " if component else ""
        out.append(f"- {prefix}{content}")
    return out


def _render_general(payload: Any) -> list[str]:
    mems = _memories_of(payload)
    if not mems:
        return []
    out: list[str] = []
    for m in mems[:5]:
        importance = m.get("importance")
        try:
            if importance is not None and float(importance) < 0.5:
                continue
        except (TypeError, ValueError):
            pass
        # Skip handoff/failure entries — they have their own sections.
        meta = m.get("metadata") if isinstance(m.get("metadata"), dict) else {}
        memtype = meta.get("type") or m.get("category")
        if memtype in ("handoff", "failure"):
            continue
        content = _truncate(str(m.get("content", "")), 180)
        out.append(f"- {content}")
        if len(out) >= 4:
            break
    return out


def render_active_context(
    project: ProjectContext,
    daemon: DaemonResult,
    budget: int = MAX_INJECTION_CHARS,
) -> str:
    """Compose the final Markdown block. Always returns a non-empty string."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections: list[str] = ["# Active Context", f"_injected by Engram runtime · {now}_", ""]

    sections.append("## Current Project")
    sections.extend(project.as_lines())
    sections.append("")

    if not daemon.available:
        sections.append("> ⚠️ Engram daemon unavailable — no runtime memory injected this session.")
        sections.append("> Start it via `engram-http` (HTTP) or check `~/.engram/server.log`.")
        return "\n".join(sections)

    handoff_lines = _render_handoff(daemon.payloads.get("handoff"))
    if handoff_lines:
        sections.append("## Latest Handoff")
        sections.extend(handoff_lines)
        sections.append("")

    task_lines = _render_tasks(daemon.payloads.get("tasks"))
    if task_lines:
        sections.append("## In Progress Tasks")
        sections.extend(task_lines)
        sections.append("")

    failure_lines = _render_failures(daemon.payloads.get("failures"))
    if failure_lines:
        sections.append("## Recent Failures")
        sections.extend(failure_lines)
        sections.append("")

    general_lines = _render_general(daemon.payloads.get("general"))
    if general_lines:
        sections.append("## Relevant Memories")
        sections.extend(general_lines)
        sections.append("")

    sections.append("---")
    sections.append(
        "> Injected automatically. The above is **already loaded** — do not re-`recall_memory` "
        "for the same scope. Use `recall_memory(query)` only for narrower follow-ups."
    )

    text = "\n".join(sections).rstrip() + "\n"
    if len(text) > budget:
        text = text[: budget - 1].rstrip() + "…\n"
    return text


# ---------- Debug artifacts ----------

def write_debug_artifacts(project: ProjectContext, daemon: DaemonResult, rendered: str) -> None:
    """Persist the last recall payload and rendered context for offline debugging."""
    try:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        (RUNTIME_DIR / "startup_context.md").write_text(rendered, encoding="utf-8")
        snapshot = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "project": {"cwd": project.cwd, "repo": project.repo, "branch": project.branch},
            "daemon_available": daemon.available,
            "errors": daemon.errors,
            "payloads": daemon.payloads,
        }
        (RUNTIME_DIR / "last_recall.json").write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        log.warning("failed to write debug artifacts: %s", exc)
