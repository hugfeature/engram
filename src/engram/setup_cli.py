"""Engram CLI — setup / doctor / recover / prompt.

Subcommands:
    engram-setup                 first-run model download + DB init (default).
    engram-setup doctor          read-only health report (no recovery).
    engram-setup recover         replay event log into a fresh DB.
                                 Default is dry-run; pass --promote to swap.

Standalone commands (wired via pyproject.toml entry_points):
    engram-prompt                print a ready-to-paste CLAUDE.md snippet
                                 based on the current interrupt checkpoint.
"""

import argparse
import json
import os
import sys


def _cmd_setup() -> int:
    print("Engram Setup")
    print("=" * 40)

    data_dir = os.path.join(os.path.expanduser("~"), ".engram")
    os.makedirs(data_dir, exist_ok=True)

    model_name = os.environ.get("ENGRAM_MODEL", "all-mpnet-base-v2")
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    print(f"\n[1/3] Downloading embedding model ({model_name})...")
    print(f"      Using HF endpoint: {hf_endpoint}")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    try:
        from engram.embedding import embed
        vec = embed("setup test")
        print(f"      Model loaded OK ({len(vec)} dimensions)")
    except Exception as e:
        print(f"      ERROR: {e}", file=sys.stderr)
        print("      Try: HF_ENDPOINT=https://huggingface.co engram-setup")
        return 1

    print("\n[2/3] Initializing DuckDB...")
    try:
        from engram.db import MemoryDB
        db = MemoryDB()
        count = db.count()
        print(f"      Database ready at {data_dir}/memories.duckdb ({count} memories)")
        if db.readonly:
            print("      WARNING: DB is in readonly degraded mode — run `engram-setup recover`")
        if db.embedding_stale:
            print("      WARNING: embedding column dim drifted — vector search degraded")
        db.close()
    except Exception as e:
        print(f"      ERROR: {e}", file=sys.stderr)
        print(
            "      If the file is corrupt: try `engram-setup recover` to "
            "rebuild from the event log.",
            file=sys.stderr,
        )
        return 1

    print("\n[3/3] Initializing graph...")
    try:
        from engram.graph import MemoryGraph
        MemoryGraph()
        print(f"      Graph ready at {data_dir}/graph.json")
    except Exception as e:
        print(f"      ERROR: {e}", file=sys.stderr)
        return 1

    import shutil
    exe_path = shutil.which("engram") or "engram"

    print("\n" + "=" * 40)
    print("Setup complete!\n")
    print("Add to your MCP client config:\n")
    config = {
        "mcpServers": {
            "engram": {
                "command": exe_path,
                "env": {"HF_ENDPOINT": "https://hf-mirror.com"},
            }
        }
    }
    print(json.dumps(config, indent=2))
    print(f"\nData directory: {data_dir}")
    print(f"Executable:     {exe_path}")
    return 0


def _cmd_doctor() -> int:
    from engram.recover import doctor
    info = doctor()
    print(json.dumps(info, indent=2, ensure_ascii=False, default=str))
    if info.get("residue_files"):
        print(
            "\nNOTE: residue files present — these indicate prior "
            "corruption. Inspect, then delete manually if no longer needed.",
            file=sys.stderr,
        )
    backups = info.get("backups") or {}
    if backups.get("live_count", 0) > backups.get("retain", 0):
        print(
            f"\nNOTE: {backups['live_count']} live backups exceed retention "
            f"limit ({backups['retain']}). Surplus will be archived to "
            f"{backups.get('dir')}/archive on next boot.",
            file=sys.stderr,
        )
    if info.get("readonly"):
        print(
            "\nDB is in readonly degraded mode. Run `engram-setup recover` to rebuild.",
            file=sys.stderr,
        )
        return 2
    return 0


def _cmd_recover(args: argparse.Namespace) -> int:
    from engram.recover import recover, RecoverReport

    print("Engram Recover")
    print("=" * 40)
    print(f"Event dir:  {args.event_dir or '<default>'}")
    print(f"Since:      {args.since or '<beginning>'}")
    print(f"Output:     {args.output or '<auto>'}")
    print(f"Promote:    {args.promote}")
    print()

    try:
        report: RecoverReport = recover(
            event_dir=args.event_dir or _default_event_dir(),
            output_dir=args.output,
            since_date=args.since,
            promote=args.promote,
        )
    except Exception as exc:
        print(f"Recover failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False, default=str))

    if not args.promote:
        print(
            "\nDry-run complete. Recovered DB at:\n  "
            f"{report.output_db}\n"
            "Inspect it, then re-run with --promote to replace the active DB.",
            file=sys.stderr,
        )
    else:
        if report.backup_path:
            print(
                "\nPromoted. Original DB backed up at:\n  "
                f"{report.backup_path}",
                file=sys.stderr,
            )
        else:
            print(
                "\nPromoted. (No original DB to back up — "
                "recovered from event log only.)",
                file=sys.stderr,
            )
    if report.errors:
        return 2
    return 0


# ---------------------------------------------------------------------------
# engram-prompt — generate a ready-to-paste CLAUDE.md snippet
# ---------------------------------------------------------------------------

def _build_prompt_snippet(user_id: str = "default") -> tuple[str, dict]:
    """Query current runtime state and build a CLAUDE.md snippet.

    Returns:
        (snippet_text, raw_state_dict)
        raw_state_dict is empty if no active state found.
    """
    from engram.db import MemoryDB
    from engram.checkpoint import get_checkpoint, build_continuation

    db = MemoryDB(log_writes=False)  # read-only open, no event log
    try:
        # 1. Latest interrupt checkpoint (from SIGTERM recovery)
        interrupt_ckpt = db.get_latest_interrupt_checkpoint(user_id)

        # 2. Most recently updated in-progress task (fallback if no interrupt ckpt)
        active_tasks = db.list_tasks(user_id=user_id, status="in_progress")
        if not active_tasks:
            active_tasks = db.list_tasks(user_id=user_id, status="planning")

        if not interrupt_ckpt and not active_tasks:
            return "", {}

        # Prefer interrupt checkpoint task; fall back to most recent active task
        if interrupt_ckpt:
            task_id = interrupt_ckpt["task_id"]
            ckpt = interrupt_ckpt
        else:
            task_id = active_tasks[0].id
            ckpt = get_checkpoint(db, task_id, user_id=user_id)

        task = db.get_task(task_id)
        goal = (task.goal if task else "") or (ckpt["state"].get("goal") if ckpt else "")

        # Extract state fields
        state = ckpt["state"] if ckpt else {}
        ws = state.get("working_set") or {}
        interrupt_meta = ws.get("_interrupt", {})

        continuation = build_continuation(ckpt) if ckpt else {}
        completed    = continuation.get("completed") or []
        in_progress  = continuation.get("in_progress") or []
        blocked      = continuation.get("blocked") or []
        must_not_redo = continuation.get("must_not_redo") or []
        modified_files = ws.get("files") or []

        confidence = ckpt.get("continuation_confidence") if ckpt else None
        ckpt_version = ckpt["version"] if ckpt else None

        raw = {
            "task_id": task_id,
            "goal": goal,
            "checkpoint_version": ckpt_version,
            "continuation_confidence": confidence,
            "completed": completed,
            "in_progress": in_progress,
            "blocked": blocked,
            "must_not_redo": must_not_redo,
            "modified_files": modified_files,
            "last_tool_called": interrupt_meta.get("last_tool_called", ""),
            "last_failure": interrupt_meta.get("last_failure", ""),
            "interrupt_reason": interrupt_meta.get("interrupt_reason", ""),
        }

        lines = [
            "## Engram Runtime State",
            "",
            "<!-- Auto-generated by `engram-prompt`. Paste into CLAUDE.md. -->",
            "<!-- Re-run `engram-prompt` at session start to get the latest state. -->",
            "",
        ]

        lines += [
            f"**Active task:** {task_id} — {goal}",
        ]
        if ckpt_version is not None:
            conf_str = f" (confidence: {confidence:.2f})" if confidence else ""
            lines.append(f"**Checkpoint:** v{ckpt_version}{conf_str}")

        if completed:
            lines += ["", "**Already completed (do not redo):**"]
            lines += [f"- {c}" for c in completed]

        if in_progress:
            lines += ["", "**In progress:**"]
            lines += [f"- {c}" for c in in_progress]

        if blocked:
            lines += ["", "**Blocked / needs attention:**"]
            lines += [f"- {c}" for c in blocked]

        if must_not_redo:
            lines += ["", "**Must NOT redo:**"]
            for item in must_not_redo:
                if isinstance(item, dict):
                    lines.append(f"- {item.get('action', item)} [{item.get('reason', '')}]")
                else:
                    lines.append(f"- {item}")

        if modified_files:
            lines += ["", "**Files modified in last session:**"]
            lines += [f"- `{f}`" for f in modified_files]

        if interrupt_meta.get("last_failure"):
            lines += ["", f"**Last failure:** {interrupt_meta['last_failure']}"]

        lines += [
            "",
            "## Engram Session Rules",
            "",
            "- **Session start:** `recall_memory(query)` — interrupt state auto-pinned",
            f"- **Resume task:** `restore_checkpoint(task_id={task_id})` — get full continuation",
            "- **Progress update:** `track_progress(feature, status, task_id=<id>)`",
            "- **On error:** `track_failure(error, component, root_cause, task_id=<id>)`",
            "- **Context filling up:** `report_interruption(reason=\"overflow\")` then `session_handoff(...)`",
            "- **Session end:** `session_handoff(summary, completed, in_progress, blocked, task_id=<id>)`",
        ]

        return "\n".join(lines), raw
    finally:
        db.close()


def _cmd_prompt(args: argparse.Namespace) -> int:
    """Print a ready-to-paste CLAUDE.md snippet based on current state."""
    user_id = getattr(args, "user_id", None) or "default"
    as_json = getattr(args, "json", False)

    try:
        snippet, raw = _build_prompt_snippet(user_id)
    except Exception as exc:
        print(f"engram-prompt error: {exc}", file=sys.stderr)
        return 1

    if not snippet:
        print(
            "No active tasks or interrupt checkpoint found.\n"
            "Start a task first: create_task(name, goal)",
            file=sys.stderr,
        )
        return 0

    if as_json:
        print(json.dumps(raw, indent=2, ensure_ascii=False, default=str))
    else:
        print(snippet)

    return 0


def prompt_main() -> None:
    """Entry point for `engram-prompt` standalone command."""
    parser = argparse.ArgumentParser(
        prog="engram-prompt",
        description=(
            "Print a ready-to-paste CLAUDE.md snippet based on the current "
            "Engram runtime state (active task + interrupt checkpoint)."
        ),
    )
    parser.add_argument(
        "--user-id", default="default",
        help="User ID to query (default: 'default')",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON instead of Markdown",
    )
    args = parser.parse_args()
    sys.exit(_cmd_prompt(args))


def _default_event_dir() -> str:
    from engram.event_log import DEFAULT_EVENT_DIR
    return DEFAULT_EVENT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(prog="engram-setup", description="Engram CLI")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("setup", help="First-run setup (download model, init DB).")
    sub.add_parser("doctor", help="Read-only health report.")

    rec = sub.add_parser(
        "recover",
        help="Replay event log into a fresh DB. Dry-run by default.",
    )
    rec.add_argument("--event-dir", default=None,
                     help="Event log directory (default: ~/.engram/events).")
    rec.add_argument("--since", default=None,
                     help="Lower bound date (YYYYMMDD), inclusive.")
    rec.add_argument("--output", default=None,
                     help="Output directory for the rebuilt DB.")
    rec.add_argument("--promote", action="store_true",
                     help="After rebuild, swap the new DB in (original goes to backups/).")

    prompt_p = sub.add_parser(
        "prompt",
        help="Print a ready-to-paste CLAUDE.md snippet (same as `engram-prompt`).",
    )
    prompt_p.add_argument("--user-id", default="default")
    prompt_p.add_argument("--json", action="store_true",
                          help="Output raw JSON instead of Markdown")

    args = parser.parse_args()
    cmd = args.cmd or "setup"
    if cmd == "setup":
        sys.exit(_cmd_setup())
    if cmd == "doctor":
        sys.exit(_cmd_doctor())
    if cmd == "recover":
        sys.exit(_cmd_recover(args))
    if cmd == "prompt":
        sys.exit(_cmd_prompt(args))


if __name__ == "__main__":
    main()