"""Engram CLI — setup / doctor / recover.

Subcommands:
    engram-setup                 first-run model download + DB init (default).
    engram-setup doctor          read-only health report (no recovery).
    engram-setup recover         replay event log into a fresh DB.
                                 Default is dry-run; pass --promote to swap.
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

    print("\n[1/3] Downloading embedding model (all-mpnet-base-v2)...")
    print("      Using HF mirror: hf-mirror.com")
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

    args = parser.parse_args()
    cmd = args.cmd or "setup"
    if cmd == "setup":
        sys.exit(_cmd_setup())
    if cmd == "doctor":
        sys.exit(_cmd_doctor())
    if cmd == "recover":
        sys.exit(_cmd_recover(args))


if __name__ == "__main__":
    main()
