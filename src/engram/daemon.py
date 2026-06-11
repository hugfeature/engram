"""Engram server daemon — start / stop / status / run."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

_ENGRAM_DIR = os.environ.get("ENGRAM_HOME") or os.path.expanduser("~/.engram")
PID_FILE = os.path.join(_ENGRAM_DIR, "engram.pid")
LOG_FILE = os.path.join(_ENGRAM_DIR, "server.log")
DEFAULT_PORT = 8900


def _read_pid() -> int | None:
    try:
        with open(PID_FILE) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def _write_pid(pid: int) -> None:
    os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)
    with open(PID_FILE, "w") as f:
        f.write(str(pid))


def _is_running(pid: int | None = None) -> bool:
    pid = pid or _read_pid()
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        out = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "command="], text=True, timeout=5
        )
        # Match engram-specific invocations, not arbitrary processes containing "engram"
        return any(
            marker in out
            for marker in ("engram.http_server", "engram.server", "-m engram")
        )
    except Exception:
        return True


def cmd_start(host: str, port: int) -> None:
    pid = _read_pid()
    if _is_running(pid):
        print(f"Already running (PID {pid})")
        return

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as log_fd:
        proc = subprocess.Popen(
            [sys.executable, "-m", "engram.http_server", "--host", host, "--port", str(port)],
            start_new_session=True,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
        )

    time.sleep(1)
    if proc.poll() is not None:
        print(f"Failed to start (exit code {proc.returncode}), check {LOG_FILE}")
        return

    _write_pid(proc.pid)
    print(f"Started engram server (PID={proc.pid}, {host}:{port})")


def cmd_stop() -> None:
    pid = _read_pid()
    if not pid or not _is_running(pid):
        print("Not running")
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
        return

    os.kill(pid, signal.SIGTERM)
    for _ in range(10):
        time.sleep(0.5)
        if not _is_running(pid):
            break

    if _is_running(pid):
        os.kill(pid, signal.SIGKILL)

    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
    print(f"Stopped (PID {pid})")


def cmd_status() -> None:
    pid = _read_pid()
    if pid and _is_running(pid):
        print(f"Running (PID={pid})")
    else:
        print("Not running")
        if pid and os.path.exists(PID_FILE):
            os.remove(PID_FILE)


def cmd_run(host: str, port: int) -> None:
    def _handle_sigterm(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    # Pre-flight: warn loudly if there are recovery artifacts on disk so
    # operators see them before serving any traffic.
    _warn_on_residue()

    # Start auto-reload watcher — restarts this process when package is updated.
    _start_reload_watcher()

    from engram.http_server import main as serve_main
    sys.argv = ["engram-server", "--host", host, "--port", str(port)]
    serve_main()


# ---- Auto-reload on package update ----

_RELOAD_CHECK_INTERVAL = 5  # seconds between version checks


def _get_package_fingerprint() -> str:
    """Get a fingerprint that changes when the package is reinstalled.

    For editable installs: uses mtime of the source directory.
    For regular installs: uses the installed package version + dist-info mtime.
    Falls back to version string from importlib.metadata.
    """
    import importlib.metadata

    try:
        dist = importlib.metadata.distribution("mcp-engram")
        version = dist.metadata["Version"]

        # For editable installs, check source file mtimes
        dist_files = dist.files
        if dist_files:
            # Use the direct_url.json or top_level.txt mtime as proxy
            import engram
            source_init = getattr(engram, "__file__", None)
            if source_init and os.path.exists(source_init):
                source_dir = os.path.dirname(source_init)
                # Hash of mtimes of all .py files in package
                mtimes = []
                for root, _dirs, files in os.walk(source_dir):
                    for fname in sorted(files):
                        if fname.endswith(".py"):
                            fpath = os.path.join(root, fname)
                            try:
                                mtimes.append(os.path.getmtime(fpath))
                            except OSError:
                                pass
                return f"{version}:{max(mtimes) if mtimes else 0}"

        return version
    except Exception:
        return "unknown"


def _start_reload_watcher() -> None:
    """Start a daemon thread that watches for package updates and restarts."""
    import threading

    initial_fingerprint = _get_package_fingerprint()

    def _watch():
        while True:
            time.sleep(_RELOAD_CHECK_INTERVAL)
            try:
                current = _get_package_fingerprint()
                if current != initial_fingerprint:
                    sys.stderr.write(
                        f"\n[engram] Package updated ({initial_fingerprint} → {current}). "
                        f"Auto-restarting server...\n"
                    )
                    sys.stderr.flush()
                    # exec replaces the current process with a fresh one
                    os.execv(sys.executable, [sys.executable] + sys.argv)
            except SystemExit:
                return
            except Exception as exc:
                # Never crash the watcher — just log and continue
                sys.stderr.write(f"[engram] reload watcher error: {exc}\n")

    thread = threading.Thread(target=_watch, name="engram-reload-watcher", daemon=True)
    thread.start()


def _warn_on_residue() -> None:
    """Surface .corrupt.* / .wal-recovery.* near the DB file."""
    try:
        from engram.db import _scan_residue, DB_PATH
    except Exception:
        return
    files = _scan_residue(DB_PATH)
    if not files:
        return
    sys.stderr.write(
        "\n[engram] WARNING: corruption-recovery artifacts present:\n"
    )
    for f in files:
        sys.stderr.write(f"  - {f}\n")
    sys.stderr.write(
        "  Inspect with `engram-setup doctor`; recover with `engram-setup recover`.\n\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Engram server daemon")
    parser.add_argument("action", choices=["start", "stop", "status", "run"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    if args.action == "start":
        cmd_start(args.host, args.port)
    elif args.action == "stop":
        cmd_stop()
    elif args.action == "status":
        cmd_status()
    elif args.action == "run":
        cmd_run(args.host, args.port)


if __name__ == "__main__":
    main()
