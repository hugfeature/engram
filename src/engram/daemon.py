"""Engram server daemon — start / stop / status / run."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

PID_FILE = os.path.expanduser("~/.engram/engram.pid")
LOG_FILE = os.path.expanduser("~/.engram/server.log")
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

    from engram.http_server import main as serve_main
    sys.argv = ["engram-server", "--host", host, "--port", str(port)]
    serve_main()


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
