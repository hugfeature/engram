"""Operational maintenance: backup pruning + DuckDB upgrade detection.

This module is intentionally side-effect-light and best-effort: every call
swallows non-critical errors so a misconfigured filesystem can never block
the runtime from booting.

Two responsibilities:

1. ``prune_backups()`` — keep ``~/.engram/backups`` from growing unbounded.
   Backups are *archived* rather than deleted so an operator can still
   recover them if a "pruned" file turns out to have been needed.

2. ``detect_duckdb_upgrade()`` — if the DuckDB library version changed
   since the last boot, copy the active DB to a side-car file *before*
   anything touches it, and emit a ``runtime.duckdb_upgrade`` event so
   the operator has a clear timeline anchor.

Both are called during boot from ``daemon.cmd_run`` / ``setup_cli`` after
the DB has connected, on a worker thread so they never block startup.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time as _time
from typing import Iterable

import duckdb

log = logging.getLogger(__name__)

# Constants live here so callers don't need to reach into db.py internals.
ENGRAM_DIR = os.path.join(os.path.expanduser("~"), ".engram")
BACKUP_DIR = os.path.join(ENGRAM_DIR, "backups")
ARCHIVE_DIR = os.path.join(BACKUP_DIR, "archive")

ENV_BACKUP_RETAIN = "ENGRAM_BACKUP_RETAIN"
DEFAULT_BACKUP_RETAIN = 10

# File-name prefixes considered "backup artifacts" we are responsible for.
# We do NOT touch user-created files in BACKUP_DIR that don't match.
_BACKUP_PREFIXES = (
    "memories-pre-recover-",
    "memories-pre-duckdb-upgrade-",
)


def _ts_suffix() -> str:
    return _time.strftime("%Y%m%d-%H%M%S")


def _read_retain_count() -> int:
    raw = os.environ.get(ENV_BACKUP_RETAIN, "").strip()
    if not raw:
        return DEFAULT_BACKUP_RETAIN
    try:
        n = int(raw)
        return max(1, n)  # always keep at least 1
    except ValueError:
        log.warning(
            "Invalid %s=%r; falling back to default %d",
            ENV_BACKUP_RETAIN, raw, DEFAULT_BACKUP_RETAIN,
        )
        return DEFAULT_BACKUP_RETAIN


# ---------------------------------------------------------------------------
# Backup pruning
# ---------------------------------------------------------------------------

def _list_managed_backups(backup_dir: str) -> list[str]:
    """Return absolute paths of files we are responsible for, sorted oldest-first."""
    if not os.path.isdir(backup_dir):
        return []
    entries: list[tuple[float, str]] = []
    for name in os.listdir(backup_dir):
        full = os.path.join(backup_dir, name)
        if not os.path.isfile(full):
            continue
        if not any(name.startswith(p) for p in _BACKUP_PREFIXES):
            continue
        try:
            entries.append((os.path.getmtime(full), full))
        except OSError:
            continue
    entries.sort(key=lambda x: x[0])
    return [path for _, path in entries]


def prune_backups(
    backup_dir: str = BACKUP_DIR,
    archive_dir: str = ARCHIVE_DIR,
    retain: int | None = None,
) -> dict:
    """Archive old backup files so the live backup dir stays under ``retain``.

    Strategy:
      1. List managed backups in ``backup_dir`` sorted oldest-first.
      2. If count <= retain → no-op.
      3. Move surplus (oldest first) into ``archive_dir`` with mtime preserved.
         We never delete; an operator can still recover an archived file.

    Returns a small report dict; safe to call repeatedly.
    """
    if retain is None:
        retain = _read_retain_count()

    managed = _list_managed_backups(backup_dir)
    if len(managed) <= retain:
        return {
            "scanned": len(managed),
            "kept": len(managed),
            "archived": [],
            "dir": backup_dir,
        }

    surplus = managed[: len(managed) - retain]
    archived: list[str] = []
    if surplus:
        try:
            os.makedirs(archive_dir, exist_ok=True)
        except OSError as exc:
            log.warning("Cannot create archive dir %s: %s", archive_dir, exc)
            return {
                "scanned": len(managed),
                "kept": len(managed),
                "archived": [],
                "dir": backup_dir,
                "error": str(exc),
            }

    for src in surplus:
        dst = os.path.join(archive_dir, os.path.basename(src))
        # Disambiguate if the archive already has a same-named file.
        if os.path.exists(dst):
            dst = f"{dst}.{_ts_suffix()}"
        try:
            shutil.move(src, dst)
            archived.append(dst)
            log.info("Archived old backup: %s -> %s", src, dst)
        except OSError as exc:
            log.warning("Could not archive %s: %s", src, exc)

    report = {
        "scanned": len(managed),
        "kept": len(managed) - len(archived),
        "archived": archived,
        "dir": backup_dir,
    }

    # Best-effort: emit an event for operator visibility. Never raise.
    if archived:
        _try_emit_event("maintenance.backup_pruned", {
            "archived": archived,
            "kept": report["kept"],
            "dir": backup_dir,
        })

    return report


# ---------------------------------------------------------------------------
# DuckDB upgrade detection
# ---------------------------------------------------------------------------

def _major_minor(v: str) -> tuple[int, int]:
    parts = v.split(".")
    try:
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return 0, 0


def _is_breaking_upgrade(old: str, new: str) -> bool:
    """Heuristic: any change in major OR minor version triggers a backup.

    DuckDB has historically broken file format on minor version bumps
    (e.g. 0.9 → 0.10), not just major. So we treat both as worth a backup.
    """
    if not old or not new or old == new:
        return False
    return _major_minor(old) != _major_minor(new)


def detect_duckdb_upgrade(
    db_path: str,
    old_version: str | None,
    new_version: str | None = None,
    backup_dir: str = BACKUP_DIR,
) -> dict | None:
    """If DuckDB version changed since last boot, snapshot the DB first.

    Returns a report dict if a backup was made, None otherwise.
    Safe to call when DB doesn't exist yet (returns None).
    """
    if new_version is None:
        try:
            new_version = duckdb.__version__
        except Exception:
            return None

    if not _is_breaking_upgrade(old_version or "", new_version):
        return None

    if not os.path.exists(db_path):
        # Fresh install on a new DuckDB version — nothing to back up.
        return None

    try:
        os.makedirs(backup_dir, exist_ok=True)
    except OSError as exc:
        log.warning("Cannot create backup dir %s: %s", backup_dir, exc)
        return None

    backup_name = (
        f"memories-pre-duckdb-upgrade-{old_version}-to-{new_version}"
        f"-{_ts_suffix()}.duckdb"
    )
    backup_path = os.path.join(backup_dir, backup_name)
    try:
        shutil.copy2(db_path, backup_path)
        log.warning(
            "DuckDB version change detected (%s -> %s); pre-upgrade backup at %s",
            old_version, new_version, backup_path,
        )
    except OSError as exc:
        log.warning("Failed to back up DB before DuckDB upgrade: %s", exc)
        return None

    report = {
        "old_version": old_version,
        "new_version": new_version,
        "backup_path": backup_path,
        "db_path": db_path,
    }

    _try_emit_event("runtime.duckdb_upgrade", report)
    return report


# ---------------------------------------------------------------------------
# Async startup hook
# ---------------------------------------------------------------------------

def schedule_startup_maintenance(
    db_path: str,
    old_duckdb_version: str | None,
    new_duckdb_version: str | None = None,
) -> threading.Thread:
    """Run maintenance tasks on a daemon thread so boot is never blocked.

    Order matters: detect_duckdb_upgrade() may *create* a new backup, so we
    run it before prune_backups() to ensure the new file is counted in the
    retention budget on the next boot, not this one.
    """
    def _run() -> None:
        try:
            detect_duckdb_upgrade(
                db_path=db_path,
                old_version=old_duckdb_version,
                new_version=new_duckdb_version,
            )
        except Exception as exc:
            log.warning("detect_duckdb_upgrade failed: %s", exc)
        try:
            prune_backups()
        except Exception as exc:
            log.warning("prune_backups failed: %s", exc)

    t = threading.Thread(target=_run, name="engram-maintenance", daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# internal: best-effort event emission
# ---------------------------------------------------------------------------

def _try_emit_event(kind: str, payload: dict) -> None:
    """Append to the default event log without ever raising into callers."""
    try:
        from .event_log import get_event_log
        get_event_log().append(kind, payload)
    except Exception as exc:
        log.debug("event emit %s skipped: %s", kind, exc)
