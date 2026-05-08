"""Automatic pruning + consolidation scheduler."""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

from .db import MemoryDB
from .decay import compute_strength, PRUNE_THRESHOLD
from .graph import MemoryGraph
from .consolidator import run_consolidate

log = logging.getLogger("engram.pruner")


class _MaintenanceState:
    """Encapsulated maintenance state — avoids module-level mutable globals."""
    __slots__ = ('last_time',)

    def __init__(self):
        self.last_time: datetime | None = None


maintenance = _MaintenanceState()


def run_prune(db: MemoryDB, graph: MemoryGraph, user_id: str = "default"):
    """Prune weak memories. Uses SQL pre-filter to avoid loading all rows."""
    now = datetime.now(timezone.utc)
    # Only consider memories not accessed in the last 7 days as pruning candidates
    cutoff = now - timedelta(days=7)
    try:
        rows = db.conn.execute(
            """
            SELECT id, category, importance, last_accessed_at, recall_count
            FROM memories WHERE user_id = ? AND last_accessed_at < ?
            """,
            [user_id, cutoff],
        ).fetchall()
    except Exception as e:
        log.error("Prune query failed for user '%s': %s", user_id, e)
        return

    pruned = 0
    with graph.batch_mode():
        for row in rows:
            mid, category, importance, last_accessed, recall_count = row[:5]
            days = (now - last_accessed.replace(tzinfo=timezone.utc)).total_seconds() / 86400
            strength = compute_strength(category, importance, days, recall_count)
            graph.update_node_strength(mid, strength)

            if strength < PRUNE_THRESHOLD:
                if graph.chain_safe_to_prune(mid, PRUNE_THRESHOLD, user_id=user_id):
                    db.delete(mid)
                    graph.remove_node(mid)
                    pruned += 1
                    log.info("Pruned memory %d (strength=%.4f)", mid, strength)

    if pruned:
        log.info("Pruning complete for user '%s': %d memories removed", user_id, pruned)


def run_maintenance(db: MemoryDB, graph: MemoryGraph, user_id: str | None = None):
    """Run consolidation + pruning for all users (or a single user if specified)."""

    if user_id is not None:
        user_ids = [user_id]
    else:
        user_ids = db.get_all_user_ids()
        if not user_ids:
            user_ids = ["default"]

    for uid in user_ids:
        try:
            run_consolidate(db, graph, uid)
        except Exception as e:
            log.error("Consolidation failed for user '%s': %s", uid, e)
        try:
            run_prune(db, graph, uid)
        except Exception as e:
            log.error("Pruning failed for user '%s': %s", uid, e)

    # Rebuild FTS index after consolidation + pruning may have changed data
    try:
        db._rebuild_fts_index()
    except Exception as e:
        log.error("FTS index rebuild failed during maintenance: %s", e)

    maintenance.last_time = datetime.now(timezone.utc)
    log.info("Maintenance completed for %d user(s)", len(user_ids))


def start_scheduler(db: MemoryDB, graph: MemoryGraph):
    """Start a 12-hour maintenance cycle covering all users."""
    try:
        from apscheduler.schedulers.background import BackgroundScheduler

        scheduler = BackgroundScheduler()
        scheduler.add_job(
            run_maintenance,
            "interval",
            hours=12,
            args=[db, graph],
            id="memory_maintenance",
        )
        scheduler.start()
        return scheduler
    except ImportError:
        log.warning("APScheduler not installed, maintenance disabled")
        return None