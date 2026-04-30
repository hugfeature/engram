"""Automatic pruning + consolidation scheduler."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from .db import MemoryDB
from .decay import compute_strength, PRUNE_THRESHOLD
from .graph import MemoryGraph
from .consolidator import run_consolidate

log = logging.getLogger("engram.pruner")

last_maintenance_time: datetime | None = None


def run_prune(db: MemoryDB, graph: MemoryGraph, user_id: str = "default"):
    memories = db.get_all(user_id)
    now = datetime.now(timezone.utc)
    pruned = 0

    for m in memories:
        days = (now - m.last_accessed_at.replace(tzinfo=timezone.utc)).total_seconds() / 86400
        strength = compute_strength(m.category, m.importance, days, m.recall_count)
        graph.update_node_strength(m.id, strength)

        if strength < PRUNE_THRESHOLD:
            if graph.chain_safe_to_prune(m.id, PRUNE_THRESHOLD):
                db.delete(m.id)
                graph.remove_node(m.id)
                pruned += 1
                log.info("Pruned memory %d (strength=%.4f)", m.id, strength)

    if pruned:
        log.info("Pruning complete: %d memories removed", pruned)


def run_maintenance(db: MemoryDB, graph: MemoryGraph, user_id: str = "default"):
    global last_maintenance_time
    run_consolidate(db, graph, user_id)
    run_prune(db, graph, user_id)
    last_maintenance_time = datetime.now(timezone.utc)


def start_scheduler(db: MemoryDB, graph: MemoryGraph, user_id: str = "default"):
    try:
        from apscheduler.schedulers.background import BackgroundScheduler

        run_maintenance(db, graph, user_id)

        scheduler = BackgroundScheduler()
        scheduler.add_job(
            run_maintenance,
            "interval",
            hours=12,
            args=[db, graph, user_id],
            id="memory_maintenance",
        )
        scheduler.start()
        return scheduler
    except ImportError:
        log.warning("APScheduler not installed, maintenance disabled")
        return None
