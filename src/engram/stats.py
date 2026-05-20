"""Engram Stats — runtime usage analytics from Event Journal + Tier 2.

Answers: "How much is Engram actually helping?"
Aggregates checkpoint usage, drift events, recovery quality, and session patterns.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

from .event_log import EventLog, DEFAULT_EVENT_DIR

log = logging.getLogger("engram.stats")


@dataclass
class EngineStats:
    """Aggregated Engram runtime statistics."""

    period_days: int = 7
    period_start: str = ""
    period_end: str = ""

    # Sessions
    total_sessions: int = 0
    interrupted_sessions: int = 0
    recovered_sessions: int = 0

    # Checkpoints
    checkpoints_created: int = 0
    checkpoints_restored: int = 0
    checkpoints_never_used: int = 0
    checkpoint_reasons: dict[str, int] = field(default_factory=dict)

    # Drift
    drift_events: int = 0
    drift_avg_composite: float = 0.0
    drift_nudges_emitted: int = 0
    top_violations: list[str] = field(default_factory=list)

    # Continuity
    continuity_scores: list[float] = field(default_factory=list)
    continuity_avg: float = 0.0
    continuity_p50: float = 0.0
    continuity_p90: float = 0.0

    # Must-not-redo
    must_not_redo_saves: int = 0

    # Memories
    memories_stored: int = 0
    memories_recalled: int = 0

    def to_dict(self) -> dict:
        return {
            "period": {
                "days": self.period_days,
                "start": self.period_start,
                "end": self.period_end,
            },
            "sessions": {
                "total": self.total_sessions,
                "interrupted": self.interrupted_sessions,
                "recovered": self.recovered_sessions,
            },
            "checkpoints": {
                "created": self.checkpoints_created,
                "restored": self.checkpoints_restored,
                "never_used": self.checkpoints_never_used,
                "by_reason": self.checkpoint_reasons,
            },
            "drift": {
                "events": self.drift_events,
                "avg_composite": round(self.drift_avg_composite, 4),
                "nudges_emitted": self.drift_nudges_emitted,
                "top_violations": self.top_violations[:5],
            },
            "continuity": {
                "avg": round(self.continuity_avg, 4),
                "p50": round(self.continuity_p50, 4),
                "p90": round(self.continuity_p90, 4),
                "samples": len(self.continuity_scores),
            },
            "must_not_redo_saves": self.must_not_redo_saves,
            "memories": {
                "stored": self.memories_stored,
                "recalled": self.memories_recalled,
            },
        }

    def format_table(self) -> str:
        """Format as a human-readable terminal table."""
        lines = [
            f"Engram Stats ({self.period_days}d: {self.period_start} → {self.period_end})",
            "=" * 60,
            "",
            "Sessions",
            f"  Total:        {self.total_sessions}",
            f"  Interrupted:  {self.interrupted_sessions}",
            f"  Recovered:    {self.recovered_sessions}",
            f"  Recovery %:   {_pct(self.recovered_sessions, self.interrupted_sessions)}",
            "",
            "Checkpoints",
            f"  Created:      {self.checkpoints_created}",
            f"  Restored:     {self.checkpoints_restored}",
            f"  Hit rate:     {_pct(self.checkpoints_restored, self.checkpoints_created)}",
        ]

        if self.checkpoint_reasons:
            lines.append("  By reason:")
            for reason, count in sorted(self.checkpoint_reasons.items(), key=lambda x: -x[1]):
                lines.append(f"    {reason:30s} {count}")

        lines += [
            "",
            "Drift",
            f"  Events:       {self.drift_events}",
            f"  Avg score:    {self.drift_avg_composite:.3f}",
            f"  Nudges:       {self.drift_nudges_emitted}",
        ]

        if self.top_violations:
            lines.append("  Top violations:")
            for violation in self.top_violations[:3]:
                lines.append(f"    - {violation}")

        lines += [
            "",
            "Continuity Quality",
            f"  Avg:          {self.continuity_avg:.3f}",
            f"  P50:          {self.continuity_p50:.3f}",
            f"  P90:          {self.continuity_p90:.3f}",
            f"  Samples:      {len(self.continuity_scores)}",
            "",
            "Protection",
            f"  Must-not-redo saves: {self.must_not_redo_saves}",
            "",
            "Memory Activity",
            f"  Stored:       {self.memories_stored}",
            f"  Recalled:     {self.memories_recalled}",
        ]

        return "\n".join(lines)


    def format_report(self) -> str:
        """Format as a markdown report suitable for pasting into docs/issues."""
        lines = [
            f"## Engram Runtime Report",
            f"**Period:** {self.period_start} → {self.period_end} ({self.period_days} days)",
            "",
            "### Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Sessions | {self.total_sessions} total, {self.interrupted_sessions} interrupted |",
            f"| Recovery rate | {_pct(self.recovered_sessions, self.interrupted_sessions)} ({self.recovered_sessions}/{self.interrupted_sessions}) |",
            f"| Checkpoints | {self.checkpoints_created} created, {self.checkpoints_restored} restored |",
            f"| Checkpoint hit rate | {_pct(self.checkpoints_restored, self.checkpoints_created)} |",
            f"| Continuity score (avg) | {self.continuity_avg:.3f} |",
            f"| Drift events | {self.drift_events} (avg composite: {self.drift_avg_composite:.3f}) |",
            f"| Drift nudges | {self.drift_nudges_emitted} |",
            f"| Must-not-redo saves | {self.must_not_redo_saves} |",
            f"| Memories stored/recalled | {self.memories_stored}/{self.memories_recalled} |",
            "",
        ]

        # Highlights / anomalies
        highlights = []
        if self.interrupted_sessions > 0 and self.recovered_sessions == 0:
            highlights.append("⚠️ **No recoveries** despite interruptions — agents may not be calling `restore_checkpoint`")
        if self.drift_avg_composite > 0.5:
            highlights.append(f"⚠️ **High average drift** ({self.drift_avg_composite:.2f}) — agents are frequently diverging from checkpoints")
        if self.checkpoints_created > 0 and self.checkpoints_restored == 0:
            highlights.append("ℹ️ **Zero checkpoint restores** — checkpoints are being created but never used")
        if self.drift_nudges_emitted > 0:
            highlights.append(f"✅ **Drift nudge active** — {self.drift_nudges_emitted} auto-corrections emitted")
        if self.continuity_avg > 0.8:
            highlights.append(f"✅ **High continuity** ({self.continuity_avg:.2f}) — recoveries are preserving agent state well")

        if highlights:
            lines += ["### Highlights", ""]
            lines += [f"- {h}" for h in highlights]
            lines += [""]

        # Checkpoint breakdown
        if self.checkpoint_reasons:
            lines += ["### Checkpoint Breakdown", "", "| Reason | Count |", "|--------|-------|"]
            for reason, count in sorted(self.checkpoint_reasons.items(), key=lambda x: -x[1]):
                lines += [f"| {reason} | {count} |"]
            lines += [""]

        # Top violations
        if self.top_violations:
            lines += ["### Top Constraint Violations", ""]
            for v in self.top_violations[:5]:
                lines += [f"- {v}"]
            lines += [""]

        return "\n".join(lines)


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "—"
    return f"{numerator / denominator * 100:.1f}%"


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * p)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def compute_stats(days: int = 7, event_dir: str | None = None) -> EngineStats:
    """Compute runtime stats by scanning the Event Journal.

    Reads events from the last N days and aggregates into EngineStats.
    """
    event_dir = event_dir or DEFAULT_EVENT_DIR
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    since_date = cutoff.strftime("%Y%m%d")

    stats = EngineStats(
        period_days=days,
        period_start=cutoff.strftime("%Y-%m-%d"),
        period_end=now.strftime("%Y-%m-%d"),
    )

    if not os.path.isdir(event_dir):
        return stats

    event_log = EventLog(event_dir=event_dir)
    drift_composites: list[float] = []
    all_violations: list[str] = []
    checkpoint_ids_created: set[str] = set()
    checkpoint_ids_restored: set[str] = set()

    for event in event_log.iter_events(since_date=since_date):
        kind = event.get("kind", "")
        payload = event.get("payload", {})

        if kind == "session.start":
            stats.total_sessions += 1

        elif kind == "session.end":
            end_type = payload.get("end_type", "normal")
            if end_type in ("interrupted", "sigterm", "context_overflow", "crash"):
                stats.interrupted_sessions += 1

        elif kind == "checkpoint.write":
            stats.checkpoints_created += 1
            reason = payload.get("checkpoint_reason") or payload.get("reason", "unknown")
            stats.checkpoint_reasons[reason] = stats.checkpoint_reasons.get(reason, 0) + 1
            task_id = payload.get("task_id")
            version = payload.get("version")
            if task_id is not None and version is not None:
                checkpoint_ids_created.add(f"{task_id}:{version}")

        elif kind == "checkpoint.restore":
            stats.checkpoints_restored += 1
            stats.recovered_sessions += 1
            task_id = payload.get("task_id")
            version = payload.get("version")
            if task_id is not None and version is not None:
                checkpoint_ids_restored.add(f"{task_id}:{version}")

        elif kind == "drift.detected":
            stats.drift_events += 1
            composite = payload.get("composite", 0.0)
            drift_composites.append(composite)
            violations = payload.get("violations") or []
            all_violations.extend(violations)

        elif kind == "drift.nudge":
            stats.drift_nudges_emitted += 1

        elif kind == "continuity.evaluated":
            composite = payload.get("composite", 0.0)
            stats.continuity_scores.append(composite)

        elif kind == "continuity.redundant_exploration":
            stats.must_not_redo_saves += 1

        elif kind == "memory.store":
            stats.memories_stored += 1

        elif kind == "session.memory_recall":
            stats.memories_recalled += 1

    # Compute aggregates
    if drift_composites:
        stats.drift_avg_composite = sum(drift_composites) / len(drift_composites)

    if stats.continuity_scores:
        stats.continuity_avg = sum(stats.continuity_scores) / len(stats.continuity_scores)
        stats.continuity_p50 = _percentile(stats.continuity_scores, 0.5)
        stats.continuity_p90 = _percentile(stats.continuity_scores, 0.9)

    # Top violations by frequency
    violation_counts = Counter(all_violations)
    stats.top_violations = [v for v, _ in violation_counts.most_common(5)]

    # Never-used checkpoints
    stats.checkpoints_never_used = len(checkpoint_ids_created - checkpoint_ids_restored)

    return stats
