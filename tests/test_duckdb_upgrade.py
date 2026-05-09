"""Tests for P1-5: DuckDB version upgrade detection.

Verifies:
- _is_breaking_upgrade triggers on minor version change but not patch
- detect_duckdb_upgrade no-ops when no prior version, no DB file, or no change
- detect_duckdb_upgrade copies DB to backup with embedded version info on real upgrade
- backup file naming follows the documented pattern
"""

from __future__ import annotations

import os

from engram.maintenance import (
    _is_breaking_upgrade,
    detect_duckdb_upgrade,
)


def test_is_breaking_upgrade_classification():
    # No prior version -> never breaking (fresh install).
    assert _is_breaking_upgrade("", "1.5.2") is False
    assert _is_breaking_upgrade("1.5.2", "") is False

    # Same version -> never breaking.
    assert _is_breaking_upgrade("1.5.2", "1.5.2") is False

    # Patch-only change -> not breaking (DuckDB historically safe here).
    assert _is_breaking_upgrade("1.5.1", "1.5.2") is False

    # Minor change -> breaking (DuckDB has broken file format on minor bumps).
    assert _is_breaking_upgrade("1.5.2", "1.6.0") is True

    # Major change -> breaking.
    assert _is_breaking_upgrade("0.9.2", "1.5.2") is True

    # Numeric (not lexical!) comparison: 0.9 vs 0.10 must be detected.
    assert _is_breaking_upgrade("0.9.2", "0.10.0") is True


def test_detect_upgrade_noop_without_prior_version(tmp_path):
    db_path = tmp_path / "fake.duckdb"
    db_path.write_bytes(b"not-a-real-db-but-good-enough")

    report = detect_duckdb_upgrade(
        db_path=str(db_path),
        old_version=None,
        new_version="1.5.2",
        backup_dir=str(tmp_path / "backups"),
    )
    assert report is None
    # No backup dir created.
    assert not (tmp_path / "backups").exists()


def test_detect_upgrade_noop_when_db_missing(tmp_path):
    """Fresh install on a new DuckDB version — nothing to back up."""
    report = detect_duckdb_upgrade(
        db_path=str(tmp_path / "no-such-file.duckdb"),
        old_version="1.5.1",
        new_version="1.6.0",
        backup_dir=str(tmp_path / "backups"),
    )
    assert report is None


def test_detect_upgrade_noop_on_patch_change(tmp_path):
    db_path = tmp_path / "fake.duckdb"
    db_path.write_bytes(b"real content")
    report = detect_duckdb_upgrade(
        db_path=str(db_path),
        old_version="1.5.1",
        new_version="1.5.2",
        backup_dir=str(tmp_path / "backups"),
    )
    assert report is None


def test_detect_upgrade_copies_on_real_change(tmp_path):
    db_path = tmp_path / "fake.duckdb"
    payload = b"original-db-bytes-must-survive"
    db_path.write_bytes(payload)
    backup_dir = tmp_path / "backups"

    report = detect_duckdb_upgrade(
        db_path=str(db_path),
        old_version="0.9.2",
        new_version="1.5.2",
        backup_dir=str(backup_dir),
    )
    assert report is not None
    assert report["old_version"] == "0.9.2"
    assert report["new_version"] == "1.5.2"
    assert report["db_path"] == str(db_path)

    # Naming pattern includes both versions, so the operator can see the jump.
    bp = report["backup_path"]
    assert os.path.basename(bp).startswith(
        "memories-pre-duckdb-upgrade-0.9.2-to-1.5.2-"
    )
    assert bp.endswith(".duckdb")
    # Original is preserved (we copy, not move).
    assert db_path.read_bytes() == payload
    # Backup contents match the source byte-for-byte.
    with open(bp, "rb") as f:
        assert f.read() == payload


def test_detect_upgrade_handles_minor_version_jump(tmp_path):
    db_path = tmp_path / "fake.duckdb"
    db_path.write_bytes(b"db content")
    report = detect_duckdb_upgrade(
        db_path=str(db_path),
        old_version="0.9.2",
        new_version="0.10.0",
        backup_dir=str(tmp_path / "backups"),
    )
    assert report is not None
    assert "0.9.2-to-0.10.0" in os.path.basename(report["backup_path"])
