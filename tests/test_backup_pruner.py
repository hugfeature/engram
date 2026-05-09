"""Tests for P1-3: backup pruning policy.

Verifies:
- _list_managed_backups picks up only files matching our prefixes
- prune_backups archives oldest-first when count exceeds retain
- ENGRAM_BACKUP_RETAIN env var overrides default
- archive collisions get a timestamp suffix instead of overwriting
"""

from __future__ import annotations

import os
import time

import pytest

from engram.maintenance import (
    DEFAULT_BACKUP_RETAIN,
    ENV_BACKUP_RETAIN,
    _list_managed_backups,
    _read_retain_count,
    prune_backups,
)


def _touch(path: str, mtime: float) -> None:
    with open(path, "wb") as f:
        f.write(b"x")
    os.utime(path, (mtime, mtime))


def _make_backup(dir_: str, name: str, age_seconds: float) -> str:
    path = os.path.join(dir_, name)
    _touch(path, time.time() - age_seconds)
    return path


def test_list_managed_backups_only_picks_our_prefixes(tmp_path):
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    keep1 = _make_backup(str(backup_dir), "memories-pre-recover-20260101-000000.duckdb", 100)
    keep2 = _make_backup(str(backup_dir), "memories-pre-duckdb-upgrade-1.5.1-to-1.5.2-x.duckdb", 50)
    # Files we should NOT touch:
    _make_backup(str(backup_dir), "user-manual-snapshot.duckdb", 10)
    _make_backup(str(backup_dir), "README.txt", 10)

    managed = _list_managed_backups(str(backup_dir))
    assert set(managed) == {keep1, keep2}
    # And sorted oldest-first.
    assert managed[0] == keep1
    assert managed[1] == keep2


def test_list_managed_backups_returns_empty_for_missing_dir(tmp_path):
    assert _list_managed_backups(str(tmp_path / "does-not-exist")) == []


def test_prune_backups_noop_when_under_limit(tmp_path):
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    _make_backup(str(backup_dir), "memories-pre-recover-001.duckdb", 100)
    _make_backup(str(backup_dir), "memories-pre-recover-002.duckdb", 50)

    report = prune_backups(
        backup_dir=str(backup_dir),
        archive_dir=str(tmp_path / "archive"),
        retain=5,
    )
    assert report["archived"] == []
    assert report["kept"] == 2
    assert report["scanned"] == 2
    # Archive dir was never created (no work to do).
    assert not (tmp_path / "archive").exists()


def test_prune_backups_archives_oldest_first(tmp_path):
    backup_dir = tmp_path / "backups"
    archive_dir = tmp_path / "backups" / "archive"
    backup_dir.mkdir()

    # Five backups with descending age.
    paths = []
    for i, age in enumerate([500, 400, 300, 200, 100]):
        paths.append(_make_backup(
            str(backup_dir),
            f"memories-pre-recover-{i:03d}.duckdb",
            age,
        ))

    report = prune_backups(
        backup_dir=str(backup_dir),
        archive_dir=str(archive_dir),
        retain=2,
    )
    # Oldest 3 should be archived, newest 2 kept.
    assert report["kept"] == 2
    assert len(report["archived"]) == 3
    # Verify the right files moved.
    assert not os.path.exists(paths[0])
    assert not os.path.exists(paths[1])
    assert not os.path.exists(paths[2])
    assert os.path.exists(paths[3])
    assert os.path.exists(paths[4])
    # Archive contains the moved files.
    archived_names = sorted(os.listdir(archive_dir))
    assert archived_names == [
        "memories-pre-recover-000.duckdb",
        "memories-pre-recover-001.duckdb",
        "memories-pre-recover-002.duckdb",
    ]


def test_prune_backups_handles_archive_name_collision(tmp_path):
    backup_dir = tmp_path / "backups"
    archive_dir = tmp_path / "backups" / "archive"
    backup_dir.mkdir()
    archive_dir.mkdir(parents=True)

    # Pre-existing file in archive with same name as one we'll archive.
    pre_existing = archive_dir / "memories-pre-recover-001.duckdb"
    pre_existing.write_bytes(b"old-archived-content")

    # Two live backups; force one to be archived.
    _make_backup(str(backup_dir), "memories-pre-recover-001.duckdb", 100)
    _make_backup(str(backup_dir), "memories-pre-recover-002.duckdb", 50)

    report = prune_backups(
        backup_dir=str(backup_dir),
        archive_dir=str(archive_dir),
        retain=1,
    )
    assert len(report["archived"]) == 1
    # Pre-existing file untouched.
    assert pre_existing.read_bytes() == b"old-archived-content"
    # New archive entry got a disambiguating suffix.
    archived = report["archived"][0]
    assert archived != str(pre_existing)
    assert os.path.basename(archived).startswith("memories-pre-recover-001.duckdb.")


def test_read_retain_count_env_override(monkeypatch):
    monkeypatch.delenv(ENV_BACKUP_RETAIN, raising=False)
    assert _read_retain_count() == DEFAULT_BACKUP_RETAIN

    monkeypatch.setenv(ENV_BACKUP_RETAIN, "3")
    assert _read_retain_count() == 3

    monkeypatch.setenv(ENV_BACKUP_RETAIN, "0")
    assert _read_retain_count() == 1  # clamped to >= 1

    monkeypatch.setenv(ENV_BACKUP_RETAIN, "garbage")
    assert _read_retain_count() == DEFAULT_BACKUP_RETAIN
