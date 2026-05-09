"""Tests for P1-6: Event log gzip rotate.

Covers:
  - rotate_old_files() compresses old .jsonl, skips today's file
  - _sorted_event_files() and _iter_file() read both .jsonl and .jsonl.gz
  - Line-count safety check prevents data loss
  - Leftover .jsonl cleaned up when .gz already exists
  - recover module reads .gz files transparently
  - rotate_event_logs() maintenance integration
"""

import gzip
import json
import os
from datetime import datetime, timezone, timedelta

import pytest

from engram.event_log import EventLog, _count_lines, _gzip_file, _count_gz_lines


@pytest.fixture
def event_dir(tmp_path):
    d = tmp_path / "events"
    d.mkdir()
    return str(d)


def _write_jsonl(path: str, events: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def _make_events(date_str: str, count: int, start_seq: int = 1) -> list[dict]:
    return [
        {"ts": f"2026-{date_str}T00:00:00Z", "seq": start_seq + i,
         "kind": "task.create", "payload": {"task_id": i},
         "engram_version": "0.13.0", "schema_version": 1}
        for i in range(count)
    ]


# --- Helper function tests ---

class TestHelperFunctions:

    def test_count_lines(self, tmp_path):
        p = tmp_path / "test.jsonl"
        p.write_text("line1\nline2\n\nline3\n")
        assert _count_lines(str(p)) == 3  # skips empty lines

    def test_gzip_file_and_count_gz_lines(self, tmp_path):
        src = tmp_path / "test.jsonl"
        src.write_text("a\nb\nc\n")
        dst = tmp_path / "test.jsonl.gz"
        _gzip_file(str(src), str(dst))
        assert dst.exists()
        assert _count_gz_lines(str(dst)) == 3


# --- rotate_old_files ---

class TestRotateOldFiles:

    def test_compresses_old_files_not_today(self, event_dir):
        elog = EventLog(event_dir=event_dir, engram_version="0.13.0")

        # Write some events to create today's file
        elog.append("task.create", {"task_id": 1})

        # Create a "yesterday" file manually
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y%m%d")
        old_path = os.path.join(event_dir, f"events-{yesterday}.jsonl")
        _write_jsonl(old_path, _make_events("05-08", 5))

        compressed = elog.rotate_old_files()

        assert len(compressed) == 1
        assert compressed[0].endswith(".jsonl.gz")
        assert not os.path.exists(old_path)  # original removed
        assert os.path.exists(old_path + ".gz")  # gz created

    def test_does_not_compress_today(self, event_dir):
        elog = EventLog(event_dir=event_dir, engram_version="0.13.0")
        elog.append("task.create", {"task_id": 1})

        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        today_path = os.path.join(event_dir, f"events-{today}.jsonl")
        assert os.path.exists(today_path)

        compressed = elog.rotate_old_files()
        assert len(compressed) == 0
        assert os.path.exists(today_path)  # still plain

    def test_cleans_up_leftover_jsonl_when_gz_exists(self, event_dir):
        elog = EventLog(event_dir=event_dir, engram_version="0.13.0")

        old_date = "20260101"
        jsonl_path = os.path.join(event_dir, f"events-{old_date}.jsonl")
        gz_path = jsonl_path + ".gz"

        _write_jsonl(jsonl_path, _make_events("01-01", 3))
        # Simulate a previous partial rotate: .gz already exists
        _gzip_file(jsonl_path, gz_path)

        compressed = elog.rotate_old_files()
        assert len(compressed) == 0  # no new compression
        assert not os.path.exists(jsonl_path)  # leftover cleaned
        assert os.path.exists(gz_path)

    def test_empty_dir_returns_empty(self, event_dir):
        elog = EventLog(event_dir=event_dir, engram_version="0.13.0")
        assert elog.rotate_old_files() == []

    def test_multiple_old_files(self, event_dir):
        elog = EventLog(event_dir=event_dir, engram_version="0.13.0")

        for day in range(1, 4):
            date_str = f"2026010{day}"
            path = os.path.join(event_dir, f"events-{date_str}.jsonl")
            _write_jsonl(path, _make_events(f"01-0{day}", 3 + day))

        compressed = elog.rotate_old_files()
        assert len(compressed) == 3
        for gz in compressed:
            assert gz.endswith(".gz")


# --- _sorted_event_files with mixed formats ---

class TestSortedEventFilesMixed:

    def test_reads_gz_files(self, event_dir):
        elog = EventLog(event_dir=event_dir, engram_version="0.13.0")

        gz_path = os.path.join(event_dir, "events-20260101.jsonl.gz")
        src_path = os.path.join(event_dir, "events-20260101.jsonl")
        _write_jsonl(src_path, _make_events("01-01", 2))
        _gzip_file(src_path, gz_path)
        os.remove(src_path)

        files = elog._sorted_event_files(None)
        assert len(files) == 1
        assert files[0].endswith(".jsonl.gz")

    def test_jsonl_takes_precedence_over_gz(self, event_dir):
        elog = EventLog(event_dir=event_dir, engram_version="0.13.0")

        jsonl = os.path.join(event_dir, "events-20260101.jsonl")
        gz = os.path.join(event_dir, "events-20260101.jsonl.gz")
        _write_jsonl(jsonl, _make_events("01-01", 2))
        _gzip_file(jsonl, gz)

        files = elog._sorted_event_files(None)
        assert len(files) == 1
        assert files[0].endswith(".jsonl")  # plain takes precedence

    def test_mixed_formats_chronological(self, event_dir):
        elog = EventLog(event_dir=event_dir, engram_version="0.13.0")

        # Day 1: compressed
        d1_src = os.path.join(event_dir, "events-20260101.jsonl")
        d1_gz = os.path.join(event_dir, "events-20260101.jsonl.gz")
        _write_jsonl(d1_src, _make_events("01-01", 2))
        _gzip_file(d1_src, d1_gz)
        os.remove(d1_src)

        # Day 2: plain
        d2 = os.path.join(event_dir, "events-20260102.jsonl")
        _write_jsonl(d2, _make_events("01-02", 3, start_seq=3))

        files = elog._sorted_event_files(None)
        assert len(files) == 2
        assert files[0].endswith(".jsonl.gz")  # day 1
        assert files[1].endswith(".jsonl")     # day 2

    def test_since_date_filter_works_with_gz(self, event_dir):
        elog = EventLog(event_dir=event_dir, engram_version="0.13.0")

        for date in ["20260101", "20260102", "20260103"]:
            gz = os.path.join(event_dir, f"events-{date}.jsonl.gz")
            src = os.path.join(event_dir, f"events-{date}.jsonl")
            _write_jsonl(src, [{"seq": 1}])
            _gzip_file(src, gz)
            os.remove(src)

        files = elog._sorted_event_files("20260102")
        assert len(files) == 2  # 0102 and 0103


# --- iter_events reads gz transparently ---

class TestIterEventsGz:

    def test_iter_events_reads_compressed(self, event_dir):
        elog = EventLog(event_dir=event_dir, engram_version="0.13.0")

        events = _make_events("01-01", 5)
        src = os.path.join(event_dir, "events-20260101.jsonl")
        _write_jsonl(src, events)
        _gzip_file(src, src + ".gz")
        os.remove(src)

        read_back = list(elog.iter_events())
        assert len(read_back) == 5
        assert read_back[0]["seq"] == 1
        assert read_back[4]["seq"] == 5

    def test_iter_events_mixed_plain_and_gz(self, event_dir):
        elog = EventLog(event_dir=event_dir, engram_version="0.13.0")

        # Day 1: compressed (3 events)
        d1_events = _make_events("01-01", 3)
        d1_src = os.path.join(event_dir, "events-20260101.jsonl")
        _write_jsonl(d1_src, d1_events)
        _gzip_file(d1_src, d1_src + ".gz")
        os.remove(d1_src)

        # Day 2: plain (2 events)
        d2_events = _make_events("01-02", 2, start_seq=4)
        d2_path = os.path.join(event_dir, "events-20260102.jsonl")
        _write_jsonl(d2_path, d2_events)

        all_events = list(elog.iter_events())
        assert len(all_events) == 5


# --- recover reads gz ---

class TestRecoverReadsGz:

    def test_recover_from_gz_event_files(self, event_dir, tmp_path):
        from engram.recover import recover

        events = [
            {"ts": "2026-01-01T00:00:00Z", "seq": 1, "kind": "task.create",
             "payload": {"task_id": 1, "name": "test-task", "goal": "g",
                         "status": "active", "user_id": "default",
                         "metadata": None},
             "engram_version": "0.13.0", "schema_version": 1},
        ]
        src = os.path.join(event_dir, "events-20260101.jsonl")
        _write_jsonl(src, events)
        _gzip_file(src, src + ".gz")
        os.remove(src)

        output_dir = str(tmp_path / "recovered")
        report = recover(
            event_dir=event_dir,
            output_dir=output_dir,
            snapshot_dir="",  # disable snapshot lookup
        )
        assert report.counts.get("task.create", 0) == 1
        assert len(report.errors) == 0


# --- maintenance integration ---

class TestMaintenanceIntegration:

    def test_rotate_event_logs_function(self, event_dir, monkeypatch):
        from engram.maintenance import rotate_event_logs
        from engram import event_log

        elog = EventLog(event_dir=event_dir, engram_version="0.13.0")
        monkeypatch.setattr(event_log, "_singleton", elog)

        old_date = "20260101"
        path = os.path.join(event_dir, f"events-{old_date}.jsonl")
        _write_jsonl(path, _make_events("01-01", 3))

        result = rotate_event_logs()
        assert len(result) == 1
        assert result[0].endswith(".gz")
