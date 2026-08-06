import json
from pathlib import Path

from worker.sinks.metrics import MetricsSink


def test_report_appends_one_line(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    sink = MetricsSink(path)
    sink.report(10, {"episode_return": 1.5})
    sink.report(20, {"episode_return": 2.5, "episode_length": 7})
    sink.close()
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines == [
        {"step": 10, "metrics": {"episode_return": 1.5}},
        {"step": 20, "metrics": {"episode_return": 2.5, "episode_length": 7.0}},
    ]


def test_report_flushes_before_close(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    sink = MetricsSink(path)
    sink.report(5, {"heartbeat": 1.0})
    with path.open(encoding="utf-8") as reader:
        line = reader.read().strip()
    assert json.loads(line) == {"step": 5, "metrics": {"heartbeat": 1.0}}
    sink.close()


def test_the_file_appears_only_with_the_first_report(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "metrics.jsonl"
    sink = MetricsSink(path)
    assert not path.exists()
    sink.report(1, {"m": 1.0})
    assert path.exists()
    sink.close()


def test_close_without_a_report_leaves_no_file(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    MetricsSink(path).close()
    assert not path.exists()


def test_report_updates_modification_time(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    sink = MetricsSink(path)
    sink.report(1, {"m": 1.0})
    first = path.stat().st_mtime_ns
    sink.report(2, {"m": 2.0})
    assert path.stat().st_mtime_ns >= first
    sink.close()
