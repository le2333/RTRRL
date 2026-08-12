import json

from memorax.observability.sinks.metrics import MetricsSink


def test_metrics_sink_keeps_the_worker_score_artifact_format(tmp_path):
    path = tmp_path / "metrics.jsonl"
    sink = MetricsSink(path)

    sink.report(8, {"train/episode/return": 4.0})
    sink.close()

    assert json.loads(path.read_text()) == {
        "step": 8,
        "metrics": {"train/episode/return": 4.0},
    }


def test_each_report_is_appended_and_flushed_immediately(tmp_path):
    path = tmp_path / "metrics.jsonl"
    sink = MetricsSink(path)
    sink.report(10, {"return": 1.5})

    assert json.loads(path.read_text()) == {
        "step": 10,
        "metrics": {"return": 1.5},
    }

    sink.report(20, {"return": 2.5, "length": 7})
    sink.close()
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"step": 10, "metrics": {"return": 1.5}},
        {"step": 20, "metrics": {"length": 7.0, "return": 2.5}},
    ]


def test_artifact_is_created_only_when_a_scalar_is_reported(tmp_path):
    path = tmp_path / "nested" / "metrics.jsonl"
    sink = MetricsSink(path)
    sink.close()
    assert not path.exists()

    sink.report(1, {"m": 1.0})
    assert path.exists()
    sink.close()
