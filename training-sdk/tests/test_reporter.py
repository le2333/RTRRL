import json
from pathlib import Path

from training_sdk.contract import RunConfig
from training_sdk.reporter import METRICS_FILENAME, Reporter


def make_config() -> RunConfig:
    return RunConfig.model_validate(
        {
            "contract": 2,
            "run_id": "smoke-20260725-000000-t0",
            "experiment": "infra-acceptance",
            "name": "smoke",
            "launch_id": "20260725-000000",
            "trial": 0,
            "entry": "e",
            "params": {"total_steps": 4},
            "logging": {"aim": "aim://127.0.0.1:1", "every_steps": 1},
            "score": {
                "metric": "episode_return",
                "window_steps": [0, 4],
                "reduce": "mean",
                "direction": "maximize",
                "non_finite": "worst",
                "s3": "s3://bucket/score.json",
            },
        }
    )


class RecordingSink:
    def __init__(self) -> None:
        self.reports: list[tuple[int, dict[str, float]]] = []
        self.closed = False

    def report(self, step: int, metrics: dict[str, float]) -> None:
        self.reports.append((step, dict(metrics)))

    def log_episode(self, episode: object) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def test_reporter_fans_out_and_writes_metrics_file(tmp_path: Path) -> None:
    sink = RecordingSink()
    with Reporter(make_config(), tmp_path, sinks=[sink]) as reporter:
        reporter.report(1, {"episode_return": 3.0})
    assert sink.reports == [(1, {"episode_return": 3.0})]
    assert sink.closed is True
    written = json.loads((tmp_path / METRICS_FILENAME).read_text().strip())
    assert written == {"step": 1, "metrics": {"episode_return": 3.0}}


def test_reporter_closes_every_sink_even_when_one_raises(tmp_path: Path) -> None:
    class Failing(RecordingSink):
        def close(self) -> None:
            raise RuntimeError("sink failed to close")

    failing, healthy = Failing(), RecordingSink()
    reporter = Reporter(make_config(), tmp_path, sinks=[failing, healthy])
    try:
        reporter.close()
    except RuntimeError:
        pass
    else:  # pragma: no cover - the test asserts the raise happens
        raise AssertionError("close must propagate the failure")
    assert healthy.closed is True
