import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from memorax.runtime.episode import Episode
from worker.contract import RunConfig
from worker.reporter import METRICS_FILENAME, Reporter


def make_config(**logging: object) -> RunConfig:
    return RunConfig.model_validate(
        {
            "contract": 6,
            "run_id": "smoke-20260725-000000-t0",
            "experiment": "infra-acceptance",
            "name": "smoke",
            "launch_id": "20260725-000000",
            "trial": 0,
            "entry": "e",
            "digest": "registry.example/trainer@sha256:" + "a" * 64,
            "environment": {
                "id": "brax::hopper",
                "backend": "spring",
                "seed": 0,
            },
            "training": {"num_envs": 1, "total_steps": 100, "epoch_steps": 100},
            "evaluation": {"steps": 0, "num_envs": 1},
            "params": {"learning_rate": 0.0003},
            "logging": {"aim": "aim://127.0.0.1:1", **logging},
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
        self.episodes: list[object] = []
        self.closed = False

    def report(self, step: int, metrics: dict[str, float]) -> None:
        self.reports.append((step, dict(metrics)))

    def log_episode(self, episode: object) -> None:
        self.episodes.append(episode)

    def close(self) -> None:
        self.closed = True


class FailingRecordingSink(RecordingSink):
    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def close(self) -> None:
        raise RuntimeError(self._message)


def test_reporter_from_env_reads_config_and_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config()
    config_path = tmp_path / "run-config.json"
    aim_repo = tmp_path / "aim"
    config_data = config.model_dump()
    config_data["logging"] = {**config_data["logging"], "aim": str(aim_repo)}
    config_path.write_text(json.dumps(config_data), encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("TRAINER_RUN_CONFIG", str(config_path))
    monkeypatch.setenv("TRAINER_SCRATCH", str(scratch))

    reporter = Reporter.from_env()

    assert reporter.config.run_id == "smoke-20260725-000000-t0"
    assert reporter.config.trial == 0
    reporter.report(3, {"episode_return": 9.0})
    reporter.close()

    written = json.loads((scratch / METRICS_FILENAME).read_text().strip())
    assert written == {"step": 3, "metrics": {"episode_return": 9.0}}


def test_reporter_fans_out_and_writes_metrics_file(tmp_path: Path) -> None:
    sink = RecordingSink()
    with Reporter(make_config(), tmp_path, sinks=[sink]) as reporter:
        reporter.report(1, {"episode_return": 3.0})
    assert sink.reports == [(1, {"episode_return": 3.0})]
    assert sink.closed is True
    written = json.loads((tmp_path / METRICS_FILENAME).read_text().strip())
    assert written == {"step": 1, "metrics": {"episode_return": 3.0}}


def test_reporter_close_first_sink_raises_still_closes_second(tmp_path: Path) -> None:
    first = FailingRecordingSink("aim sink failed to close")
    second = RecordingSink()
    reporter = Reporter(make_config(), tmp_path, sinks=[first, second])
    with pytest.raises(RuntimeError, match="aim sink failed to close"):
        reporter.close()
    assert second.closed is True


def test_reporter_close_both_sinks_raise_propagates_first(tmp_path: Path) -> None:
    first = FailingRecordingSink("first failure")
    second = FailingRecordingSink("second failure")
    reporter = Reporter(make_config(), tmp_path, sinks=[first, second])
    with pytest.raises(RuntimeError, match="first failure"):
        reporter.close()


def make_episode() -> Episode:
    return Episode(
        number=1,
        phase="train",
        start_env_steps=0,
        end_env_steps=8,
        rewards=[1.0, 3.0],
        terminals=[False, True],
        truncations=[False, False],
        series={"td_error": [0.0, 2.0]},
    )


def test_an_episode_arrives_as_statistics_and_as_the_series_they_came_from(
    tmp_path: Path,
) -> None:
    """One occasion, two readings of it.

    A scalar is what two runs are compared on and a series is what one run is
    looked at, so the episode carries both and each sink takes the one it holds.
    """

    sink = RecordingSink()
    with Reporter(make_config(), tmp_path, sinks=[sink]) as reporter:
        reporter.log_episode(make_episode())

    assert sink.episodes == [make_episode()]
    step, metrics = sink.reports[0]
    assert step == 8
    assert metrics["train/episode/return"] == 4.0
    assert metrics["train/episode/td_error_variance"] == 1.0


def test_an_episode_is_dated_by_the_step_it_ended_on(tmp_path: Path) -> None:
    sink = RecordingSink()
    with Reporter(make_config(), tmp_path, sinks=[sink]) as reporter:
        reporter.log_episode(make_episode())

    written = json.loads((tmp_path / METRICS_FILENAME).read_text().strip())
    assert written["step"] == 8


def test_an_experiment_naming_a_reporting_stride_is_refused() -> None:
    """There is no stride to name: an episode ends when the environment says.

    ``every_steps`` discarded writes of an already-computed mean while its name
    claimed a granularity nothing ever measured at.
    """

    with pytest.raises(ValidationError, match="every_steps"):
        make_config(every_steps=1000)
