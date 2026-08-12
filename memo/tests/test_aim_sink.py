import json
from pathlib import Path

import pytest
from aim import Repo
from test_reporter import make_config, make_episode

from worker.reporter import METRICS_FILENAME, Reporter
from worker.sinks.aim import AimSink, close_aim_run

pytestmark = [pytest.mark.integration, pytest.mark.service]


def _metric_steps(repo_path: str, metric_name: str = "episode_return") -> list[int]:
    repo = Repo.from_path(repo_path)
    for run in repo.iter_runs():
        for metric in run.metrics():
            if metric.name == metric_name:
                steps, _ = metric.data.items_list()
                close_aim_run(run)
                return list(steps)
    raise AssertionError(f"metric {metric_name!r} not found in {repo_path}")


def _open_readonly_run(repo_path: str):
    repo = Repo.from_path(repo_path, read_only=True)
    runs = list(repo.iter_runs())
    assert len(runs) == 1
    return runs[0]


def test_run_is_named_by_run_id_and_carries_launch_fields(tmp_path: Path) -> None:
    repo_path = str(tmp_path / "aim")
    Repo.from_path(repo_path, init=True)
    config = make_config()

    sink = AimSink(config, repo=repo_path)
    sink.report(1, {"episode_return": 2.0})
    sink.report(2, {"episode_return": 4.0})
    sink.close()

    repo = Repo.from_path(repo_path)
    runs = list(repo.iter_runs())
    assert len(runs) == 1
    run = runs[0]
    assert run.name == config.run_id
    assert run["launch_id"] == config.launch_id
    assert run["trial"] == config.trial
    assert run["entry"] == config.entry
    assert run["digest"] == config.digest
    assert run["params"]["learning_rate"] == 0.0003
    values = list(run.metrics())
    assert values, "the metric sequence must exist"
    close_aim_run(run)


def test_read_only_run_close_raises_sequence_infos_attribute_error(
    tmp_path: Path,
) -> None:
    # Documents Aim 3.28.0: Run.close() on a read-only run fails because
    # RunTracker never creates sequence_infos. If this test fails, Aim fixed
    # the bug and close_aim_run() can be retired.
    repo_path = str(tmp_path / "aim")
    Repo.from_path(repo_path, init=True)
    sink = AimSink(make_config(), repo=repo_path)
    sink.report(1, {"episode_return": 1.0})
    sink.close()

    run = _open_readonly_run(repo_path)
    with pytest.raises(AttributeError, match="sequence_infos"):
        run.close()


def test_close_aim_run_on_read_only_run_does_not_raise_and_data_readable(
    tmp_path: Path,
) -> None:
    repo_path = str(tmp_path / "aim")
    Repo.from_path(repo_path, init=True)
    sink = AimSink(make_config(), repo=repo_path)
    sink.report(1, {"episode_return": 1.0})
    sink.close()

    run = _open_readonly_run(repo_path)
    for metric in run.metrics():
        if metric.name == "episode_return":
            steps, _ = metric.data.items_list()
            assert list(steps) == [1]
            assert metric.data.values_list()[0] == [1.0]
            break
    else:
        raise AssertionError("episode_return metric not found")
    close_aim_run(run)


def test_every_report_reaches_aim(tmp_path: Path) -> None:
    """Nothing thins the stream: an episode is already the thinning.

    Reporting used to happen on a fixed step interval and a stride discarded
    most of it. Both were arrangements of a granularity nothing measured at.
    """

    repo_path = str(tmp_path / "aim")
    Repo.from_path(repo_path, init=True)
    sink = AimSink(make_config(), repo=repo_path)
    for step in (0, 300, 600, 1200):
        sink.report(step, {"episode_return": float(step)})
    sink.close()

    assert _metric_steps(repo_path) == [0, 300, 600, 1200]


def test_two_reports_at_one_step_both_arrive(tmp_path: Path) -> None:
    # Two streams whose episodes ended at the same step are two reports at that
    # step, and neither is a repeat of the other.
    repo_path = str(tmp_path / "aim")
    Repo.from_path(repo_path, init=True)
    sink = AimSink(make_config(), repo=repo_path)
    sink.report(25600, {"train_return": 1.0})
    sink.report(25600, {"eval_return": 2.0})
    sink.close()

    assert _metric_steps(repo_path, "train_return") == [25600]
    assert _metric_steps(repo_path, "eval_return") == [25600]


def test_an_episodes_statistics_reach_aim_at_the_step_it_ended_on(
    tmp_path: Path,
) -> None:
    """``AimSink.log_episode`` was a no-op, which is why none of this arrived."""

    repo_path = str(tmp_path / "aim")
    Repo.from_path(repo_path, init=True)
    config = make_config().model_copy(
        update={"logging": make_config().logging.model_copy(update={"aim": repo_path})}
    )

    with Reporter(
        config, tmp_path, sinks=[AimSink(config, repo=repo_path)]
    ) as reporter:
        reporter.log_episode(make_episode())

    assert _metric_steps(repo_path, "train/episode/return") == [8]
    metrics_path = tmp_path / METRICS_FILENAME
    records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert records[0]["metrics"]["train/episode/return_per_step"] == 2.0
