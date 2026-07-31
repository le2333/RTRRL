import json
from pathlib import Path

import pytest
from aim import Repo

from training_sdk.reporter import METRICS_FILENAME, Reporter
from training_sdk.sinks.aim import AimSink, close_aim_run
from tests.test_reporter import make_config


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
    assert run["params"]["total_steps"] == 4
    values = list(run.metrics())
    assert values, "the metric sequence must exist"
    close_aim_run(run)


def test_read_only_run_close_raises_sequence_infos_attribute_error(tmp_path: Path) -> None:
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


def test_every_step_reported_when_every_steps_is_one(tmp_path: Path) -> None:
    repo_path = str(tmp_path / "aim")
    Repo.from_path(repo_path, init=True)
    sink = AimSink(make_config(every_steps=1), repo=repo_path)
    for step in (0, 1, 2, 3):
        sink.report(step, {"episode_return": float(step)})
    sink.close()

    assert _metric_steps(repo_path) == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("steps", "expected"),
    [
        ((0, 1024, 2048), [0, 1024, 2048]),
        ((0, 300, 600, 1200), [0, 1200]),
    ],
)
def test_throttle_uses_elapsed_steps_not_modulus(
    tmp_path: Path, steps: tuple[int, ...], expected: list[int]
) -> None:
    repo_path = str(tmp_path / "aim")
    Repo.from_path(repo_path, init=True)
    sink = AimSink(make_config(every_steps=1000), repo=repo_path)
    for step in steps:
        sink.report(step, {"episode_return": float(step)})
    sink.close()

    assert _metric_steps(repo_path) == expected


def test_two_reports_at_one_step_both_arrive(tmp_path: Path) -> None:
    # A training epoch reports its own metrics and then its evaluation metrics
    # at the same step. Throttling is meant to thin out a stream that advances
    # too fast, not to drop the second half of one step's report.
    repo_path = str(tmp_path / "aim")
    Repo.from_path(repo_path, init=True)
    sink = AimSink(make_config(every_steps=1), repo=repo_path)
    sink.report(25600, {"train_return": 1.0})
    sink.report(25600, {"eval_return": 2.0})
    sink.close()

    assert _metric_steps(repo_path, "train_return") == [25600]
    assert _metric_steps(repo_path, "eval_return") == [25600]


def test_metrics_file_keeps_every_report_while_aim_is_throttled(tmp_path: Path) -> None:
    repo_path = str(tmp_path / "aim")
    Repo.from_path(repo_path, init=True)
    config = make_config(every_steps=1000)
    steps = (0, 512, 1024, 1536, 2048)

    with Reporter(config, tmp_path, sinks=[AimSink(config, repo=repo_path)]) as reporter:
        for step in steps:
            reporter.report(step, {"episode_return": float(step)})

    assert _metric_steps(repo_path) == [0, 1024, 2048]
    metrics_path = tmp_path / METRICS_FILENAME
    records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert [record["step"] for record in records] == list(steps)
    assert records[2]["metrics"]["episode_return"] == 1024.0
