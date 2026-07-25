from pathlib import Path

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
    assert run["source_hash"] == config.source_hash
    assert run["params"]["total_steps"] == 4
    values = list(run.metrics())
    assert values, "the metric sequence must exist"
    close_aim_run(run)


def test_reading_a_finished_run_and_closing_it_does_not_raise(tmp_path: Path) -> None:
    repo_path = str(tmp_path / "aim")
    Repo.from_path(repo_path, init=True)
    sink = AimSink(make_config(), repo=repo_path)
    sink.report(1, {"episode_return": 1.0})
    sink.close()

    repo = Repo.from_path(repo_path, read_only=True)
    for run in repo.iter_runs():
        close_aim_run(run)


def test_every_step_reported_when_every_steps_is_one(tmp_path: Path) -> None:
    repo_path = str(tmp_path / "aim")
    Repo.from_path(repo_path, init=True)
    sink = AimSink(make_config(every_steps=1), repo=repo_path)
    for step in (0, 1, 2, 3):
        sink.report(step, {"episode_return": float(step)})
    sink.close()

    assert _metric_steps(repo_path) == [0, 1, 2, 3]


def test_throttle_uses_elapsed_steps_not_modulus(tmp_path: Path) -> None:
    repo_path = str(tmp_path / "aim")
    Repo.from_path(repo_path, init=True)
    sink = AimSink(make_config(every_steps=1000), repo=repo_path)
    for step in (0, 512, 1024, 1536, 2048):
        sink.report(step, {"episode_return": float(step)})
    sink.close()

    assert _metric_steps(repo_path) == [0, 1024, 2048]


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
    assert len(metrics_path.read_text().splitlines()) == len(steps)
