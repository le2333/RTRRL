from pathlib import Path

from aim import Repo

from training_sdk.sinks.aim import AimSink, close_aim_run
from tests.test_reporter import make_config


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
