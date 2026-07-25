import dataclasses
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import optuna
import pytest
from aim import Repo
from training_sdk import objects

from trainer_infra.backends.local import LocalBackend
from trainer_infra.launch import create_launch
from trainer_infra.loop import LaunchFailed, run_launch
from tests.conftest import AimServer
from tests.helpers import EXAMPLE, make_plan


def plan_using(s3_base: str, aim_server: AimServer):
    plan = make_plan(s3_base)
    experiment = plan.experiment.model_copy(
        update={
            "logging": plan.experiment.logging.model_copy(update={"aim": aim_server.uri})
        }
    )
    return dataclasses.replace(plan, experiment=experiment)


def test_two_round_study_completes_and_reports(
    s3_base: str, tmp_path: Path, aim_endpoint: AimServer, acceptance_catalog: Path
) -> None:
    launch = create_launch(
        plan_using(s3_base, aim_endpoint), tmp_path / "archive", EXAMPLE, datetime.now(UTC)
    )
    backend = LocalBackend(tmp_path / "jobs", acceptance_catalog)

    report = run_launch(launch, backend)

    assert report.status == "succeeded"
    assert len(report.trials) == 4  # 2 rounds x 2 trials per round
    assert report.best is not None and report.best.value is not None
    trial_values = [record.value for record in report.trials]
    assert report.best.value == max(trial_values)
    for record in report.trials:
        assert objects.exists(f"{launch.prefix}/trials/t{record.trial}/score.json")
        assert record.job_id is not None
    archived = json.loads((launch.archive / "report.json").read_text())
    assert archived["status"] == "succeeded"
    assert archived["best"]["trial"] == report.best.trial

    study = optuna.load_study(
        study_name=f"{launch.plan.experiment.name}-{launch.launch_id}",
        storage=f"sqlite:///{launch.archive / 'study.db'}",
    )
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    assert len(completed) == 4
    assert all(trial.value is not None for trial in completed)
    assert objects.exists(f"{launch.prefix}/rounds/round-001/job-0.json")

    runs = list(Repo.from_path(aim_endpoint.path).iter_runs())
    assert {run.name for run in runs} == {
        f"brax-ppo-smoke-{launch.launch_id}-t{index}" for index in range(4)
    }


def _is_sleep_process(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        return False
    return "sleep" in cmdline and "600" in cmdline


def test_failing_run_stops_the_launch_and_prints_the_log(
    s3_base: str,
    tmp_path: Path,
    aim_endpoint: AimServer,
    failing_with_long_sibling_catalog: Path,
) -> None:
    launch = create_launch(
        plan_using(s3_base, aim_endpoint), tmp_path / "archive", EXAMPLE, datetime.now(UTC)
    )
    backend = LocalBackend(tmp_path / "jobs", failing_with_long_sibling_catalog)
    printed: list[str] = []
    pid_file = tmp_path / "long-sibling.pid"
    started = time.monotonic()

    with pytest.raises(LaunchFailed):
        run_launch(launch, backend, printer=printed.append)

    elapsed = time.monotonic() - started
    assert elapsed < 15.0

    assert any("worker failed" in line for line in printed)
    archived = json.loads((launch.archive / "report.json").read_text())
    assert archived["status"] == "failed"
    assert archived["trials"] == []
    assert archived["failure"] is not None
    assert not objects.exists(f"{launch.prefix}/rounds/round-001/job-0.json")
    if pid_file.exists():
        assert not _is_sleep_process(int(pid_file.read_text(encoding="utf-8")))
