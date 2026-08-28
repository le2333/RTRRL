"""Two controllers started in the same second, which is how issue 53 happened.

On 2026-08-18 two `trainerctl` processes were started under the experiment
`rtrrl-issue37`. Both read the same UTC second, both generated the launch id
`20260818-151037`, and both therefore wrote to the same control prefix. The
second submission overwrote the first's round manifest, so the job named for
the D-RTRRL configuration ran the TD-out one, and the D-RTRRL controller died
hours later looking for a result.json nothing had written.

Two separate things have to hold for that to be impossible: a generated launch
id must not be a function of the clock alone, and taking a control prefix must
be a create that only one process can win.
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from fakes import FakeBatch, FakeLogs, FakeS3, split

from trainer_infra import experiment as experiment_module
from trainer_infra.batch import LAUNCH_CLAIM, BatchRoundExecutor, LaunchCollisionError
from trainer_infra.cli import main
from trainer_infra.experiment import ExperimentRunner, new_launch_id

pytestmark = pytest.mark.integration

# The second the two colliding launches were both started in.
MOMENT = datetime(2026, 8, 18, 15, 10, 37, tzinfo=UTC)
STAMP = "20260818-151037"
PREFIX = f"s3://bucket/trainer/rtrrl-issue37/{STAMP}/control"


@pytest.fixture
def one_second(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze the clock, so only what is not the clock can tell launches apart."""

    monkeypatch.setattr(
        experiment_module,
        "new_launch_id",
        lambda moment=None: new_launch_id(MOMENT),
    )


def test_two_controllers_starting_in_one_second_do_not_share_a_launch_id(
    one_second: None,
    experiment: dict[str, Any],
    catalog: dict[str, Any],
    tmp_path: Path,
) -> None:
    def launched() -> str:
        return ExperimentRunner(
            experiment=copy.deepcopy(experiment),
            catalog=catalog,
            database=tmp_path / "study.db",
        ).launch_id

    first, second = launched(), launched()

    assert first.startswith(f"{STAMP}-") and second.startswith(f"{STAMP}-")
    assert first != second


def test_two_launches_of_one_experiment_in_one_second_keep_their_own_manifests(
    one_second: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
    experiment: dict[str, Any],
    catalog: dict[str, Any],
    tmp_path: Path,
) -> None:
    """The acceptance the issue asks for, driven the way the operator drove it.

    Two `trainerctl run` invocations, one experiment file, one second, neither
    passing `--launch-id`. What is checked is what was lost: each launch's own
    round manifest still names the runs that launch submitted.
    """

    s3 = FakeS3()
    _batch_backend(monkeypatch, s3)
    paths = _files(experiment, catalog, tmp_path)

    launches = [_run(paths, capsys), _run(paths, capsys)]

    identities = [launch["launch_id"] for launch in launches]
    assert identities[0] != identities[1]
    manifests = {
        key: json.loads(body)
        for (_, key), body in s3.data.items()
        if key.endswith("round-000/job-000.json")
    }
    assert len(manifests) == 2, "one launch wrote over the other's manifest"
    for launch in launches:
        control = f"trainer/streamac-test/{launch['launch_id']}/control"
        assert manifests[f"{control}/round-000/job-000.json"] == {"runs": launch["runs"]}
    # And each took its prefix on the way in, so a third launch that did
    # somehow arrive at either id would be stopped rather than let in beside it.
    assert len({key for _, key in s3.data if key.endswith(LAUNCH_CLAIM)}) == 2


def test_a_generated_launch_refuses_a_control_prefix_that_is_already_taken() -> None:
    s3 = FakeS3()
    executor = _executor(s3)
    executor.claim(_claim(pid=4242), exclusive=True)

    with pytest.raises(LaunchCollisionError) as refusal:
        executor.claim(_claim(pid=9999), exclusive=True)

    # The message names who holds it: the operator's next move is to find out
    # whether that launch is still running.
    assert "pid=4242" in str(refusal.value)
    assert _held(s3)["pid"] == 4242, "the refused launch rewrote the claim anyway"


def test_a_launch_told_which_prefix_to_use_lands_in_the_one_that_exists() -> None:
    """`--launch-id` is how a killed launch's rounds are asked for again.

    An existing claim is not a collision there -- it is the prefix the operator
    named. So it is adopted rather than refused, and the launch that took it
    stays the one recorded as having taken it.
    """

    s3 = FakeS3()
    executor = _executor(s3)
    executor.claim(_claim(pid=4242), exclusive=True)

    executor.claim(_claim(pid=9999), exclusive=False)

    assert _held(s3)["pid"] == 4242


def _executor(s3: FakeS3) -> BatchRoundExecutor:
    return BatchRoundExecutor(
        s3=s3,
        batch=FakeBatch(s3),
        logs=FakeLogs(),
        exchange=PREFIX,
        job_name=f"rtrrl-issue37-{STAMP}",
        job_queue="dev-cpu-c7al-queue",
        job_definition="trainer-c7al-digest",
        timeout_seconds=5400,
        parallel_jobs=1,
        poll_seconds=0,
    )


def _claim(*, pid: int) -> dict[str, Any]:
    return {"launch_id": STAMP, "experiment": "rtrrl-issue37", "pid": pid}


def _held(s3: FakeS3) -> dict[str, Any]:
    return json.loads(s3.data[split(f"{PREFIX}/{LAUNCH_CLAIM}")])


def _batch_backend(monkeypatch: pytest.MonkeyPatch, s3: FakeS3) -> None:
    clients: dict[str, Any] = {"s3": s3, "batch": FakeBatch(s3), "logs": FakeLogs()}

    class Session:
        def client(self, name: str) -> Any:
            return clients[name]

    monkeypatch.setattr("trainer_infra.cli._batch_session", Session)


def _files(
    experiment: dict[str, Any], catalog: dict[str, Any], tmp_path: Path
) -> dict[str, Path]:
    """One experiment file and one study database, launched from twice."""

    experiment = copy.deepcopy(experiment)
    experiment["storage"] = "s3://bucket/trainer"
    experiment["compute"] = {"instance_type": "c7a.large", "timeout_minutes": 90}
    experiment["hpo"] = {
        "rounds": 1,
        "trials_per_round": 1,
        "startup_trials": 1,
        "seed": 7,
        "parallel_jobs": 1,
    }
    experiment["score"] = {
        "metric": "objective",
        "window_steps": [0, 10],
        "reduce": "last",
        "non_finite": "worst",
        "direction": "maximize",
    }
    catalog = copy.deepcopy(catalog)
    catalog["entries"]["stream_ac"]["metrics"] = ["objective"]
    paths = {
        "experiment": tmp_path / "experiment.yaml",
        "catalog": tmp_path / "catalog.json",
        "database": tmp_path / "study.db",
    }
    paths["experiment"].write_text(yaml.safe_dump(experiment), encoding="utf-8")
    paths["catalog"].write_text(json.dumps(catalog), encoding="utf-8")
    return paths


def _run(paths: dict[str, Path], capsys: Any) -> dict[str, Any]:
    """One `trainerctl run`: the launch id it took, and the runs it submitted."""

    assert (
        main(
            [
                "run",
                str(paths["experiment"]),
                "--backend",
                "batch",
                "--catalog",
                str(paths["catalog"]),
                "--database",
                str(paths["database"]),
                "--queues",
                "dev",
                "--poll-seconds",
                "0",
            ]
        )
        == 0
    )
    study = json.loads(capsys.readouterr().out)
    launch_id = study["launch_id"]
    trial = study["trials"][-1]["number"]
    control = f"s3://bucket/trainer/streamac-test/{launch_id}/control/round-000"
    return {
        "launch_id": launch_id,
        "runs": [f"{control}/trial-{trial:06d}-seed-000000.json"],
    }
