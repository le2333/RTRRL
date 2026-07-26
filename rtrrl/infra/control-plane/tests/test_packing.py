import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from training_sdk import objects
from training_sdk.contract import RunConfig

from trainer_infra.launch import build_run_config, create_launch
from trainer_infra.packing import publish_round, split
from tests.helpers import EXAMPLE, make_plan

WHEN = datetime(2026, 7, 25, 5, 14, 0, tzinfo=UTC)


def test_remainder_goes_to_the_earliest_jobs() -> None:
    assert split(8, 3) == [3, 3, 2]
    assert split(8, 2) == [4, 4]
    assert split(2, 2) == [1, 1]


def test_split_even_division_differs_from_uneven() -> None:
    assert split(6, 3) == [2, 2, 2]
    assert split(7, 3) == [3, 2, 2]


def test_split_remainder_not_on_trailing_jobs() -> None:
    # Wrong implementation putting remainder on the last jobs would yield [2, 2, 3].
    assert split(7, 3) == [3, 2, 2]
    assert split(5, 3) == [2, 2, 1]


def test_split_rejects_more_jobs_than_trials() -> None:
    with pytest.raises(ValueError, match="jobs must be between one and the number of trials"):
        split(2, 3)


def test_configs_and_manifests_are_uploaded(s3_base: str, tmp_path: Path) -> None:
    launch = create_launch(make_plan(s3_base), tmp_path, EXAMPLE, WHEN)
    configs = [
        build_run_config(launch, trial, {"total_steps": 128, "learning_rate": 0.0003})
        for trial in range(3)
    ]
    plans = publish_round(launch, 0, configs, jobs=2)
    assert [plan.manifest_uri for plan in plans] == [
        f"{launch.prefix}/rounds/round-000/job-0.json",
        f"{launch.prefix}/rounds/round-000/job-1.json",
    ]
    assert [plan.trials for plan in plans] == [(0, 1), (2,)]
    first = json.loads(objects.get_bytes(plans[0].manifest_uri))
    second = json.loads(objects.get_bytes(plans[1].manifest_uri))
    assert len(first["runs"]) == 2 and len(second["runs"]) == 1
    for uri in first["runs"] + second["runs"]:
        assert json.loads(objects.get_bytes(uri))["contract"] == 2


def test_every_trial_appears_exactly_once_in_manifests(
    s3_base: str, tmp_path: Path
) -> None:
    launch = create_launch(make_plan(s3_base), tmp_path, EXAMPLE, WHEN)
    trial_count = 8
    configs = [
        build_run_config(launch, trial, {"total_steps": 128, "learning_rate": 0.0003})
        for trial in range(trial_count)
    ]
    plans = publish_round(launch, 1, configs, jobs=3)

    seen_trials: list[int] = []
    for plan in plans:
        manifest = json.loads(objects.get_bytes(plan.manifest_uri))
        assert set(manifest) == {"runs"}
        assert isinstance(manifest["runs"], list)
        for config_uri in manifest["runs"]:
            config = RunConfig.model_validate(json.loads(objects.get_bytes(config_uri)))
            seen_trials.append(config.trial)
        assert plan.trials == tuple(seen_trials[-len(plan.trials) :])

    assert sorted(seen_trials) == list(range(trial_count))
    assert len(seen_trials) == len(set(seen_trials))
