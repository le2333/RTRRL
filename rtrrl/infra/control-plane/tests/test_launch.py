import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from training_sdk import objects

from trainer_infra.experiment import load_experiment
from trainer_infra.launch import build_run_config, config_uri, create_launch
from trainer_infra.preflight import LaunchPlan
from tests.helpers import CATALOG, EXAMPLE, _document, make_plan

WHEN = datetime(2026, 7, 25, 5, 14, 0, tzinfo=UTC)
SOURCE = EXAMPLE
TRIAL_PARAMS = {"total_steps": 128, "learning_rate": 0.0003}


def _launch(tmp_path):
    source = tmp_path / "experiment.yaml"
    source.write_text(json.dumps(_document()), encoding="utf-8")
    experiment = load_experiment(source)
    plan = LaunchPlan(
        experiment=experiment,
        entry_name=experiment.entry,
        entry=next(iter(CATALOG.entries.values())),
        space={},
        digest="sha256:" + "a" * 64,
        queue="run-cpu-c7am-queue",
        job_definition="trainer-c7am-" + "a" * 64,
    )
    return create_launch(plan, tmp_path, source, WHEN)


def test_launch_id_is_a_utc_timestamp(s3_base: str, tmp_path: Path) -> None:
    launch = create_launch(make_plan(s3_base), tmp_path, SOURCE, WHEN)
    assert launch.launch_id == "20260725-051400"
    assert launch.prefix == f"{s3_base}/infra-acceptance/brax-ppo-smoke/20260725-051400"


def test_launch_metadata_is_written_to_archive_and_s3(s3_base: str, tmp_path: Path) -> None:
    source_bytes = SOURCE.read_bytes()
    launch = create_launch(make_plan(s3_base), tmp_path, SOURCE, WHEN)
    assert (launch.archive / "experiment.yaml").read_bytes() == source_bytes
    assert objects.get_bytes(f"{launch.prefix}/experiment.yaml") == source_bytes
    archived = json.loads((launch.archive / "launch.json").read_text())
    assert archived["digest"] == "sha256:" + "a" * 64
    assert archived["source_hash"] == "sha256:0"
    assert archived["contract"] == 2
    remote = json.loads(objects.get_bytes(f"{launch.prefix}/launch.json"))
    assert remote == archived
    assert json.loads(objects.get_bytes(f"{launch.prefix}/space.json"))["total_steps"] == [128]


@pytest.mark.parametrize("trial", [3, 7])
def test_run_config_uses_trial_params_verbatim(
    s3_base: str, tmp_path: Path, trial: int
) -> None:
    launch = create_launch(make_plan(s3_base), tmp_path, SOURCE, WHEN)
    config = build_run_config(launch, trial, TRIAL_PARAMS)
    assert config.run_id == f"brax-ppo-smoke-20260725-051400-t{trial}"
    assert config.params == TRIAL_PARAMS
    assert config.digest == "sha256:" + "a" * 64
    assert config.source_hash == "sha256:0"
    assert config.score.s3 == f"{launch.prefix}/trials/t{trial}/score.json"
    assert config.logging.rerun_s3 == f"{launch.prefix}/trials/t{trial}/episodes/"
    assert config_uri(launch, trial) == f"{launch.prefix}/trials/t{trial}/config.json"


def test_trial_s3_subtrees_are_disjoint(s3_base: str, tmp_path: Path) -> None:
    launch = create_launch(make_plan(s3_base), tmp_path, SOURCE, WHEN)
    configs = [
        build_run_config(launch, trial, TRIAL_PARAMS) for trial in (3, 7)
    ]
    score_prefixes = {config.score.s3.removesuffix("/score.json") for config in configs}
    rerun_prefixes = {
        config.logging.rerun_s3.removesuffix("/episodes/")
        for config in configs
        if config.logging.rerun_s3 is not None
    }
    assert score_prefixes == {
        f"{launch.prefix}/trials/t3",
        f"{launch.prefix}/trials/t7",
    }
    assert rerun_prefixes == score_prefixes
    assert len(score_prefixes) == 2


def test_run_config_disables_rerun_when_not_configured(
    s3_base: str, tmp_path: Path
) -> None:
    launch = create_launch(
        make_plan(s3_base, rerun_enabled=False), tmp_path, SOURCE, WHEN
    )
    config = build_run_config(launch, 7, TRIAL_PARAMS)
    assert config.logging.rerun_s3 is None
    assert config.score.s3 == f"{launch.prefix}/trials/t7/score.json"


def test_the_run_config_carries_the_environment_and_the_budget(tmp_path):
    launch = _launch(tmp_path)

    config = build_run_config(launch, trial=0, params={"learning_rate": 0.001})

    assert config.environment.id == "brax::hopper"
    assert config.environment.observed == (0, 1, 2, 3, 4)
    assert config.budget.total_steps == 2000


def test_the_archived_launch_records_both_sections(tmp_path):
    launch = _launch(tmp_path)

    archived = json.loads((launch.archive / "launch.json").read_text(encoding="utf-8"))

    assert archived["environment"]["observed"] == [0, 1, 2, 3, 4]
    assert archived["budget"]["epoch_steps"] == 1000
