from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from trainer_infra.experiment import load_experiment
from tests.helpers import EXAMPLE, _document, replace_once


def _modified_example(tmp_path: Path, old: str, new: str) -> Path:
    text = replace_once(EXAMPLE.read_text(), old, new)
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_example_file_loads() -> None:
    experiment = load_experiment(EXAMPLE)
    assert experiment.experiment == "infra-acceptance"
    assert experiment.entry == "brax_ppo_acceptance"
    assert experiment.compute.instance_type == "c7a.medium"
    assert experiment.hpo.trials_per_round >= experiment.hpo.parallel_jobs
    assert experiment.training.total_steps == 128


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(EXAMPLE.read_text() + "\ngroups: {}\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="groups"):
        load_experiment(path)


def test_the_job_timeout_must_be_positive(tmp_path: Path) -> None:
    path = _modified_example(tmp_path, "timeout_minutes: 60", "timeout_minutes: 0")
    with pytest.raises(ValidationError, match="timeout_minutes must be positive"):
        load_experiment(path)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("rounds: 2", "rounds: 0"),
        ("trials_per_round: 2", "trials_per_round: 0"),
        ("parallel_jobs: 2", "parallel_jobs: 0"),
    ],
)
def test_hpo_counts_must_be_positive(tmp_path: Path, old: str, new: str) -> None:
    path = _modified_example(tmp_path, old, new)
    with pytest.raises(
        ValidationError,
        match="rounds, trials_per_round and parallel_jobs must be positive",
    ):
        load_experiment(path)


def test_parallel_jobs_may_not_exceed_trials_per_round(tmp_path: Path) -> None:
    path = _modified_example(tmp_path, "parallel_jobs: 2", "parallel_jobs: 99")
    with pytest.raises(ValidationError, match="parallel_jobs"):
        load_experiment(path)


def test_score_window_steps_must_be_ordered(tmp_path: Path) -> None:
    path = _modified_example(tmp_path, "window_steps: [0, 128]", "window_steps: [128, 0]")
    with pytest.raises(ValidationError, match="window_steps must be ordered"):
        load_experiment(path)


def test_an_experiment_carries_environment_training_and_evaluation(tmp_path):
    document = _document()
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    experiment = load_experiment(path)

    assert experiment.environment.seed == 0
    assert experiment.environment.observed == (0, 1, 2, 3, 4)
    assert experiment.training.total_steps == 2000
    assert experiment.evaluation.steps == 100


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seed", -1, "seed must not be negative"),
        ("observed", [], "observed must name at least one index"),
        ("observed", [0, 0, 1], "observed must not repeat an index"),
        ("observed", [-1, 0], "observed indices must not be negative"),
    ],
)
def test_an_environment_must_be_usable(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    document = _document()
    document["environment"][field] = value
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ValidationError, match=message):
        load_experiment(path)


def test_an_omitted_observed_field_means_fully_observed(tmp_path: Path) -> None:
    document = _document()
    document["environment"].pop("observed")
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    experiment = load_experiment(path)

    assert experiment.environment.observed is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("total_steps", 0, "total_steps must be positive"),
        ("epoch_steps", 0, "epoch_steps must be positive"),
        ("epoch_steps", 300, "total_steps 2000 is not whole epochs of 300"),
    ],
)
def test_training_must_describe_whole_positive_epochs(
    tmp_path: Path, field: str, value: int, message: str
) -> None:
    document = _document()
    document["training"][field] = value
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ValidationError, match=message):
        load_experiment(path)


def test_evaluation_steps_may_be_zero(tmp_path: Path) -> None:
    document = _document()
    document["evaluation"]["steps"] = 0
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    experiment = load_experiment(path)

    assert experiment.evaluation.steps == 0


@pytest.mark.parametrize(
    "reserved",
    [
        "environment",
        "env_mode",
        "env_backend",
        "observed",
        "seed",
        "num_envs",
        "total_steps",
        "epoch_steps",
        "eval_steps",
        "chunk_steps",
        "early_stop_patience",
        "eval_envs",
    ],
)
def test_a_space_may_not_name_non_algorithm_fields(
    tmp_path: Path, reserved: str
) -> None:
    document = _document()
    document["space"][reserved] = [2000]
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ValidationError) as raised:
        load_experiment(path)

    assert reserved in str(raised.value)


def test_epoch_steps_must_be_a_whole_number_of_environment_streams(
    tmp_path: Path,
) -> None:
    document = _document()
    document["training"]["num_envs"] = 3
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ValidationError, match="epoch_steps 1000"):
        load_experiment(path)
