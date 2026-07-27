from pathlib import Path

import pytest
from pydantic import ValidationError

from trainer_infra.experiment import load_experiment
from tests.helpers import EXAMPLE, replace_once


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
    assert experiment.space["total_steps"].choices == (128,)


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
