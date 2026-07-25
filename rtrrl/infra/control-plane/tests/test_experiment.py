from pathlib import Path

import pytest
from pydantic import ValidationError

from trainer_infra.experiment import load_experiment

EXAMPLE = Path("examples/experiment-acceptance.yaml")


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


def test_parallel_jobs_may_not_exceed_trials_per_round(tmp_path: Path) -> None:
    text = EXAMPLE.read_text().replace("parallel_jobs: 2", "parallel_jobs: 99")
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError, match="parallel_jobs"):
        load_experiment(path)
