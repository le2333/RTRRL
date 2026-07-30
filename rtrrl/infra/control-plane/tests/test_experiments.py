from pathlib import Path

import pytest

from trainer_infra.experiment import load_experiment

EXPERIMENTS = sorted(Path("../../../experiments").glob("*.yaml"))


def test_experiments_exist() -> None:
    assert EXPERIMENTS, "experiments/ has no experiment files"


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda path: path.name)
def test_experiment_loads(path: Path) -> None:
    load_experiment(path)
