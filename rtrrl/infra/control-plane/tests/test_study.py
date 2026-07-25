from pathlib import Path

import optuna
import pytest
from optuna.distributions import CategoricalDistribution, FloatDistribution

from trainer_infra.study import ask_round, create_study, tell_value

optuna.logging.set_verbosity(optuna.logging.WARNING)

DISTRIBUTIONS = {
    "total_steps": CategoricalDistribution(choices=[128]),
    "learning_rate": FloatDistribution(low=1e-4, high=1e-3, log=True),
}


def make(tmp_path: Path) -> optuna.Study:
    return create_study(
        "sweep-20260725-000000",
        tmp_path / "study.db",
        sampler="tpe",
        direction="maximize",
        user_attrs={"launch_id": "20260725-000000", "digest": "sha256:0"},
    )


def test_every_trial_carries_every_parameter(tmp_path: Path) -> None:
    study = make(tmp_path)
    trials = ask_round(study, DISTRIBUTIONS, 3)
    assert len(trials) == 3
    for trial in trials:
        assert set(trial.params) == {"total_steps", "learning_rate"}
        assert trial.params["total_steps"] == 128


def test_told_values_persist_in_the_sqlite_file(tmp_path: Path) -> None:
    study = make(tmp_path)
    for index, trial in enumerate(ask_round(study, DISTRIBUTIONS, 2)):
        tell_value(study, trial, float(index))
    reopened = optuna.load_study(
        study_name="sweep-20260725-000000",
        storage=f"sqlite:///{tmp_path / 'study.db'}",
    )
    assert sorted(t.value for t in reopened.trials) == [0.0, 1.0]
    assert reopened.user_attrs["digest"] == "sha256:0"


def test_unknown_sampler_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sampler"):
        create_study("s", tmp_path / "s.db", sampler="cma", direction="maximize",
                     user_attrs={})


def test_ask_round_samples_distinct_values_and_matches_study_numbers(
    tmp_path: Path,
) -> None:
    study = create_study(
        "sweep-distinct",
        tmp_path / "distinct.db",
        sampler="random",
        direction="maximize",
        user_attrs={},
    )
    trials = ask_round(study, DISTRIBUTIONS, 3)
    learning_rates = [trial.params["learning_rate"] for trial in trials]
    assert len(set(learning_rates)) == 3
    for trial in trials:
        assert trial.number == study.trials[trial.number].number
        assert study.trials[trial.number].params == trial.params


def test_user_attrs_survive_sqlite_round_trip(tmp_path: Path) -> None:
    attrs = {
        "experiment": "rtrrl",
        "name": "sweep-a",
        "launch_id": "20260725-120000",
        "entry": "default",
        "digest": "sha256:abc",
        "source_hash": "sha256:def",
    }
    create_study(
        "sweep-attrs",
        tmp_path / "attrs.db",
        sampler="tpe",
        direction="maximize",
        user_attrs=attrs,
    )
    reopened = optuna.load_study(
        study_name="sweep-attrs",
        storage=f"sqlite:///{tmp_path / 'attrs.db'}",
    )
    for key, value in attrs.items():
        assert reopened.user_attrs[key] == value


def test_grid_sampler_requires_search_space(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="grid sampler requires the search space"):
        create_study(
            "s",
            tmp_path / "grid.db",
            sampler="grid",
            direction="maximize",
            user_attrs={},
        )
