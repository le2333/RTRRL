from pathlib import Path

import optuna
import pytest
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution

from trainer_infra.study import ask_round, check_sampler, create_study, tell_value

optuna.logging.set_verbosity(optuna.logging.WARNING)

DISTRIBUTIONS = {
    "total_steps": CategoricalDistribution(choices=[128]),
    "learning_rate": FloatDistribution(low=1e-4, high=1e-3, log=True),
}

GRID_DISTRIBUTIONS = {
    "total_steps": CategoricalDistribution(choices=[128, 256]),
    "learning_rate": CategoricalDistribution(choices=[1e-4, 1e-3]),
}


def make(tmp_path: Path) -> optuna.Study:
    return create_study(
        "sweep-20260725-000000",
        tmp_path / "study.db",
        sampler="tpe",
        direction="maximize",
        user_attrs={"launch_id": "20260725-000000", "digest": "sha256:0"},
        space=DISTRIBUTIONS,
        round_size=1,
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


def test_tpe_stops_sampling_at_random_after_one_round(tmp_path: Path) -> None:
    """The round is what TPE could not have modelled, so it is what it spends.

    Optuna's own default is ten trials, which a launch of five rounds of four
    would spend half its budget on. The first round has no results to fit to and
    every round after it does.
    """

    study = create_study(
        "sweep-startup",
        tmp_path / "startup.db",
        sampler="tpe",
        direction="maximize",
        user_attrs={},
        space=DISTRIBUTIONS,
        round_size=4,
    )
    sampler = study.sampler
    assert isinstance(sampler, optuna.samplers.TPESampler)
    assert sampler._n_startup_trials == 4


def test_two_searches_given_one_seed_open_with_the_same_questions(
    tmp_path: Path,
) -> None:
    """A comparison of two searches is only about what differs between them.

    Two entries searched over one space are compared by what each one's best
    point scores, and an unseeded sampler hands them different points to start
    from, so part of the difference would be which points each happened to draw.
    Seeded, the round neither of them could have modelled is the same round.
    """

    def opening_round(name: str, seed: int | None) -> list[dict[str, float]]:
        study = create_study(
            name,
            tmp_path / f"{name}.db",
            sampler="tpe",
            direction="maximize",
            user_attrs={},
            space=DISTRIBUTIONS,
            round_size=4,
            seed=seed,
        )
        return [trial.params for trial in ask_round(study, DISTRIBUTIONS, 4)]

    assert opening_round("seeded-one", 7) == opening_round("seeded-two", 7)
    assert opening_round("seeded-one-again", 7) != opening_round("other-seed", 8)


def test_unknown_sampler_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported sampler 'cma'"):
        create_study(
            "s",
            tmp_path / "s.db",
            sampler="cma",
            direction="maximize",
            user_attrs={},
            space=DISTRIBUTIONS,
        round_size=1,
        )
    with pytest.raises(ValueError, match="tpe, random, grid"):
        check_sampler("cma", DISTRIBUTIONS)


def test_ask_round_samples_distinct_values_and_matches_study_numbers(
    tmp_path: Path,
) -> None:
    study = create_study(
        "sweep-distinct",
        tmp_path / "distinct.db",
        sampler="random",
        direction="maximize",
        user_attrs={},
        space=DISTRIBUTIONS,
        round_size=1,
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
        space=DISTRIBUTIONS,
        round_size=1,
    )
    reopened = optuna.load_study(
        study_name="sweep-attrs",
        storage=f"sqlite:///{tmp_path / 'attrs.db'}",
    )
    for key, value in attrs.items():
        assert reopened.user_attrs[key] == value


def test_grid_sampler_with_fixed_lists_can_ask_for_a_trial(tmp_path: Path) -> None:
    study = create_study(
        "sweep-grid",
        tmp_path / "grid.db",
        sampler="grid",
        direction="maximize",
        user_attrs={},
        space=GRID_DISTRIBUTIONS,
        round_size=1,
    )
    trials = ask_round(study, GRID_DISTRIBUTIONS, 1)
    assert len(trials) == 1
    assert trials[0].params["total_steps"] in (128, 256)
    assert trials[0].params["learning_rate"] in (1e-4, 1e-3)


SINGLE_POINT = {
    "total_steps": CategoricalDistribution(choices=[128]),
    "seed": CategoricalDistribution(choices=[0]),
}


def _grid_study(tmp_path: Path, space: dict[str, CategoricalDistribution]):
    return create_study(
        "sweep-grid",
        tmp_path / "grid.db",
        sampler="grid",
        direction="maximize",
        user_attrs={},
        space=space,
        round_size=1,
    )


def test_the_last_point_of_a_grid_can_be_told(tmp_path: Path) -> None:
    # An exhausted GridSampler ends the search by calling Study.stop(), which
    # optuna refuses outside its own optimize(). A pinned experiment is a grid
    # of one, so without this every such run dies on its only result.
    study = _grid_study(tmp_path, SINGLE_POINT)
    (trial,) = ask_round(study, SINGLE_POINT, 1)
    assert tell_value(study, trial, 1.0) is True
    assert study.best_value == 1.0


def test_a_grid_with_points_left_does_not_ask_to_stop(tmp_path: Path) -> None:
    study = _grid_study(tmp_path, GRID_DISTRIBUTIONS)
    (trial,) = ask_round(study, GRID_DISTRIBUTIONS, 1)
    assert tell_value(study, trial, 1.0) is False


def test_a_continuous_search_never_asks_to_stop(tmp_path: Path) -> None:
    study = make(tmp_path)
    (trial,) = ask_round(study, DISTRIBUTIONS, 1)
    assert tell_value(study, trial, 1.0) is False


def test_grid_sampler_rejects_continuous_parameters() -> None:
    space = {
        "learning_rate": FloatDistribution(low=1e-4, high=1e-3, log=True),
        "seed": IntDistribution(low=0, high=1000),
    }
    with pytest.raises(ValueError, match="learning_rate") as error:
        check_sampler("grid", space)
    message = str(error.value)
    assert "seed" in message
    assert "learning_rate" in message


def test_grid_sampler_rejects_continuous_parameters_at_study_creation(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        create_study(
            "s",
            tmp_path / "grid.db",
            sampler="grid",
            direction="maximize",
            user_attrs={},
            space=DISTRIBUTIONS,
        round_size=1,
        )
