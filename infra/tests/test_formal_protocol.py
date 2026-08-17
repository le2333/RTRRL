"""Seeds as a declared dimension, and what makes a launch reportable.

A seed is not a hyperparameter, so it is listed rather than searched, and every
configuration is run on every seed the file names. A launch that reports a
number rather than choosing one says where its configuration came from, and
that claim is what the refusals here are about.
"""

from pathlib import Path
from typing import Any

import pytest

from trainer_infra import ExperimentError, ExperimentRunner
from trainer_infra.scoring import ScoreSpec

LAUNCH = "20260817-120000"


def runner(experiment: Any, catalog: Any, tmp_path: Path) -> ExperimentRunner:
    return ExperimentRunner(
        experiment=experiment,
        catalog=catalog,
        database=tmp_path / "study.db",
        launch_id=LAUNCH,
    )


def frozen(experiment: Any) -> Any:
    """The experiment as a formal launch: one configuration, fresh seeds."""

    experiment["environment"]["seeds"] = [100, 101, 102]
    experiment["space"]["gamma"] = [0.9]
    experiment["hpo"]["trials_per_round"] = 1
    experiment["selection"] = {
        "study": "stream-ac-test",
        "trial": 3,
        "tuning_seeds": [0],
    }
    return experiment


# ------------------------------------------------------- the seeds are listed
def test_every_configuration_runs_on_every_declared_seed(
    experiment: Any, catalog: Any, tmp_path: Path
) -> None:
    experiment["environment"]["seeds"] = [0, 7]

    configurations = runner(experiment, catalog, tmp_path).next_round()

    assert [
        (run["identity"]["trial"], run["identity"]["seed"]) for run in configurations
    ] == [(0, 0), (0, 7), (1, 0), (1, 7)]
    # One configuration, two runs: the parameters are the same dictionary and
    # only the seed differs.
    first, second = configurations[:2]
    assert first["algorithm"] == second["algorithm"]
    assert (first["training"]["seed"], second["training"]["seed"]) == (0, 7)


def test_a_run_is_named_and_stored_by_its_configuration_and_its_seed(
    experiment: Any, catalog: Any, tmp_path: Path
) -> None:
    experiment["environment"]["seeds"] = [0, 7]

    configurations = runner(experiment, catalog, tmp_path).next_round()

    names = [run["identity"]["run_id"] for run in configurations[:2]]
    assert names == [
        f"stream-ac-test-{LAUNCH}-t0-s0",
        f"stream-ac-test-{LAUNCH}-t0-s7",
    ]
    assert len({run["artifacts"]["root"] for run in configurations}) == 4


def test_the_seed_never_reaches_the_graph(
    experiment: Any, catalog: Any, tmp_path: Path
) -> None:
    """It is a budget field: what it seeds is the runtime, not a component."""

    configuration = runner(experiment, catalog, tmp_path).next_round()[0]

    assert "seeds" not in configuration["algorithm"]["environment"]
    assert "seed" not in configuration["algorithm"]["environment"]
    assert "seed" not in configuration["algorithm"]["parameters"]


@pytest.mark.parametrize(
    "seeds, complaint",
    [
        ([], "at least one seed"),
        ([0, 0], "repeats a seed"),
        ([-1], "must not be negative"),
        (0, "must be a list"),
    ],
)
def test_a_seed_list_that_says_nothing_usable_starts_nothing(
    experiment: Any, catalog: Any, tmp_path: Path, seeds: Any, complaint: str
) -> None:
    experiment["environment"]["seeds"] = seeds

    with pytest.raises(ExperimentError, match=complaint):
        runner(experiment, catalog, tmp_path)


def test_the_optimizer_hears_the_mean_and_the_launch_keeps_the_seeds(
    experiment: Any, catalog: Any, tmp_path: Path
) -> None:
    """One seed's score reaches the study unchanged; several are summarised.

    The mean is all a study can store, and it is not what a result table
    reports, so the per-seed scores stay on the launch.
    """

    experiment["environment"]["seeds"] = [0, 7]
    experiment["hpo"]["rounds"] = 1
    values = {(0, 0): 1.0, (0, 7): 3.0, (1, 0): 10.0, (1, 7): 20.0}

    def execute(configurations: tuple[dict, ...], score: ScoreSpec) -> tuple[dict, ...]:
        del score
        return tuple(
            {
                "trial": run["identity"]["trial"],
                "seed": run["identity"]["seed"],
                "value": values[(run["identity"]["trial"], run["identity"]["seed"])],
            }
            for run in configurations
        )

    launch = runner(experiment, catalog, tmp_path)
    study = launch.run(execute)

    assert [trial.value for trial in study.trials] == [2.0, 15.0]
    assert launch.seed_scores == {0: {0: 1.0, 7: 3.0}, 1: {0: 10.0, 7: 20.0}}


def test_a_round_that_skips_a_seed_fails_the_trial(
    experiment: Any, catalog: Any, tmp_path: Path
) -> None:
    experiment["environment"]["seeds"] = [0, 7]

    def execute(configurations: tuple[dict, ...], score: ScoreSpec) -> tuple[dict, ...]:
        del score
        return tuple(
            {
                "trial": run["identity"]["trial"],
                "seed": run["identity"]["seed"],
                "value": 1.0,
            }
            for run in configurations
            if run["identity"]["seed"] == 0
        )

    with pytest.raises(ExperimentError, match="returned seeds"):
        runner(experiment, catalog, tmp_path).run(execute)


# ---------------------------------------------------- what makes it reportable
def test_a_launch_is_tuning_unless_it_says_where_its_choice_came_from(
    experiment: Any, catalog: Any, tmp_path: Path
) -> None:
    tuning = runner(experiment, catalog, tmp_path)
    formal = runner(frozen(dict(experiment)), catalog, tmp_path / "formal")

    assert tuning.role == "tuning"
    assert tuning.next_round()[0]["identity"]["role"] == "tuning"
    assert formal.role == "formal"
    assert formal.next_round()[0]["identity"]["role"] == "formal"


def test_a_formal_seed_that_tuned_the_configuration_starts_nothing(
    experiment: Any, catalog: Any, tmp_path: Path
) -> None:
    """Fresh is the whole of what makes a formal seed a formal seed."""

    formal = frozen(experiment)
    formal["environment"]["seeds"] = [0, 100, 101]

    with pytest.raises(ExperimentError, match=r"formal seeds \[0\] were already used"):
        runner(formal, catalog, tmp_path)


def test_a_formal_launch_that_is_still_searching_starts_nothing(
    experiment: Any, catalog: Any, tmp_path: Path
) -> None:
    """It reports a configuration, so there has to be exactly one of them."""

    formal = frozen(experiment)
    formal["space"]["gamma"] = [0.9, 0.95]

    with pytest.raises(ExperimentError, match="still offer more than one value"):
        runner(formal, catalog, tmp_path)


def test_a_formal_launch_scored_on_training_return_starts_nothing(
    experiment: Any, catalog: Any, tmp_path: Path
) -> None:
    formal = frozen(experiment)
    formal["score"]["metric"] = "train/episode/return_per_step"
    catalog["entries"]["stream_ac"]["metrics"] = [
        "eval/episode/return_per_step",
        "train/episode/return_per_step",
    ]

    with pytest.raises(ExperimentError, match="training return is diagnostic"):
        runner(formal, catalog, tmp_path)


def test_a_tuning_launch_may_still_be_scored_on_whatever_it_likes(
    experiment: Any, catalog: Any, tmp_path: Path
) -> None:
    """The refusal is about what may be reported, not about what may be tried."""

    experiment["score"]["metric"] = "train/episode/return_per_step"
    catalog["entries"]["stream_ac"]["metrics"] = [
        "eval/episode/return_per_step",
        "train/episode/return_per_step",
    ]

    assert runner(experiment, catalog, tmp_path).role == "tuning"


def test_a_selection_block_missing_its_provenance_starts_nothing(
    experiment: Any, catalog: Any, tmp_path: Path
) -> None:
    formal = frozen(experiment)
    del formal["selection"]["tuning_seeds"]

    with pytest.raises(ExperimentError, match=r"does not say \['tuning_seeds'\]"):
        runner(formal, catalog, tmp_path)


def test_the_study_archives_the_seeds_the_launch_ran_on(
    experiment: Any, catalog: Any, tmp_path: Path
) -> None:
    """Read off the study afterwards, the launch can still say what it was."""

    formal = frozen(experiment)
    launch = runner(formal, catalog, tmp_path)
    launch.next_round()

    attributes = launch.hpo._open().user_attrs

    assert attributes["role"] == "formal"
    assert attributes["seeds"] == [100, 101, 102]
    assert attributes["evaluation_seed"] == 1000
    assert attributes["selection"]["tuning_seeds"] == [0]
