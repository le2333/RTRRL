from pathlib import Path
from typing import Any

import pytest
from conftest import DIGEST

from trainer_infra import ExperimentError, ExperimentRunner, SpaceError

LAUNCH = "20260807-120000"


def runner(experiment: Any, catalog: Any, tmp_path: Path) -> ExperimentRunner:
    return ExperimentRunner(
        experiment=experiment,
        catalog=catalog,
        database=tmp_path / "study.db",
        launch_id=LAUNCH,
    )


def test_a_configuration_carries_every_field_the_worker_validates(
    experiment: Any, catalog: Any, tmp_path: Path
) -> None:
    first, second = runner(experiment, catalog, tmp_path).next_round()

    assert set(first) == {
        "contract",
        "run_id",
        "experiment",
        "launch_id",
        "trial",
        "entry",
        "digest",
        "environment",
        "training",
        "evaluation",
        "params",
        "logging",
        "score",
    }
    assert first["contract"] == 7
    assert first["experiment"] == "streamac-test"
    assert first["entry"] == "stream_ac"
    assert first["digest"] == DIGEST
    assert (first["trial"], second["trial"]) == (0, 1)
    assert first["run_id"] == f"stream-ac-test-{LAUNCH}-t0"
    assert first["launch_id"] == LAUNCH


def test_the_five_blocks_are_passed_through_untouched(
    experiment: Any, catalog: Any, tmp_path: Path
) -> None:
    configuration = runner(experiment, catalog, tmp_path).next_round()[0]

    assert configuration["environment"] == experiment["environment"]
    assert configuration["training"] == experiment["training"]
    assert configuration["evaluation"] == experiment["evaluation"]


def test_each_run_is_told_where_its_own_result_goes(
    experiment: Any, catalog: Any, tmp_path: Path
) -> None:
    """The experiment says a storage root; which run writes where is ours."""

    first, second = runner(experiment, catalog, tmp_path).next_round()
    root = f"s3://artifacts/trainer/streamac-test/{LAUNCH}"

    assert first["score"]["s3"] == f"{root}/stream-ac-test-{LAUNCH}-t0/score.json"
    assert second["score"]["s3"] == f"{root}/stream-ac-test-{LAUNCH}-t1/score.json"
    assert first["logging"]["rerun_s3"] == f"{root}/stream-ac-test-{LAUNCH}-t0/rerun"
    # What to optimise stays the experiment's word.
    assert first["score"]["direction"] == "maximize"
    assert first["score"]["metric"] == "eval/episode/return_per_step"


def test_rerun_gets_no_destination_when_it_is_off(
    experiment: Any, catalog: Any, tmp_path: Path
) -> None:
    experiment["logging"]["enable_rerun"] = False

    configuration = runner(experiment, catalog, tmp_path).next_round()[0]

    assert "rerun_s3" not in configuration["logging"]


def test_the_sampled_parameters_honour_what_the_experiment_pinned(
    experiment: Any, catalog: Any, tmp_path: Path
) -> None:
    for configuration in runner(experiment, catalog, tmp_path).next_round():
        assert set(configuration["params"]) == {"gamma", "backbone.hidden_dim"}
        assert configuration["params"]["gamma"] in (0.9, 0.95)
        assert configuration["params"]["backbone.hidden_dim"] == 32


def test_a_file_missing_a_field_the_worker_needs_starts_nothing(
    experiment: Any, catalog: Any, tmp_path: Path
) -> None:
    del experiment["training"]["epoch_steps"]
    del experiment["storage"]

    with pytest.raises(ExperimentError) as raised:
        runner(experiment, catalog, tmp_path)

    assert "storage" in str(raised.value)
    assert "training.epoch_steps" in str(raised.value)


def test_a_pin_the_image_declares_no_parameter_for_starts_nothing(
    experiment: Any, catalog: Any, tmp_path: Path
) -> None:
    experiment["space"]["gamma_"] = [0.9]

    with pytest.raises(SpaceError, match="gamma_"):
        runner(experiment, catalog, tmp_path)


def test_an_image_on_a_movable_tag_starts_nothing(
    experiment: Any, catalog: Any, tmp_path: Path
) -> None:
    experiment["image"] = "registry.example/trainer:latest"

    with pytest.raises(ExperimentError, match="not pinned to a digest"):
        runner(experiment, catalog, tmp_path)


def test_an_entry_the_image_does_not_declare_starts_nothing(
    experiment: Any, catalog: Any, tmp_path: Path
) -> None:
    experiment["entry"] = "nowhere"

    with pytest.raises(ExperimentError, match="nowhere"):
        runner(experiment, catalog, tmp_path)


def test_two_rounds_complete_from_out_of_order_worker_results(
    experiment: Any, catalog: Any, tmp_path: Path
) -> None:
    experiment["hpo"]["rounds"] = 2
    rounds: list[tuple[dict[str, object], ...]] = []

    def execute(configurations: tuple[dict[str, object], ...]) -> tuple[dict, ...]:
        rounds.append(configurations)
        return tuple(
            {"trial": configuration["trial"], "value": float(configuration["trial"])}
            for configuration in reversed(configurations)
        )

    study = runner(experiment, catalog, tmp_path).run(execute)

    assert [[c["trial"] for c in round_] for round_ in rounds] == [[0, 1], [2, 3]]
    assert [trial.value for trial in study.trials] == [0.0, 1.0, 2.0, 3.0]
