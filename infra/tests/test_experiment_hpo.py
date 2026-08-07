from pathlib import Path

from trainer_infra import HPO, ExperimentRunner, SampledTrial, resolve_parameter_ranges


def test_resolved_experiment_ranges_feed_hpo_configuration_groups(
    tmp_path: Path,
) -> None:
    catalog = {
        "algorithms": {
            "stream_ac": {
                "parameters": {
                    "gamma": {
                        "valid": {"type": "float", "low": 0.0, "high": 1.0},
                        "search": {"type": "choice", "values": [0.99]},
                    },
                    "lr": {
                        "valid": {"type": "float", "low": 0.0, "high": None},
                        "search": {"type": "choice", "values": [1.0]},
                    },
                }
            }
        }
    }
    experiment = {
        "name": "stream-ac-test",
        "algorithm": "stream_ac",
        "hpo": {
            "rounds": 1,
            "trials_per_round": 2,
            "startup_trials": 2,
            "seed": 7,
        },
        "objective": {"metric": "episodic_return", "direction": "maximize"},
        "parameters": {"gamma": [0.9, 0.95, 0.99], "lr": [1.0]},
    }
    groups: list[tuple[SampledTrial, ...]] = []

    declared = catalog["algorithms"][experiment["algorithm"]]["parameters"]
    parameters = resolve_parameter_ranges(declared, experiment["parameters"])
    hpo_settings = experiment["hpo"]
    hpo = HPO(
        name=experiment["name"],
        database=tmp_path / "study.db",
        direction=experiment["objective"]["direction"],
        rounds=hpo_settings["rounds"],
        trials_per_round=hpo_settings["trials_per_round"],
        startup_trials=hpo_settings["startup_trials"],
        seed=hpo_settings["seed"],
        parameters=parameters,
    )

    def run_group(group: tuple[SampledTrial, ...]) -> list[float]:
        groups.append(group)
        return [float(trial.number) for trial in group]

    study = hpo.run(run_group)

    assert len(groups) == 1
    assert [trial.number for trial in groups[0]] == [0, 1]
    assert all(trial.parameters["gamma"] in [0.9, 0.95, 0.99] for trial in groups[0])
    assert all(trial.parameters["lr"] == 1.0 for trial in groups[0])
    assert [trial.value for trial in study.trials] == [0.0, 1.0]


def test_runner_completes_two_rounds_from_out_of_order_worker_results(
    tmp_path: Path,
) -> None:
    catalog = {
        "algorithms": {
            "stream_ac": {
                "parameters": {
                    "gamma": {
                        "valid": {"type": "float", "low": 0.0, "high": 1.0},
                        "search": {"type": "choice", "values": [0.9]},
                    },
                }
            }
        }
    }
    experiment = {
        "name": "stream-ac-test",
        "algorithm": "stream_ac",
        "hpo": {
            "rounds": 2,
            "trials_per_round": 2,
            "startup_trials": 2,
            "seed": 7,
        },
        "objective": {
            "metric": "episodic_return",
            "reduction": "last",
            "direction": "maximize",
        },
        "loggers": {"local": {"directory": "logs"}},
        "parameters": {"gamma": [0.9]},
        "run": {
            "environment": {"id": "Pendulum-v1", "seed": 0},
            "training": {"total_steps": 200},
            "debug": False,
            "overshooting_info": False,
            "render": False,
        },
    }
    configuration_rounds: list[tuple[dict[str, object], ...]] = []

    def execute_round(
        configurations: tuple[dict[str, object], ...],
    ) -> tuple[dict[str, float | int], ...]:
        configuration_rounds.append(configurations)
        return tuple(
            {"trial": configuration["trial"], "value": float(configuration["trial"])}
            for configuration in reversed(configurations)
        )

    study = ExperimentRunner(
        experiment=experiment,
        catalog=catalog,
        database=tmp_path / "study.db",
    ).run(execute_round)

    assert [[configuration["trial"] for configuration in round_] for round_ in configuration_rounds] == [
        [0, 1],
        [2, 3],
    ]
    assert all(
        configuration["experiment"] == "stream-ac-test"
        for round_ in configuration_rounds
        for configuration in round_
    )
    assert [trial.value for trial in study.trials] == [0.0, 1.0, 2.0, 3.0]
