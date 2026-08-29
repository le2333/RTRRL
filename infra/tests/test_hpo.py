from pathlib import Path

import optuna
import pytest

from trainer_infra import HPO, SampledTrial, StudyError

NO_PARAMETERS = {}


def test_ask_generates_a_round_without_metric_feedback(tmp_path: Path) -> None:
    database = tmp_path / "study.db"
    hpo = HPO(
        name="streamac",
        database=database,
        direction="maximize",
        rounds=2,
        trials_per_round=2,
        startup_trials=2,
        seed=7,
        parameters={
            "gamma": {"type": "choice", "values": [0.9, 0.95]},
        },
    )

    trials = hpo.ask()

    assert [trial.number for trial in trials] == [0, 1]
    assert all(trial.parameters["gamma"] in [0.9, 0.95] for trial in trials)
    persisted = optuna.load_study(
        study_name="streamac",
        storage=f"sqlite:///{database}",
    )
    assert all(trial.state == optuna.trial.TrialState.RUNNING for trial in persisted.trials)


def test_run_hpo_records_each_round_before_starting_the_next(tmp_path: Path) -> None:
    database = tmp_path / "nested" / "study.db"
    round_trials: list[tuple[int, ...]] = []
    completed_before_round: list[int] = []

    sampled_parameters: list[tuple[dict[str, object], ...]] = []

    def evaluate_round(trials: tuple[SampledTrial, ...]) -> list[float]:
        persisted = optuna.load_study(
            study_name="streamac",
            storage=f"sqlite:///{database}",
        )
        completed_before_round.append(
            sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in persisted.trials)
        )
        round_trials.append(tuple(trial.number for trial in trials))
        sampled_parameters.append(tuple(trial.parameters for trial in trials))
        return [float(trial.number) for trial in trials]

    hpo = HPO(
        name="streamac",
        database=database,
        direction="maximize",
        rounds=2,
        trials_per_round=2,
        startup_trials=2,
        seed=7,
        metadata={"experiment": "baseline"},
        parameters={"gamma": {"type": "choice", "values": [0.9]}},
    )
    study = hpo.run(evaluate_round)

    assert database.is_file()
    assert round_trials == [(0, 1), (2, 3)]
    assert completed_before_round == [0, 2]
    assert sampled_parameters == [
        ({"gamma": 0.9}, {"gamma": 0.9}),
        ({"gamma": 0.9}, {"gamma": 0.9}),
    ]
    assert [trial.value for trial in study.trials] == [0.0, 1.0, 2.0, 3.0]
    assert study.user_attrs == {"experiment": "baseline"}



def test_a_launch_that_stops_claiming_what_the_study_recorded_is_refused(
    tmp_path: Path,
) -> None:
    """The half of a re-stamp that hides: a key deleted rather than edited.

    Comparing what the file still says against what the study holds finds an
    edited value and misses a removed one -- there is nothing left on the file's
    side to disagree with. So the comparison starts from the record, and a
    launch that no longer claims a key the study stores is a different launch.

    Driven here rather than through an experiment file because every block that
    file can drop is archived even when empty, which is the other half of the
    same fix; this is the rule itself.
    """

    config = {
        "name": "streamac",
        "database": tmp_path / "study.db",
        "direction": "maximize",
        "rounds": 1,
        "trials_per_round": 1,
        "startup_trials": 1,
        "seed": 7,
        "parameters": {"gamma": {"type": "choice", "values": [0.9]}},
    }
    HPO(**config, metadata={"role": "formal", "selection": {"trial": 3}}).ask()

    with pytest.raises(StudyError, match=r"\['selection'\], which this launch does not claim"):
        HPO(**config, metadata={"role": "formal"}).ask()


def test_a_launch_that_claims_no_provenance_overwrites_none(tmp_path: Path) -> None:
    """A bare optimizer is a library object, not a second author of the record."""

    config = {
        "name": "streamac",
        "database": tmp_path / "study.db",
        "direction": "maximize",
        "rounds": 1,
        "trials_per_round": 1,
        "startup_trials": 1,
        "seed": 7,
        "parameters": {"gamma": {"type": "choice", "values": [0.9]}},
    }
    HPO(**config, metadata={"role": "formal"}).ask()
    HPO(**config).ask()

    assert HPO(**config)._open().user_attrs == {"role": "formal"}


def test_running_names_the_trials_the_study_never_heard_back_about(tmp_path: Path) -> None:
    """A new controller has to find the open trials in the database, not in itself."""

    database = tmp_path / "study.db"
    config = {
        "name": "streamac",
        "database": database,
        "direction": "maximize",
        "rounds": 1,
        "trials_per_round": 2,
        "startup_trials": 2,
        "seed": 7,
        "parameters": {"gamma": {"type": "choice", "values": [0.9, 0.95]}},
    }
    asked = HPO(**config).ask()
    HPO(**config).tell(asked[:1], (1.0,))

    still_open = HPO(**config).running()

    assert [trial.number for trial in still_open] == [1]
    assert still_open[0].parameters == asked[1].parameters


def test_run_continues_an_existing_study(tmp_path: Path) -> None:
    database = tmp_path / "study.db"
    config = {
        "name": "streamac",
        "database": database,
        "direction": "maximize",
        "rounds": 1,
        "trials_per_round": 1,
        "startup_trials": 1,
        "parameters": NO_PARAMETERS,
    }

    HPO(**config).run(lambda trials: [1.0])
    study = HPO(**config).run(lambda trials: [2.0])

    assert [trial.number for trial in study.trials] == [0, 1]
    assert [trial.value for trial in study.trials] == [1.0, 2.0]


def test_a_failed_round_marks_every_asked_trial_failed_and_starts_no_next_round(
    tmp_path: Path,
) -> None:
    database = tmp_path / "study.db"
    calls = 0
    hpo = HPO(
        name="streamac",
        database=database,
        direction="maximize",
        rounds=2,
        trials_per_round=2,
        startup_trials=2,
        parameters=NO_PARAMETERS,
    )

    def fail_round(trials: tuple[SampledTrial, ...]) -> list[float]:
        nonlocal calls
        calls += 1
        raise RuntimeError("executor failed")

    with pytest.raises(RuntimeError, match="executor failed"):
        hpo.run(fail_round)

    persisted = optuna.load_study(
        study_name="streamac",
        storage=f"sqlite:///{database}",
    )
    assert calls == 1
    assert [trial.state for trial in persisted.trials] == [
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.FAIL,
    ]
