import optuna

from trainer_infra import sample_parameters


def test_sample_parameters_samples_every_declared_range() -> None:
    ranges = {
        "gamma": {"type": "choice", "values": [0.9, 0.95, 0.99]},
        "actor.optim.lr": {
            "type": "float",
            "low": 1e-5,
            "high": 1e-3,
            "log": True,
        },
        "batch_size": {"type": "int", "low": 32, "high": 128, "step": 32},
        "meta_rl": {"type": "choice", "values": [False]},
        "backbone.kind": {"type": "choice", "values": ["rtu"]},
    }
    trial = optuna.trial.FixedTrial(
        {
            "gamma": 0.95,
            "actor.optim.lr": 1e-4,
            "batch_size": 64,
            "meta_rl": False,
            "backbone.kind": "rtu",
        }
    )

    assert sample_parameters(trial, ranges) == {
        "gamma": 0.95,
        "actor.optim.lr": 1e-4,
        "batch_size": 64,
        "meta_rl": False,
        "backbone.kind": "rtu",
    }


def test_single_choice_is_still_declared_as_a_trial_parameter() -> None:
    study = optuna.create_study()
    trial = study.ask()

    sampled = sample_parameters(
        trial,
        {"backbone.kind": {"type": "choice", "values": ["rtu"]}},
    )

    assert sampled == {"backbone.kind": "rtu"}
    assert trial.params == {"backbone.kind": "rtu"}
