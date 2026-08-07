import optuna

from trainer_infra import sample_parameters
from trainer_infra.adapter import resolve_parameter_ranges


def test_resolve_parameter_ranges_applies_experiment_choice_overrides() -> None:
    declared = {
        "gamma": {
            "valid": {"type": "float", "low": 0.0, "high": 1.0, "log": False},
            "search": {"type": "choice", "values": [0.99]},
        },
        "lr": {
            "valid": {"type": "float", "low": 0.0, "high": None, "log": False},
            "search": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        },
    }
    ranges = resolve_parameter_ranges(declared, {"gamma": [0.9, 0.95, 0.99]})
    trial = optuna.trial.FixedTrial({"gamma": 0.95, "lr": 1e-4})

    assert ranges == {
        "gamma": {"type": "choice", "values": [0.9, 0.95, 0.99]},
        "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    }
    assert sample_parameters(trial, ranges) == {"gamma": 0.95, "lr": 1e-4}


def test_resolve_parameter_ranges_flattens_only_the_declared_tree() -> None:
    declared = {
        "actor": {
            "kind": {
                "valid": {"type": "choice", "values": ["mlp", "rtu"]},
                "search": {"type": "choice", "values": ["mlp"]},
            }
        }
    }
    ranges = resolve_parameter_ranges(declared, {"actor": {"kind": ["rtu"]}})
    trial = optuna.trial.FixedTrial({"actor.kind": "rtu"})

    assert ranges == {
        "actor.kind": {"type": "choice", "values": ["rtu"]},
    }
    assert sample_parameters(trial, ranges) == {"actor.kind": "rtu"}
