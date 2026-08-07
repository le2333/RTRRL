import optuna
import pytest

from trainer_infra import sample_parameters
from trainer_infra.adapter import SpaceError, resolve_parameter_ranges


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


def test_the_resolved_space_keeps_the_shape_the_image_declared() -> None:
    """A group stays a group: it is the condition under which its leaves exist."""

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

    assert ranges == {"actor": {"kind": {"type": "choice", "values": ["rtu"]}}}
    assert sample_parameters(trial, ranges) == {"actor.kind": "rtu"}


def test_a_pin_for_an_undeclared_name_is_refused_rather_than_ignored() -> None:
    declared = {
        "gamma": {
            "valid": {"type": "float", "low": 0.0, "high": 1.0},
            "search": {"type": "choice", "values": [0.99]},
        }
    }

    with pytest.raises(SpaceError, match="gammma"):
        resolve_parameter_ranges(declared, {"gammma": [0.9]})


def test_a_pin_nested_under_a_group_is_named_by_where_it_was_written() -> None:
    declared = {
        "actor": {
            "kind": {
                "valid": {"type": "choice", "values": ["sgd"]},
                "search": {"type": "choice", "values": ["sgd"]},
            }
        }
    }

    with pytest.raises(SpaceError, match=r"actor\.knid"):
        resolve_parameter_ranges(declared, {"actor": {"knid": ["sgd"]}})
