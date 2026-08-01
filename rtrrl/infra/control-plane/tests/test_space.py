from __future__ import annotations

import optuna
import pytest
from training_sdk.contract import ChoiceSpec, EntryDescriptor

from trainer_infra.space import (
    SpaceError,
    grid_distributions,
    resolve_parameters,
    sample_parameters,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_entry(parameters: dict) -> EntryDescriptor:
    return EntryDescriptor.model_validate(
        {
            "command": ["run"],
            "metrics": ["eval/episode/return"],
            "parameters": parameters,
        }
    )


def learning_rate() -> dict:
    return {
        "kind": "param",
        "value_type": "float",
        "valid": {"type": "float", "low": 1e-9, "high": 10.0},
        "search": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "placeholder": 0.001,
    }


def single_point(placeholder: bool = True) -> dict:
    return {
        "kind": "param",
        "value_type": "bool",
        "valid": [False, True],
        "search": [placeholder],
        "placeholder": placeholder,
    }


def kappa(placeholder: float) -> dict:
    return {
        "kind": "param",
        "value_type": "float",
        "valid": {"type": "float", "low": 0.0, "high": 100.0},
        "search": {"type": "float", "low": 0.5, "high": 10.0},
        "placeholder": placeholder,
    }


def bound() -> dict:
    return {
        "kind": "structure",
        "placeholder": "ob",
        "branches": {
            "none": {},
            "ob": {"kappa": kappa(1.0)},
            "adaptive_ob": {
                "kappa": kappa(2.0),
                "beta2": {
                    "kind": "param",
                    "value_type": "float",
                    "valid": {"type": "float", "low": 0.0, "high": 1.0},
                    "search": {"type": "float", "low": 0.9, "high": 0.9999},
                    "placeholder": 0.999,
                },
            },
        },
    }


def a_trial() -> optuna.trial.Trial:
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    return study.ask()


def test_an_unknown_override_is_rejected() -> None:
    entry = make_entry({"learning_rate": learning_rate()})

    with pytest.raises(SpaceError, match="learnign_rate"):
        resolve_parameters(entry, {"learnign_rate": ChoiceSpec.model_validate([0.1])})


def test_an_unknown_override_lists_what_the_entry_declares() -> None:
    entry = make_entry({"learning_rate": learning_rate()})

    with pytest.raises(SpaceError, match="entry declares: learning_rate"):
        resolve_parameters(entry, {"learnign_rate": ChoiceSpec.model_validate([0.1])})


def test_an_override_outside_valid_is_rejected() -> None:
    entry = make_entry({"learning_rate": learning_rate()})

    with pytest.raises(SpaceError, match="learning_rate"):
        resolve_parameters(entry, {"learning_rate": ChoiceSpec.model_validate([20.0])})


def test_an_override_range_outside_valid_is_rejected() -> None:
    entry = make_entry({"learning_rate": learning_rate()})
    wide = {"type": "float", "low": 1e-4, "high": 50.0}

    with pytest.raises(SpaceError, match="learning_rate"):
        resolve_parameters(entry, {"learning_rate": _spec(wide)})


def test_a_single_point_search_yields_that_point() -> None:
    entry = make_entry({"reward_trace_reset_on_done": single_point()})
    resolved = resolve_parameters(entry, {})

    assert sample_parameters(a_trial(), resolved) == {
        "reward_trace_reset_on_done": True
    }


def test_an_experiment_may_override_a_single_point_search() -> None:
    entry = make_entry({"reward_trace_reset_on_done": single_point()})
    resolved = resolve_parameters(
        entry, {"reward_trace_reset_on_done": ChoiceSpec.model_validate([False])}
    )

    assert sample_parameters(a_trial(), resolved) == {
        "reward_trace_reset_on_done": False
    }


def test_branch_parameters_are_keyed_by_their_path() -> None:
    entry = make_entry({"optimizer_bound": bound()})
    resolved = resolve_parameters(
        entry, {"optimizer_bound": ChoiceSpec.model_validate(["ob"])}
    )

    chosen = sample_parameters(a_trial(), resolved)

    assert chosen["optimizer_bound"] == "ob"
    assert "ob.kappa" in chosen
    assert "adaptive_ob.kappa" in chosen


def test_an_unchosen_branch_collapses_to_its_placeholders() -> None:
    entry = make_entry({"optimizer_bound": bound()})
    resolved = resolve_parameters(
        entry, {"optimizer_bound": ChoiceSpec.model_validate(["ob"])}
    )

    chosen = sample_parameters(a_trial(), resolved)

    assert 0.5 <= chosen["ob.kappa"] <= 10.0
    assert chosen["adaptive_ob.kappa"] == 2.0
    assert chosen["adaptive_ob.beta2"] == 0.999


def test_the_manifest_carries_every_declared_parameter() -> None:
    entry = make_entry({"learning_rate": learning_rate(), "optimizer_bound": bound()})
    resolved = resolve_parameters(entry, {})

    chosen = sample_parameters(a_trial(), resolved)

    assert set(chosen) == {
        "learning_rate",
        "optimizer_bound",
        "ob.kappa",
        "adaptive_ob.kappa",
        "adaptive_ob.beta2",
    }


def test_a_branch_override_is_checked_against_that_branch() -> None:
    entry = make_entry({"optimizer_bound": bound()})

    with pytest.raises(SpaceError, match=r"ob\.kappa"):
        resolve_parameters(
            entry, {"ob.kappa": ChoiceSpec.model_validate([500.0])}
        )


def test_a_structure_may_not_choose_a_branch_it_does_not_have() -> None:
    entry = make_entry({"optimizer_bound": bound()})

    with pytest.raises(SpaceError, match="obgd"):
        resolve_parameters(
            entry, {"optimizer_bound": ChoiceSpec.model_validate(["obgd"])}
        )


def test_a_structure_left_alone_takes_its_placeholder_branch() -> None:
    entry = make_entry({"optimizer_bound": bound()})

    chosen = sample_parameters(a_trial(), resolve_parameters(entry, {}))

    assert chosen["optimizer_bound"] == "ob"


def test_a_structure_may_not_be_given_more_than_one_branch() -> None:
    entry = make_entry({"optimizer_bound": bound()})

    with pytest.raises(SpaceError, match="not searched"):
        resolve_parameters(
            entry, {"optimizer_bound": ChoiceSpec.model_validate(["ob", "none"])}
        )


def test_the_grid_sampler_enumerates_a_pinned_branch() -> None:
    entry = make_entry({"optimizer_bound": bound()})
    resolved = resolve_parameters(
        entry,
        {
            "optimizer_bound": ChoiceSpec.model_validate(["ob"]),
            "ob.kappa": ChoiceSpec.model_validate([1.0, 2.0]),
        },
    )

    built = grid_distributions(resolved)

    assert sorted(built) == ["ob.kappa"]
    assert list(built["ob.kappa"].choices) == [1.0, 2.0]


def test_the_grid_sampler_refuses_a_range() -> None:
    entry = make_entry({"learning_rate": learning_rate()})

    with pytest.raises(SpaceError, match="learning_rate"):
        grid_distributions(resolve_parameters(entry, {}))


def _spec(raw: dict):
    from training_sdk.contract import FloatSpec

    return FloatSpec.model_validate(raw)
