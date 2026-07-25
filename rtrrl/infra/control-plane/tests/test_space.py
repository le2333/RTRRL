import optuna
import pytest
from training_sdk.contract import ChoiceSpec, EntryDescriptor

from trainer_infra.space import (
    SpaceError,
    distributions,
    minimum_total_steps,
    resolve_space,
)


def make_entry(space: dict) -> EntryDescriptor:
    return EntryDescriptor.model_validate(
        {
            "command": ["run"],
            "source_hash": "sha256:0",
            "metrics": ["episode_return"],
            "space": space,
        }
    )


def test_override_replaces_entry_by_key() -> None:
    entry = make_entry(
        {
            "total_steps": {"type": "int", "low": 1, "high": 1000},
            "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-2},
        }
    )
    resolved = resolve_space(entry, {"total_steps": ChoiceSpec.model_validate([128])})
    assert resolved["total_steps"].choices == (128,)
    assert resolved["learning_rate"].high == 1e-2


def test_unknown_override_key_is_rejected() -> None:
    entry = make_entry({"total_steps": [128]})
    with pytest.raises(SpaceError, match="learnign_rate"):
        resolve_space(entry, {"learnign_rate": ChoiceSpec.model_validate([0.1])})


def test_space_without_total_steps_is_rejected() -> None:
    entry = make_entry({"learning_rate": [0.1]})
    with pytest.raises(SpaceError, match="total_steps"):
        resolve_space(entry, {})


def test_distributions_cover_every_key() -> None:
    entry = make_entry(
        {
            "total_steps": [128],
            "env": ["walker2d", "ant"],
            "num_envs": {"type": "int", "low": 256, "high": 1024, "step": 256},
            "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
        }
    )
    built = distributions(resolve_space(entry, {}))
    assert set(built) == {"total_steps", "env", "num_envs", "learning_rate"}
    assert isinstance(built["total_steps"], optuna.distributions.CategoricalDistribution)
    assert isinstance(built["num_envs"], optuna.distributions.IntDistribution)
    assert built["learning_rate"].log is True


def test_minimum_total_steps_uses_smallest_producible_value() -> None:
    entry = make_entry({"total_steps": {"type": "int", "low": 100, "high": 900}})
    assert minimum_total_steps(resolve_space(entry, {})) == 100
    entry = make_entry({"total_steps": [900, 300]})
    assert minimum_total_steps(resolve_space(entry, {})) == 300


def test_unknown_override_key_lists_declared_keys() -> None:
    entry = make_entry({"total_steps": [128], "learning_rate": [0.1]})
    with pytest.raises(
        SpaceError,
        match="experiment declares parameters the entry does not accept: learnign_rate; "
        "entry declares: learning_rate, total_steps",
    ):
        resolve_space(entry, {"learnign_rate": ChoiceSpec.model_validate([0.1])})


def test_total_steps_with_non_integer_choices_is_rejected() -> None:
    entry = make_entry({"total_steps": [128, 256.0]})
    with pytest.raises(SpaceError, match="total_steps choices must all be integers"):
        minimum_total_steps(resolve_space(entry, {}))


def test_total_steps_float_range_is_rejected() -> None:
    entry = make_entry({"total_steps": {"type": "float", "low": 100.0, "high": 900.0}})
    with pytest.raises(
        SpaceError,
        match="total_steps must be an integer range or an integer choice list",
    ):
        minimum_total_steps(resolve_space(entry, {}))


def test_total_steps_string_choice_is_rejected() -> None:
    entry = make_entry({"total_steps": ["128"]})
    with pytest.raises(SpaceError, match="total_steps choices must all be integers"):
        minimum_total_steps(resolve_space(entry, {}))


def test_total_steps_boolean_choice_is_rejected() -> None:
    entry = make_entry({"total_steps": [True]})
    with pytest.raises(SpaceError, match="total_steps choices must all be integers"):
        minimum_total_steps(resolve_space(entry, {}))


def test_total_steps_float_choice_is_rejected() -> None:
    entry = make_entry({"total_steps": [128.0]})
    with pytest.raises(SpaceError, match="total_steps choices must all be integers"):
        minimum_total_steps(resolve_space(entry, {}))


def test_distributions_rejects_unsupported_space_entry() -> None:
    class UnsupportedSpec:
        pass

    with pytest.raises(SpaceError, match="unsupported space entry for rogue"):
        distributions({"rogue": UnsupportedSpec()})  # type: ignore[dict-item]
