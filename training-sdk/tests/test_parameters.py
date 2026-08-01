from __future__ import annotations

from dataclasses import dataclass

import pytest

from training_sdk.contract import ChoiceSpec, FloatValidSpec
from training_sdk.parameters import describe_parameters, param, structure


@dataclass(frozen=True)
class Adam:
    b1: float = param(valid=(0.0, 1.0), search=(0.5, 0.999), placeholder=0.9)


@dataclass(frozen=True)
class Agent:
    learning_rate: float = param(
        valid=(1e-9, 10.0), search=(1e-4, 1e-2), placeholder=0.001, log=True
    )
    hidden_dim: int = param(valid=(1, 512), search=(1, 512), placeholder=128)
    eta_pi: float = param(valid=(0.0, None), search=[0.0], placeholder=0.0)
    reward_trace_reset_on_done: bool = param(
        valid=[False, True], search=[True], placeholder=True
    )
    optimizer_base: str = structure(
        placeholder="sgd",
        search=["sgd", "adam"],
        branches={"sgd": (), "adam": Adam},
    )


def test_dataclass_metadata_exports_a_parameter_tree() -> None:
    tree = describe_parameters(Agent)

    assert tree["learning_rate"].search.low == 1e-4
    assert tree["learning_rate"].search.log is True
    assert tree["hidden_dim"].value_type == "int"
    assert tree["optimizer_base"].branches["adam"]["b1"].placeholder == 0.9
    assert tree["optimizer_base"].branches["sgd"] == {}


def test_a_single_point_search_is_how_a_parameter_stays_out_of_a_sweep() -> None:
    tree = describe_parameters(Agent)

    assert tree["reward_trace_reset_on_done"].search.choices == (True,)


def test_a_parameter_must_declare_a_search() -> None:
    @dataclass(frozen=True)
    class Bad:
        x: float = param(valid=(0.0, 1.0), search=None, placeholder=0.5)

    with pytest.raises(ValueError, match="x"):
        describe_parameters(Bad)


def test_a_valid_domain_may_be_open_on_one_side() -> None:
    tree = describe_parameters(Agent)

    assert isinstance(tree["eta_pi"].valid, FloatValidSpec)
    assert tree["eta_pi"].valid.low == 0.0
    assert tree["eta_pi"].valid.high is None


def test_a_list_valid_domain_becomes_a_choice() -> None:
    tree = describe_parameters(Agent)

    assert isinstance(tree["reward_trace_reset_on_done"].valid, ChoiceSpec)
    assert tree["reward_trace_reset_on_done"].valid.choices == (False, True)


def test_search_must_lie_inside_valid() -> None:
    @dataclass(frozen=True)
    class Bad:
        x: float = param(valid=(0.0, 1.0), search=(2.0, 3.0), placeholder=0.5)

    with pytest.raises(ValueError, match="x"):
        describe_parameters(Bad)


def test_placeholder_must_lie_inside_valid() -> None:
    @dataclass(frozen=True)
    class Bad:
        x: int = param(valid=(1, 5), search=[1], placeholder=0)

    with pytest.raises(ValueError, match="x"):
        describe_parameters(Bad)


def test_a_log_search_must_start_above_zero() -> None:
    @dataclass(frozen=True)
    class Bad:
        x: float = param(valid=(0.0, 1.0), search=(0.0, 1.0), placeholder=0.5, log=True)

    with pytest.raises(ValueError, match="x"):
        describe_parameters(Bad)


def test_a_search_domain_must_be_closed() -> None:
    @dataclass(frozen=True)
    class Bad:
        x: float = param(valid=(0.0, None), search=(0.0, None), placeholder=0.5)

    with pytest.raises(ValueError, match="closed"):
        describe_parameters(Bad)


def test_a_structure_placeholder_must_name_a_branch() -> None:
    @dataclass(frozen=True)
    class Bad:
        s: str = structure(placeholder="nope", branches={"a": (), "b": ()})

    with pytest.raises(ValueError, match="s"):
        describe_parameters(Bad)


def test_a_structure_search_may_not_name_an_unknown_branch() -> None:
    @dataclass(frozen=True)
    class Bad:
        s: str = structure(
            placeholder="a", search=["a", "c"], branches={"a": (), "b": ()}
        )

    with pytest.raises(ValueError, match="c"):
        describe_parameters(Bad)


def test_a_field_that_is_not_a_declaration_is_refused() -> None:
    @dataclass(frozen=True)
    class Bad:
        x: float = 0.5

    with pytest.raises(ValueError, match="x"):
        describe_parameters(Bad)
