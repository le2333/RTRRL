"""The vocabulary a component declares its space in, and what it becomes.

One kind of node. A choice of component is a parameter called ``kind`` sitting
in the group its branches sit in, so nothing here has to ask whether a node is
a value or a structure, and a name used at two sites is two nodes rather than
a clash.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from memorax.parameters import (
    KIND,
    Choice,
    FloatRange,
    IntRange,
    Parameter,
    describe,
    describe_parameters,
    expand,
    flatten,
    group,
    param,
    read_branch,
    read_parameters,
    structure,
)


@dataclass(frozen=True)
class Adam:
    b1: float = param(valid=(0.0, 1.0), search=(0.5, 0.999))


@dataclass(frozen=True)
class Agent:
    learning_rate: float = param(valid=(1e-9, 10.0), search=(1e-4, 1e-2), log=True)
    hidden_dim: int = param(valid=(1, 512), search=(1, 512))
    eta_pi: float = param(valid=(0.0, None), search=[0.0])
    reward_trace_reset_on_done: bool = param(valid=[False, True], search=[True])
    optimizer_base: str = structure(branches={"sgd": (), "adam": Adam})


def test_dataclass_metadata_exports_a_parameter_tree() -> None:
    tree = describe_parameters(Agent)

    assert tree["learning_rate"].search.low == 1e-4
    assert tree["learning_rate"].search.log is True
    assert isinstance(tree["hidden_dim"].valid, IntRange)
    assert tree["optimizer_base"]["adam"]["b1"].search.low == 0.5


def test_a_choice_of_component_is_a_parameter_and_not_another_kind_of_node() -> None:
    """Which is why nothing downstream branches on what a node is."""

    group = describe_parameters(Agent)["optimizer_base"]

    assert isinstance(group[KIND], Parameter)
    assert set(group[KIND].valid.values) == {"sgd", "adam"}


def test_a_branch_declaring_nothing_takes_no_room_in_the_tree() -> None:
    group = describe_parameters(Agent)["optimizer_base"]

    assert "sgd" not in group
    assert "sgd" in group[KIND].valid.values


def test_a_single_point_search_is_how_a_parameter_stays_out_of_a_sweep() -> None:
    tree = describe_parameters(Agent)

    assert tree["reward_trace_reset_on_done"].search.values == (True,)


def test_a_valid_domain_may_be_open_on_one_side() -> None:
    tree = describe_parameters(Agent)

    assert isinstance(tree["eta_pi"].valid, FloatRange)
    assert tree["eta_pi"].valid.low == 0.0
    assert tree["eta_pi"].valid.high is None


def test_a_list_valid_domain_becomes_a_choice() -> None:
    tree = describe_parameters(Agent)

    assert isinstance(tree["reward_trace_reset_on_done"].valid, Choice)
    assert tree["reward_trace_reset_on_done"].valid.values == (False, True)


def test_search_must_lie_inside_valid() -> None:
    with pytest.raises(ValueError, match="above the valid high"):
        Parameter(valid=FloatRange(0.0, 1.0), search=FloatRange(2.0, 3.0))


def test_a_log_search_must_start_above_zero() -> None:
    with pytest.raises(ValueError, match="above zero"):
        Parameter(valid=FloatRange(0.0, 1.0), search=FloatRange(0.0, 1.0, log=True))


def test_a_search_domain_must_be_closed() -> None:
    with pytest.raises(ValueError, match="closed"):
        Parameter(valid=FloatRange(0.0, None), search=FloatRange(0.0, None))


def test_a_choice_must_offer_something() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Choice([])


def test_a_range_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        FloatRange(low=1.0, high=0.0)


def test_the_name_a_group_selects_with_is_reserved() -> None:
    @dataclass(frozen=True)
    class Bad:
        kind: float = param(valid=(0.0, 1.0), search=[0.5])

    with pytest.raises(ValueError, match=KIND):
        describe_parameters(Bad)


def test_a_field_that_is_not_a_declaration_is_refused() -> None:
    @dataclass(frozen=True)
    class Bad:
        x: float = 0.5

    with pytest.raises(ValueError, match="x"):
        describe_parameters(Bad)


def test_a_component_is_read_back_from_its_subtree() -> None:
    params = {
        f"optimizer_base.{KIND}": "adam",
        "optimizer_base.adam.b1": 0.95,
    }

    branch, component = read_branch(params, "optimizer_base", {"sgd": (), "adam": Adam})

    assert branch == "adam"
    assert component == Adam(b1=0.95)


def test_a_branch_without_parameters_reads_as_none() -> None:
    branch, component = read_branch(
        {f"optimizer_base.{KIND}": "sgd"}, "optimizer_base", {"sgd": (), "adam": Adam}
    )

    assert branch == "sgd"
    assert component is None


def test_a_missing_key_is_an_error_rather_than_a_default() -> None:
    with pytest.raises(KeyError, match="optimizer_base.adam.b1"):
        read_branch(
            {f"optimizer_base.{KIND}": "adam"}, "optimizer_base", {"adam": Adam}
        )


def test_an_unknown_branch_is_refused() -> None:
    with pytest.raises(KeyError, match="rmsprop"):
        read_branch(
            {f"optimizer_base.{KIND}": "rmsprop"}, "optimizer_base", {"adam": Adam}
        )


def test_the_catalog_names_every_branch_and_a_run_reaches_only_one() -> None:
    """The two are different questions, and the tree answers them separately."""

    tree = describe_parameters(Agent)

    assert "optimizer_base.adam.b1" in flatten(tree)
    assert "optimizer_base.adam.b1" not in expand(
        tree, {f"optimizer_base.{KIND}": "sgd"}
    )
    assert "optimizer_base.adam.b1" in expand(tree, {f"optimizer_base.{KIND}": "adam"})


def test_the_shipped_shape_is_valid_and_search_and_nothing_else() -> None:
    """What a machine that cannot import any of this reads off the image."""

    shipped = describe(describe_parameters(Agent))

    assert shipped["hidden_dim"] == {
        "valid": {"type": "int", "low": 1, "high": 512, "step": 1, "log": False},
        "search": {"type": "int", "low": 1, "high": 512, "step": 1, "log": False},
    }
    assert set(shipped["optimizer_base"]) == {KIND, "adam"}


@dataclass(frozen=True)
class Role:
    """One declaration, used at two sites. What a scope buys."""

    optimizer_base: str = structure(branches={"sgd": (), "adam": Adam})


@dataclass(frozen=True)
class Pair:
    actor: Role = group(of=Role)
    critic: Role = group(of=Role)
    gamma: float = param(valid=(0.0, 1.0), search=[0.9])


def test_a_group_is_a_level_without_a_choice() -> None:
    """Because there is no second actor to choose between."""

    tree = describe_parameters(Pair)

    assert KIND not in tree["actor"]
    assert set(tree["actor"]) == {"optimizer_base"}
    assert KIND in tree["actor"]["optimizer_base"]


def test_the_same_declaration_at_two_sites_is_two_sets_of_names() -> None:
    """Which is the whole point: no prefix written into a field name."""

    declared = flatten(describe_parameters(Pair))

    assert "actor.optimizer_base.adam.b1" in declared
    assert "critic.optimizer_base.adam.b1" in declared


def test_a_group_is_descended_unconditionally() -> None:
    reached = expand(describe_parameters(Pair), {"critic.optimizer_base.kind": "adam"})

    assert reached["actor.optimizer_base.kind"] == "sgd"
    assert "critic.optimizer_base.adam.b1" in reached
    assert "actor.optimizer_base.adam.b1" not in reached


def test_a_group_reads_back_as_the_class_it_names() -> None:
    read = read_parameters(
        Pair,
        {
            "gamma": 0.9,
            f"actor.optimizer_base.{KIND}": "adam",
            "actor.optimizer_base.adam.b1": 0.95,
            f"critic.optimizer_base.{KIND}": "sgd",
        },
    )

    assert read.actor.optimizer_base == "adam"
    assert read.critic.optimizer_base == "sgd"


def test_the_shipped_tree_has_no_third_kind_of_node_for_a_group() -> None:
    """A group and a structure are both mappings; only the kind leaf differs."""

    shipped = describe(describe_parameters(Pair))

    assert set(shipped["actor"]) == {"optimizer_base"}
    assert set(shipped["actor"]["optimizer_base"]) == {KIND, "adam"}
