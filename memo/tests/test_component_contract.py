from __future__ import annotations

from dataclasses import fields, is_dataclass

import pytest
from training_sdk.contract import ChoiceSpec, ParameterSpec, StructureSpec
from training_sdk.parameters import describe_parameters, read_branch

from memorax.networks.backbones import Lru, Mlp, Rtu
from memorax.networks.initialization import Sparse
from memorax.rl.normalization import DiscountedNormalization, RunningNormalization
from memorax.rl.updates import Adam, AdaptiveObBound, ObBound, Sgd

COMPONENTS = (
    Rtu,
    Lru,
    Mlp,
    ObBound,
    AdaptiveObBound,
    Sgd,
    Adam,
    RunningNormalization,
    DiscountedNormalization,
    Sparse,
)


def inside(spec, value) -> bool:
    if isinstance(spec, ChoiceSpec):
        return value in spec.choices
    return (spec.low is None or value >= spec.low) and (
        spec.high is None or value <= spec.high
    )


@pytest.fixture(params=COMPONENTS, ids=[one.__name__ for one in COMPONENTS])
def component(request):
    return request.param


def test_a_component_is_a_frozen_dataclass(component) -> None:
    assert is_dataclass(component)
    assert component.__dataclass_params__.frozen


def test_a_component_carries_declarations_and_no_methods(component) -> None:
    own = {
        name
        for name, value in vars(component).items()
        if not name.startswith("__") and callable(value)
    }
    assert not own, f"{component.__name__} defines {sorted(own)}"


def test_every_field_of_a_component_expands(component) -> None:
    tree = describe_parameters(component)

    assert set(tree) == {item.name for item in fields(component)}


def test_every_value_a_component_declares_is_inside_its_own_bounds(component) -> None:
    for name, node in describe_parameters(component).items():
        if not isinstance(node, ParameterSpec):
            continue
        assert inside(node.valid, node.placeholder), name
        if isinstance(node.search, ChoiceSpec):
            assert all(inside(node.valid, one) for one in node.search.choices), name
        else:
            assert inside(node.valid, node.search.low), name
            assert inside(node.valid, node.search.high), name


def test_a_component_reads_back_as_what_it_declared(component) -> None:
    tree = describe_parameters(component)
    manifest = {"chosen": "only"}
    for name, node in tree.items():
        assert not isinstance(node, StructureSpec), "nested here needs a path walk"
        manifest[f"chosen.only.{name}"] = node.placeholder

    _, read = read_branch(manifest, "chosen", {"only": component})

    assert read == component()
