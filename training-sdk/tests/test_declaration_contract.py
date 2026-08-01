from __future__ import annotations

from dataclasses import dataclass, fields

import pytest

from training_sdk.contract import ChoiceSpec, ParameterSpec, StructureSpec
from training_sdk.parameters import describe_parameters, param, read_branch, structure


@dataclass(frozen=True)
class Conforming:
    width: int = param(valid=(1, 4096), search=(16, 512), placeholder=64)
    rate: float = param(valid=(1e-9, 10.0), search=(1e-5, 1.0), placeholder=0.1, log=True)
    mode: str = param(valid=["a", "b"], search=["a", "b"], placeholder="a")


@dataclass(frozen=True)
class Selecting:
    choice: str = structure(placeholder="left", branches={"left": Conforming, "right": ()})


def keys(tree, prefix=""):
    found = {}
    for name, node in tree.items():
        key = prefix + name
        found[key] = node
        if isinstance(node, StructureSpec):
            for branch, sub in node.branches.items():
                found |= keys(sub, f"{key}.{branch}.")
    return found


def inside(spec, value) -> bool:
    if isinstance(spec, ChoiceSpec):
        return value in spec.choices
    low, high = spec.low, spec.high
    return (low is None or value >= low) and (high is None or value <= high)


def test_every_declared_field_becomes_a_key() -> None:
    tree = describe_parameters(Conforming)

    assert set(tree) == {item.name for item in fields(Conforming)}


def test_every_branch_of_a_structure_expands_under_its_own_path() -> None:
    found = keys(describe_parameters(Selecting))

    assert "choice" in found
    for item in fields(Conforming):
        assert f"choice.left.{item.name}" in found


def test_every_placeholder_lies_inside_its_valid() -> None:
    for key, node in keys(describe_parameters(Selecting)).items():
        if isinstance(node, ParameterSpec):
            assert inside(node.valid, node.placeholder), key


def test_every_search_lies_inside_its_valid() -> None:
    for key, node in keys(describe_parameters(Selecting)).items():
        if not isinstance(node, ParameterSpec):
            continue
        if isinstance(node.search, ChoiceSpec):
            assert all(inside(node.valid, one) for one in node.search.choices), key
        else:
            assert inside(node.valid, node.search.low), key
            assert inside(node.valid, node.search.high), key


def test_a_component_reads_back_as_what_it_declared() -> None:
    tree = describe_parameters(Selecting)
    manifest = {"choice": "left"}
    for key, node in keys(tree).items():
        if isinstance(node, ParameterSpec):
            manifest[key] = node.placeholder

    branch, component = read_branch(
        manifest, "choice", {"left": Conforming, "right": ()}
    )

    assert branch == "left"
    assert component == Conforming()


def test_a_field_that_is_not_declared_is_refused() -> None:
    @dataclass(frozen=True)
    class Bare:
        width: int = 64

    with pytest.raises(ValueError, match="width"):
        describe_parameters(Bare)


def test_a_search_outside_valid_is_refused() -> None:
    @dataclass(frozen=True)
    class Wide:
        width: int = param(valid=(1, 10), search=(1, 100), placeholder=1)

    with pytest.raises(ValueError, match="width"):
        describe_parameters(Wide)


def test_a_placeholder_outside_valid_is_refused() -> None:
    @dataclass(frozen=True)
    class Outside:
        width: int = param(valid=(1, 10), search=[1], placeholder=99)

    with pytest.raises(ValueError, match="width"):
        describe_parameters(Outside)


def test_a_log_search_starting_at_zero_is_refused() -> None:
    @dataclass(frozen=True)
    class AtZero:
        rate: float = param(valid=(0.0, 1.0), search=(0.0, 1.0), placeholder=0.5, log=True)

    with pytest.raises(ValueError, match="rate"):
        describe_parameters(AtZero)


def test_a_structure_placeholder_outside_its_branches_is_refused() -> None:
    @dataclass(frozen=True)
    class Astray:
        choice: str = structure(placeholder="missing", branches={"left": (), "right": ()})

    with pytest.raises(ValueError, match="choice"):
        describe_parameters(Astray)


def test_a_component_that_is_not_a_dataclass_is_refused() -> None:
    class Loose:
        width = 64

    with pytest.raises(TypeError, match="Loose"):
        describe_parameters(Loose)
