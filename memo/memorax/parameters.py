from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import Field as DataclassField
from dataclasses import field, fields, is_dataclass
from typing import Any

from training_sdk.contract import (
    ChoiceSpec,
    FloatSpec,
    FloatValidSpec,
    IntSpec,
    IntValidSpec,
    ParameterNode,
    ParameterSpec,
    Scalar,
    SpaceEntry,
    StructureSpec,
    ValidSpec,
)


def param(
    *,
    valid: Any,
    search: Any,
    placeholder: Scalar,
    log: bool = False,
    step: int = 1,
) -> Any:
    return field(
        default=placeholder,
        metadata={
            "trainer_kind": "param",
            "valid": valid,
            "search": search,
            "placeholder": placeholder,
            "log": log,
            "step": step,
        },
    )


def structure(*, placeholder: Scalar, branches: dict[str, Any]) -> Any:
    return field(
        default=placeholder,
        metadata={
            "trainer_kind": "structure",
            "placeholder": placeholder,
            "branches": branches,
        },
    )


def describe_parameters(model: type) -> dict[str, ParameterNode]:
    if not is_dataclass(model):
        raise TypeError(f"{model.__name__} must be a dataclass")
    described: dict[str, ParameterNode] = {}
    for item in fields(model):
        kind = item.metadata.get("trainer_kind")
        if kind == "param":
            described[item.name] = _parameter(item)
        elif kind == "structure":
            described[item.name] = _structure(item)
        else:
            raise ValueError(
                f"{model.__name__}.{item.name} is not declared with param() or "
                "structure()"
            )
    return described


def _parameter(item: DataclassField) -> ParameterSpec:
    step = int(item.metadata["step"])
    valid = _valid_domain(item.name, item.metadata["valid"], step=step)
    placeholder = item.metadata["placeholder"]
    _check_value(item.name, "placeholder", valid, placeholder)

    declared = item.metadata["search"]
    if declared is None:
        raise ValueError(f"{item.name} must declare a search domain")
    search = _search_domain(
        item.name, declared, log=bool(item.metadata["log"]), step=step
    )
    _check_search(item.name, valid, search)

    return ParameterSpec(
        value_type=_value_type(item.name, item.type),
        valid=valid,
        search=search,
        placeholder=placeholder,
    )


def _structure(item: DataclassField) -> StructureSpec:
    branches: dict[str, dict[str, ParameterNode]] = {}
    for name, branch in item.metadata["branches"].items():
        branches[name] = {} if branch in (None, (), {}) else describe_parameters(branch)

    placeholder = item.metadata["placeholder"]
    if placeholder not in branches:
        raise ValueError(
            f"{item.name} placeholder {placeholder!r} is not one of its branches: "
            f"{', '.join(sorted(branches))}"
        )
    return StructureSpec(placeholder=placeholder, branches=branches)


def _valid_domain(name: str, value: Any, *, step: int) -> ValidSpec:
    if isinstance(value, list):
        return ChoiceSpec.model_validate(value)
    if isinstance(value, tuple) and len(value) == 2:
        low, high = value
        if all(
            side is None or isinstance(side, bool) is False and isinstance(side, int)
            for side in value
        ):
            return IntValidSpec(type="int", low=low, high=high, step=step)
        return FloatValidSpec(type="float", low=low, high=high)
    raise TypeError(f"{name} has an unsupported valid domain {value!r}")


def _search_domain(name: str, value: Any, *, log: bool, step: int) -> SpaceEntry:
    if isinstance(value, list):
        return ChoiceSpec.model_validate(value)
    if isinstance(value, tuple) and len(value) == 2:
        low, high = value
        if low is None or high is None:
            raise ValueError(f"{name} search domains must be closed on both sides")
        if all(
            isinstance(side, bool) is False and isinstance(side, int) for side in value
        ):
            return IntSpec(type="int", low=low, high=high, step=step, log=log)
        return FloatSpec(type="float", low=low, high=high, log=log)
    raise TypeError(f"{name} has an unsupported search domain {value!r}")


def _value_type(name: str, annotation: Any) -> str:
    for kind, names in (
        ("bool", (bool, "bool")),
        ("int", (int, "int")),
        ("float", (float, "float")),
        ("str", (str, "str")),
    ):
        if annotation in names:
            return kind
    raise TypeError(f"{name} has an unsupported type {annotation!r}")


def _check_value(name: str, label: str, valid: ValidSpec, value: Scalar) -> None:
    if isinstance(valid, ChoiceSpec):
        if value not in valid.choices:
            raise ValueError(f"{name} {label} {value!r} is outside its valid choices")
        return
    if isinstance(valid, IntValidSpec):
        if type(value) is not int:
            raise ValueError(f"{name} {label} {value!r} is not an int")
    elif type(value) not in (int, float):
        raise ValueError(f"{name} {label} {value!r} is not numeric")
    numeric = float(value)
    if valid.low is not None and numeric < valid.low:
        raise ValueError(f"{name} {label} {value!r} is below its valid low {valid.low}")
    if valid.high is not None and numeric > valid.high:
        raise ValueError(
            f"{name} {label} {value!r} is above its valid high {valid.high}"
        )


def _check_search(name: str, valid: ValidSpec, search: SpaceEntry) -> None:
    if isinstance(search, ChoiceSpec):
        for choice in search.choices:
            _check_value(name, "search choice", valid, choice)
        return
    if search.log and search.low <= 0:
        raise ValueError(f"{name} log search low must be above zero")
    _check_value(name, "search low", valid, search.low)
    _check_value(name, "search high", valid, search.high)


def read_parameters(model: type, params: Mapping[str, Any], prefix: str = "") -> Any:
    if not is_dataclass(model):
        raise TypeError(f"{model.__name__} must be a dataclass")
    values = {}
    for item in fields(model):
        key = f"{prefix}{item.name}"
        if key not in params:
            raise KeyError(f"the manifest carries no {key!r}")
        values[item.name] = params[key]
    return model(**values)


def read_branch(
    params: Mapping[str, Any],
    field_name: str,
    branches: Mapping[str, Any],
    prefix: str = "",
) -> tuple[str, Any]:
    key = f"{prefix}{field_name}"
    if key not in params:
        raise KeyError(f"the manifest carries no {key!r}")
    branch = str(params[key])
    if branch not in branches:
        raise KeyError(
            f"{key} is {branch!r}, which is not one of {', '.join(sorted(branches))}"
        )
    model = branches[branch]
    if model in (None, (), {}):
        return branch, None
    return branch, read_parameters(model, params, prefix=f"{key}.{branch}.")


def flatten(tree: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    found: dict[str, Any] = {}
    for name, node in tree.items():
        key = f"{prefix}{name}"
        if key in found:
            raise ValueError(f"parameter {key!r} is declared more than once")
        found[key] = node
        if isinstance(node, StructureSpec):
            for branch, subtree in node.branches.items():
                for sub_key, sub_node in flatten(subtree, f"{key}.{branch}.").items():
                    if sub_key in found:
                        raise ValueError(
                            f"parameter {sub_key!r} is declared more than once"
                        )
                    found[sub_key] = sub_node
    return found


def walk(
    tree: Mapping[str, Any],
    *,
    choose: Callable[[str, StructureSpec], str],
    fill: Callable[[str, ParameterSpec, bool], Scalar],
    prefix: str = "",
    active: bool = True,
) -> dict[str, Scalar]:
    chosen: dict[str, Scalar] = {}
    for name, node in tree.items():
        key = f"{prefix}{name}"
        if isinstance(node, StructureSpec):
            branch = choose(key, node) if active else str(node.placeholder)
            chosen[key] = branch
            for candidate, subtree in node.branches.items():
                chosen |= walk(
                    subtree,
                    choose=choose,
                    fill=fill,
                    prefix=f"{key}.{candidate}.",
                    active=active and candidate == branch,
                )
        else:
            chosen[key] = fill(key, node, active)
    return chosen


def expand(
    tree: Mapping[str, Any], overrides: Mapping[str, Scalar] | None = None
) -> dict[str, Scalar]:
    pinned = dict(overrides or {})
    return walk(
        tree,
        choose=lambda key, node: str(pinned.get(key, node.placeholder)),
        fill=lambda key, node, active: (
            pinned.get(key, _single(node)) if active else node.placeholder
        ),
    )


def _single(node: ParameterSpec) -> Scalar:
    if isinstance(node.search, ChoiceSpec):
        return node.search.choices[0]
    return node.search.low
