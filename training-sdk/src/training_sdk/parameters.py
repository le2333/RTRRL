from __future__ import annotations

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


def structure(
    *, placeholder: Scalar, branches: dict[str, Any], search: Any = None
) -> Any:
    return field(
        default=placeholder,
        metadata={
            "trainer_kind": "structure",
            "placeholder": placeholder,
            "branches": branches,
            "search": search,
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

    search = item.metadata["search"]
    placeholder = item.metadata["placeholder"]
    if placeholder not in branches:
        raise ValueError(
            f"{item.name} placeholder {placeholder!r} is not one of its branches: "
            f"{', '.join(sorted(branches))}"
        )
    if search is not None:
        unknown = sorted(set(search) - set(branches))
        if unknown:
            raise ValueError(
                f"{item.name} search names unknown branches: "
                f"{', '.join(map(str, unknown))}"
            )
    return StructureSpec(
        placeholder=placeholder,
        search=tuple(search) if search is not None else None,
        branches=branches,
    )


def _valid_domain(name: str, value: Any, *, step: int) -> ValidSpec:
    if isinstance(value, list):
        return ChoiceSpec.model_validate(value)
    if isinstance(value, tuple) and len(value) == 2:
        low, high = value
        if all(side is None or isinstance(side, bool) is False and isinstance(side, int) for side in value):
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
        if all(isinstance(side, bool) is False and isinstance(side, int) for side in value):
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
