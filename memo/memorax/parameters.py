"""How a component declares its parameter space, and what that becomes.

A tree with one kind of node. A leaf is a :class:`Parameter`, carrying the two
domains that belong to two different knowers: ``valid`` is what the component
can mean, ``search`` is what it suggests looking through, and an experiment
narrows the second within the first. Anything else is a mapping, and nesting is
the only grouping there is.

A choice of component is not a second kind of node. It is a parameter named
``kind`` living in the group its branches live in, so ``backbone.kind`` selects
and ``backbone.rtu.hidden_dim`` belongs to what it selected. Nothing has to
decide whether a node is a value or a structure, and a name repeated at two
sites -- ``ob`` under both roles, ``sparse`` at every initialisation -- is two
nodes rather than a collision, because there is no flat namespace for them to
collide in.

What ships is this tree as JSON: ``{"valid": ..., "search": ...}`` at the
leaves and mappings above them. The control plane reads that and never imports
any of this, which is why the vocabulary is dataclasses and standard library
rather than anything either side would have to install.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import MISSING
from dataclasses import Field as DataclassField
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, TypeAlias

Scalar: TypeAlias = bool | int | float | str

# The name a group gives to the parameter that selects among its branches.
# Reserved: a component cannot declare a parameter called this.
KIND = "kind"


@dataclass(frozen=True, init=False)
class Choice:
    """A fixed set of values. One value is how a domain says "not searched"."""

    values: tuple[Scalar, ...]

    def __init__(self, values: Sequence[Scalar]) -> None:
        object.__setattr__(self, "values", tuple(values))
        if not self.values:
            raise ValueError("a choice must offer at least one value")


@dataclass(frozen=True)
class FloatRange:
    low: float | None = None
    high: float | None = None
    log: bool = False

    def __post_init__(self) -> None:
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError(f"float low {self.low} exceeds high {self.high}")


@dataclass(frozen=True)
class IntRange:
    low: int | None = None
    high: int | None = None
    step: int = 1
    log: bool = False

    def __post_init__(self) -> None:
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError(f"int low {self.low} exceeds high {self.high}")
        if self.step < 1:
            raise ValueError("int step must be positive")


Range: TypeAlias = Choice | FloatRange | IntRange


@dataclass(frozen=True)
class Parameter:
    """One value's two domains.

    ``valid`` is what the component can mean and only it knows -- a discount
    outside [0, 1] is not a bad idea, it is not a discount. ``search`` is a
    suggestion an experiment may narrow, so it has to be bounded on both sides
    even where ``valid`` is not: a sampler cannot draw from an open interval.
    """

    valid: Range
    search: Range

    def __post_init__(self) -> None:
        _bounded(self.search)
        _within(self.valid, self.search)


ParameterNode: TypeAlias = "Parameter | Mapping[str, ParameterNode]"
ParameterTree: TypeAlias = Mapping[str, ParameterNode]


def _bounded(search: Range) -> None:
    if isinstance(search, Choice):
        return
    if search.low is None or search.high is None:
        raise ValueError("a search domain must be closed on both sides")
    if search.log and search.low <= 0:
        raise ValueError("a log search domain must start above zero")


def _within(valid: Range, search: Range) -> None:
    """Every value the search can produce is one the component accepts."""

    if isinstance(search, Choice):
        for value in search.values:
            _accepts(valid, value)
        return
    _accepts(valid, search.low)
    _accepts(valid, search.high)


def _accepts(valid: Range, value: Scalar) -> None:
    if isinstance(valid, Choice):
        if value not in valid.values:
            raise ValueError(f"{value!r} is not one of {list(valid.values)}")
        return
    if isinstance(valid, IntRange) and type(value) is not int:
        raise ValueError(f"{value!r} is not an int")
    if type(value) not in (int, float):
        raise ValueError(f"{value!r} is not numeric")
    if valid.low is not None and value < valid.low:
        raise ValueError(f"{value!r} is below the valid low {valid.low}")
    if valid.high is not None and value > valid.high:
        raise ValueError(f"{value!r} is above the valid high {valid.high}")


# ---------------------------------------------------------------- serialising


def describe(tree: ParameterTree) -> dict[str, Any]:
    """The tree as the JSON a machine that cannot import this reads."""

    return {
        name: (
            _describe_parameter(node) if isinstance(node, Parameter) else describe(node)
        )
        for name, node in tree.items()
    }


def _describe_parameter(parameter: Parameter) -> dict[str, Any]:
    return {
        "valid": _describe_range(parameter.valid),
        "search": _describe_range(parameter.search),
    }


def _describe_range(domain: Range) -> dict[str, Any]:
    if isinstance(domain, Choice):
        return {"type": "choice", "values": list(domain.values)}
    if isinstance(domain, FloatRange):
        return {
            "type": "float",
            "low": domain.low,
            "high": domain.high,
            "log": domain.log,
        }
    return {
        "type": "int",
        "low": domain.low,
        "high": domain.high,
        "step": domain.step,
        "log": domain.log,
    }


# ------------------------------------------------------------------ declaring


def param(
    *,
    valid: Any,
    search: Any,
    default: Any = MISSING,
    log: bool = False,
    step: int = 1,
) -> Any:
    """Declare one parameter as a dataclass field.

    A tuple is a range and a list is a choice, on both domains, so a component
    writes ``(0.0, 1.0)`` and ``["lecun", "sparse"]`` rather than naming the
    class each time.

    ``default`` is the language's own, for a component constructed directly
    where the value is beside the point -- Adam's betas in a test about
    bounding. It is not in the tree and not in the catalog, and reading a
    component out of a run configuration never reaches it: a name the
    configuration does not carry is an error there, not a default. That is the
    difference between this and the ``placeholder`` it replaces, which was in
    the contract and filled in branches nobody selected.
    """

    metadata = {
        "parameter": Parameter(
            valid=_domain(valid, log=log, step=step, open_ended=True),
            search=_domain(search, log=log, step=step, open_ended=False),
        )
    }
    if default is MISSING:
        return field(metadata=metadata)
    return field(default=default, metadata=metadata)


def structure(
    *, branches: Mapping[str, Any], search: Sequence[str] | None = None
) -> Any:
    """Declare a choice among components, and the branches it chooses between.

    ``branches`` maps a name to the dataclass that branch declares, or to
    nothing when it declares none. The same mapping is what the consuming side
    reads back with :func:`read_branch`, so the space and the registry are one
    object rather than two that have to agree.
    """

    return field(
        metadata={"branches": dict(branches), "search": tuple(search or branches)}
    )


def _domain(value: Any, *, log: bool, step: int, open_ended: bool) -> Range:
    if isinstance(value, Choice | FloatRange | IntRange):
        return value
    if isinstance(value, list):
        return Choice(value)
    if isinstance(value, tuple) and len(value) == 2:
        low, high = value
        if all(
            side is None or (isinstance(side, bool) is False and isinstance(side, int))
            for side in value
        ):
            return IntRange(low=low, high=high, step=step, log=log)
        return FloatRange(low=low, high=high, log=log)
    if open_ended:
        raise TypeError(f"unsupported valid domain {value!r}")
    raise TypeError(f"unsupported search domain {value!r}")


def describe_parameters(model: type) -> dict[str, ParameterNode]:
    """The tree a declaring dataclass stands for.

    A ``structure`` field becomes a group: the choice itself under ``kind``,
    beside one subtree per branch. So the group is the scope, and a branch name
    is only ever read relative to it.
    """

    if not is_dataclass(model):
        raise TypeError(f"{model.__name__} must be a dataclass")
    described: dict[str, ParameterNode] = {}
    for item in fields(model):
        if item.name == KIND:
            raise ValueError(f"{model.__name__}.{KIND} uses the reserved name")
        if "parameter" in item.metadata:
            described[item.name] = item.metadata["parameter"]
        elif "branches" in item.metadata:
            described[item.name] = _group(item)
        else:
            raise ValueError(
                f"{model.__name__}.{item.name} is not declared with param() or structure()"
            )
    return described


def _group(item: DataclassField) -> dict[str, ParameterNode]:
    branches: Mapping[str, Any] = item.metadata["branches"]
    group: dict[str, ParameterNode] = {
        KIND: Parameter(
            valid=Choice(sorted(branches)),
            search=Choice(item.metadata["search"]),
        )
    }
    for name, branch in branches.items():
        if branch in (None, (), {}):
            continue
        group[name] = describe_parameters(branch)
    return group


# ------------------------------------------------------------------- reading


def read_parameters(model: type, params: Mapping[str, Any], prefix: str = "") -> Any:
    if not is_dataclass(model):
        raise TypeError(f"{model.__name__} must be a dataclass")
    values = {}
    for item in fields(model):
        key = f"{prefix}{item.name}"
        if "branches" in item.metadata:
            values[item.name] = read_branch(
                params, item.name, item.metadata["branches"], prefix
            )[0]
            continue
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
    """Which branch a group selected, and that branch read back.

    Only the selected branch is read. The others are not filled in with
    anything, because a value nobody chose is a value nobody should be able to
    mistake for one that was chosen.
    """

    key = f"{prefix}{field_name}.{KIND}"
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
    return branch, read_parameters(
        model, params, prefix=f"{prefix}{field_name}.{branch}."
    )


# ------------------------------------------------------------------- walking


def flatten(tree: ParameterTree, prefix: str = "") -> dict[str, Parameter]:
    """Every leaf, by the dotted path that reaches it.

    The path is unique because the tree is, so this is a rendering rather than
    a namespace: nothing is decided here that the tree had not already decided.
    """

    found: dict[str, Parameter] = {}
    for name, node in tree.items():
        key = f"{prefix}{name}"
        if isinstance(node, Parameter):
            found[key] = node
        else:
            found |= flatten(node, f"{key}.")
    return found


def expand(
    tree: ParameterTree, overrides: Mapping[str, Scalar] | None = None
) -> dict[str, Scalar]:
    """One value per reachable leaf, taking the override where there is one.

    Only the branches actually selected are descended. What an unselected
    branch would have been given is not a smaller question than what it means,
    and neither has an answer worth writing down.
    """

    pinned = dict(overrides or {})
    return walk(tree, fill=lambda key, node: pinned.get(key, _single(node)))


def _single(node: Parameter) -> Scalar:
    """The one value a domain stands for when nothing is sampling it."""

    if isinstance(node.search, Choice):
        return node.search.values[0]
    return node.search.low


def walk(
    tree: ParameterTree,
    *,
    fill: Callable[[str, Parameter], Scalar],
    prefix: str = "",
) -> dict[str, Scalar]:
    """Descend the selected branches, asking ``fill`` for each leaf it reaches."""

    chosen: dict[str, Scalar] = {}
    groups: dict[str, ParameterTree] = {}
    for name, node in tree.items():
        key = f"{prefix}{name}"
        if isinstance(node, Parameter):
            chosen[key] = fill(key, node)
        else:
            groups[name] = node

    selected = chosen.get(f"{prefix}{KIND}")
    for name, group in groups.items():
        if selected is not None and name != str(selected):
            continue
        chosen |= walk(group, fill=fill, prefix=f"{prefix}{name}.")
    return chosen
