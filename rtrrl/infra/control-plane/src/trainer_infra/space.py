from __future__ import annotations

import math

from dataclasses import dataclass, field

from optuna.distributions import CategoricalDistribution
from optuna.trial import Trial
from training_sdk.parameters import flatten, walk
from training_sdk.contract import (
    ChoiceSpec,
    EntryDescriptor,
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


class SpaceError(ValueError):
    """The resolved search space is not usable."""


@dataclass(frozen=True)
class ResolvedParameters:
    tree: dict[str, ParameterNode]
    overrides: dict[str, SpaceEntry] = field(default_factory=dict)


def resolve_parameters(
    entry: EntryDescriptor, overrides: dict[str, SpaceEntry]
) -> ResolvedParameters:
    declared = flatten(entry.parameters)
    unknown = sorted(set(overrides) - set(declared))
    if unknown:
        raise SpaceError(
            "experiment declares parameters the entry does not accept: "
            f"{', '.join(unknown)}; entry declares: {', '.join(sorted(declared))}"
        )
    for key, override in overrides.items():
        node = declared[key]
        if isinstance(node, StructureSpec):
            _check_structure_override(key, node, override)
        else:
            _check_override(key, node, override)
    return ResolvedParameters(tree=entry.parameters, overrides=dict(overrides))



def sample_parameters(trial: Trial, resolved: ResolvedParameters) -> dict[str, Scalar]:
    return walk(
        resolved.tree,
        choose=lambda key, node: branch_of(key, node, resolved.overrides),
        fill=lambda key, node, active: _value(
            trial, key, node, resolved.overrides, active=active
        ),
    )


def grid_distributions(
    resolved: ResolvedParameters, *, points: int | None = None
) -> dict[str, CategoricalDistribution]:
    """Every searched parameter as a finite set the grid can walk.

    A list is already one. A range is laid out over ``points`` values, spaced
    the way the declaration says it is searched, and an integer range that has
    fewer values than that gives the values it has. Without ``points`` a range
    is refused, because how finely to cut one is a question the declaration
    does not answer and this should not invent.
    """

    built: dict[str, CategoricalDistribution] = {}
    _grid(resolved.tree, resolved.overrides, built, prefix="", points=points)
    return built



def branch_of(key: str, node: StructureSpec, overrides: dict[str, SpaceEntry]) -> str:
    override = overrides.get(key)
    if override is None:
        return str(node.placeholder)
    return str(override.choices[0])


def _value(
    trial: Trial,
    key: str,
    node: ParameterSpec,
    overrides: dict[str, SpaceEntry],
    *,
    active: bool,
) -> Scalar:
    if not active:
        return node.placeholder
    return _suggest(trial, key, overrides.get(key, node.search))


def _suggest(trial: Trial, key: str, spec: SpaceEntry) -> Scalar:
    if isinstance(spec, FloatSpec):
        return trial.suggest_float(key, spec.low, spec.high, log=spec.log)
    if isinstance(spec, IntSpec):
        return trial.suggest_int(key, spec.low, spec.high, step=spec.step, log=spec.log)
    if isinstance(spec, ChoiceSpec):
        return trial.suggest_categorical(key, list(spec.choices))
    raise SpaceError(f"unsupported space entry for {key}")



def _grid(
    tree: dict[str, ParameterNode],
    overrides: dict[str, SpaceEntry],
    built: dict[str, CategoricalDistribution],
    *,
    prefix: str,
    points: int | None,
) -> None:
    for name, node in tree.items():
        key = f"{prefix}{name}"
        if isinstance(node, StructureSpec):
            branch = branch_of(key, node, overrides)
            _grid(
                node.branches[branch],
                overrides,
                built,
                prefix=f"{key}.{branch}.",
                points=points,
            )
            continue
        spec = overrides.get(key, node.search)
        if isinstance(spec, ChoiceSpec):
            built[key] = CategoricalDistribution(list(spec.choices))
            continue
        if points is None:
            raise SpaceError(
                f"the grid sampler needs a finite set for every parameter, and {key} "
                "is a range; set hpo.points to say how many values to cut it into, "
                "or pin it to a list, or use the tpe or random sampler"
            )
        built[key] = CategoricalDistribution(_cut(spec, points))



def _cut(spec: SpaceEntry, points: int) -> list[Scalar]:
    """A range as evenly spaced values, in the spacing it declares."""

    if points < 1:
        raise SpaceError(f"hpo.points must be positive, not {points}")
    low, high = float(spec.low), float(spec.high)
    if getattr(spec, "log", False):
        drawn = [
            math.exp(
                math.log(low)
                + (math.log(high) - math.log(low)) * index / max(points - 1, 1)
            )
            for index in range(points)
        ]
    else:
        drawn = [low + (high - low) * index / max(points - 1, 1) for index in range(points)]
    if points == 1:
        drawn = [low]
    if isinstance(spec, IntSpec):
        step = spec.step or 1
        rounded = sorted({int(spec.low + round((one - spec.low) / step) * step) for one in drawn})
        return [one for one in rounded if spec.low <= one <= spec.high]
    return drawn


def _check_structure_override(
    key: str, node: StructureSpec, override: SpaceEntry
) -> None:
    if not isinstance(override, ChoiceSpec):
        raise SpaceError(f"{key} is a structure and must be pinned to one branch")
    if len(override.choices) != 1:
        raise SpaceError(
            f"{key} is a structure and is not searched; pin it to one branch, "
            f"not {len(override.choices)}"
        )
    unknown = sorted(str(c) for c in set(override.choices) - set(node.branches))
    if unknown:
        raise SpaceError(
            f"{key} names branches it does not have: {', '.join(unknown)}; "
            f"it has: {', '.join(sorted(node.branches))}"
        )


def _check_override(key: str, node: ParameterSpec, override: SpaceEntry) -> None:
    if isinstance(override, ChoiceSpec):
        for choice in override.choices:
            _inside(key, node.valid, choice)
        return
    if node.value_type == "int" and isinstance(override, FloatSpec):
        raise SpaceError(f"{key} is an int but the experiment gives a float range")
    if override.log and override.low <= 0:
        raise SpaceError(f"{key} log search low must be above zero")
    _inside(key, node.valid, override.low)
    _inside(key, node.valid, override.high)


def _inside(key: str, valid: ValidSpec, value: Scalar) -> None:
    if isinstance(valid, ChoiceSpec):
        if value not in valid.choices:
            raise SpaceError(
                f"{key} value {value!r} is outside its valid choices: "
                f"{', '.join(map(str, valid.choices))}"
            )
        return
    if isinstance(valid, IntValidSpec):
        if type(value) is not int:
            raise SpaceError(f"{key} value {value!r} is not an int")
    elif not isinstance(valid, FloatValidSpec):
        raise SpaceError(f"unsupported valid spec for {key}")
    elif type(value) not in (int, float):
        raise SpaceError(f"{key} value {value!r} is not numeric")
    numeric = float(value)
    if valid.low is not None and numeric < valid.low:
        raise SpaceError(f"{key} value {value!r} is below its valid low {valid.low}")
    if valid.high is not None and numeric > valid.high:
        raise SpaceError(f"{key} value {value!r} is above its valid high {valid.high}")
