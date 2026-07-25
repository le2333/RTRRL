from __future__ import annotations

from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from training_sdk.contract import ChoiceSpec, EntryDescriptor, FloatSpec, IntSpec, SpaceEntry

TOTAL_STEPS = "total_steps"


class SpaceError(ValueError):
    """The resolved search space is not usable."""


def resolve_space(
    entry: EntryDescriptor, overrides: dict[str, SpaceEntry]
) -> dict[str, SpaceEntry]:
    unknown = sorted(set(overrides) - set(entry.space))
    if unknown:
        declared = ", ".join(sorted(entry.space))
        raise SpaceError(
            f"experiment declares parameters the entry does not accept: "
            f"{', '.join(unknown)}; entry declares: {declared}"
        )
    resolved = dict(entry.space) | dict(overrides)
    if TOTAL_STEPS not in resolved:
        raise SpaceError(f"entry must declare the reserved parameter {TOTAL_STEPS}")
    return resolved


def distributions(space: dict[str, SpaceEntry]) -> dict[str, BaseDistribution]:
    built: dict[str, BaseDistribution] = {}
    for key, spec in space.items():
        if isinstance(spec, ChoiceSpec):
            built[key] = CategoricalDistribution(choices=list(spec.choices))
        elif isinstance(spec, IntSpec):
            built[key] = IntDistribution(
                low=spec.low, high=spec.high, step=spec.step, log=spec.log
            )
        elif isinstance(spec, FloatSpec):
            built[key] = FloatDistribution(low=spec.low, high=spec.high, log=spec.log)
        else:  # pragma: no cover - the union is closed
            raise SpaceError(f"unsupported space entry for {key}")
    return built


def minimum_total_steps(space: dict[str, SpaceEntry]) -> int:
    spec = space[TOTAL_STEPS]
    if isinstance(spec, ChoiceSpec):
        values = [value for value in spec.choices if type(value) is int]
        if len(values) != len(spec.choices):
            raise SpaceError(f"{TOTAL_STEPS} choices must all be integers")
        return min(values)
    if isinstance(spec, IntSpec):
        return spec.low
    raise SpaceError(f"{TOTAL_STEPS} must be an integer range or an integer choice list")
