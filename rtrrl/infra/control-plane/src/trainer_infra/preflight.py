from __future__ import annotations

from dataclasses import dataclass

from training_sdk.contract import CONTRACT_VERSION, Catalog, ChoiceSpec, EntryDescriptor
from training_sdk.contract import SpaceEntry

from trainer_infra.experiment import Experiment
from trainer_infra.space import minimum_total_steps, resolve_space


class PreflightError(ValueError):
    """A launch precondition failed before anything was spent."""


@dataclass(frozen=True)
class LaunchPlan:
    experiment: Experiment
    entry_name: str
    entry: EntryDescriptor
    space: dict[str, SpaceEntry]
    digest: str
    queue: str
    job_definition: str


def check_offline(experiment: Experiment, catalog: Catalog) -> dict[str, SpaceEntry]:
    if catalog.contract != CONTRACT_VERSION:
        raise PreflightError(
            f"image declares contract {catalog.contract}; "
            f"this control plane implements contract {CONTRACT_VERSION}"
        )
    entry = catalog.entries.get(experiment.entry)
    if entry is None:
        available = ", ".join(sorted(catalog.entries))
        raise PreflightError(
            f"image does not declare entry {experiment.entry!r}; available: {available}"
        )
    if experiment.score.metric not in entry.metrics:
        reported = ", ".join(entry.metrics)
        raise PreflightError(
            f"entry {experiment.entry} does not report metric "
            f"{experiment.score.metric!r}; it reports: {reported}"
        )
    space = resolve_space(entry, experiment.space)
    budget = minimum_total_steps(space)
    if experiment.score.window_steps[1] > budget:
        raise PreflightError(
            f"score window upper bound {experiment.score.window_steps[1]} exceeds the "
            f"smallest total_steps the space can produce ({budget})"
        )
    return space


def format_space(space: dict[str, SpaceEntry]) -> str:
    lines = []
    for key in sorted(space):
        spec = space[key]
        if isinstance(spec, ChoiceSpec):
            rendered = " | ".join(repr(choice) for choice in spec.choices)
        else:
            rendered = spec.model_dump_json()
        lines.append(f"  {key}: {rendered}")
    return "resolved search space:\n" + "\n".join(lines)
