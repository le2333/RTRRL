"""Algorithm-neutral contracts for selecting and constructing components."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from memorax.parameters import read_branch


@dataclass(frozen=True)
class BuildContext:
    """The static facts about a run that a graph reads but does not choose.

    ``episode_length`` is the run document's declared horizon rather than
    whatever the environment's own parameters happen to call it: a graph whose
    shape depends on it -- a full-episode replay window -- needs it before any
    environment has stepped, and each adapter spells its own field differently.
    """

    environment: Any
    environment_parameters: Any
    observation_space: Any
    action_space: Any
    num_envs: int
    episode_length: int


@dataclass(frozen=True)
class ComponentSelection:
    """One selected leaf and the path at which its own children live."""

    kind: str
    parameters: Any
    path: str


@dataclass(frozen=True)
class ComponentFamily:
    """A component choice, its parameter declarations, and its constructor."""

    branches: Mapping[str, Any]
    construct: Callable[..., Any]

    def restricted(self, *names: str) -> ComponentFamily:
        return ComponentFamily(
            branches={name: self.branches[name] for name in names},
            construct=self.construct,
        )


class ComponentBuilder:
    """Route resolved paths to families; families decide what a branch builds."""

    def __init__(self, parameters: Mapping[str, Any], context: BuildContext) -> None:
        self.parameters = parameters
        self.context = context

    def build(self, family: ComponentFamily, path: str, **inputs: Any) -> Any:
        parent, _, field = path.rpartition(".")
        prefix = f"{parent}." if parent else ""
        kind, parameters = read_branch(
            self.parameters,
            field,
            family.branches,
            prefix=prefix,
        )
        selection = ComponentSelection(
            kind=kind,
            parameters=parameters,
            path=f"{path}.{kind}",
        )
        return family.construct(selection, self, **inputs)
