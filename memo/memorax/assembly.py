"""Build an algorithm graph from resolved parameters and component families.

Assembly knows the mechanics shared by every algorithm: create an environment,
derive its static dimensions, route component requests, and close the resulting
graph over Runtime's three-arrow program.  It deliberately knows no algorithm
names and no graph topology.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from memorax.building import BuildContext, ComponentBuilder
from memorax.environments import make
from memorax.runtime import BuiltAlgorithm, Program


@dataclass(frozen=True)
class EnvironmentSpec:
    id: str
    backend: str | None
    observed: Any
    episode_length: int


@dataclass(frozen=True)
class BuildRequest:
    parameters: Mapping[str, Any]
    environment: EnvironmentSpec
    num_envs: int
    # Which of the observation schema's heavy per-step fields this run needs.
    # Empty is the ordinary case: nothing walks the trajectories, so nothing
    # keeps them.
    record: frozenset[str] = frozenset()


def assemble(
    definition: Any,
    request: BuildRequest,
    *,
    environment_factory: Callable[..., tuple[Any, Any]] = make,
) -> BuiltAlgorithm:
    """Build one graph without interpreting any algorithm-specific relation."""

    specification = request.environment
    environment, environment_parameters = environment_factory(
        specification.id,
        observed=specification.observed,
        backend=specification.backend,
        episode_length=specification.episode_length,
    )
    context = BuildContext(
        environment=environment,
        environment_parameters=environment_parameters,
        observation_space=environment.observation_space(environment_parameters),
        action_space=environment.action_space(environment_parameters),
        num_envs=request.num_envs,
        episode_length=specification.episode_length,
    )
    components = ComponentBuilder(request.parameters, context)
    graph = definition.graph(
        request.parameters, components, context, record=request.record
    )
    return BuiltAlgorithm(
        program=Program(
            init=graph.init,
            train=graph.train,
            evaluate=graph.evaluate,
            interact=graph.interact,
        ),
        observations=definition.observations.recording(request.record),
    )
