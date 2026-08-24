"""Build an algorithm graph from resolved parameters and component families.

Assembly knows the mechanics shared by every algorithm: create an environment,
derive its static dimensions, route component requests, and close the resulting
graph over Runtime's three-arrow program.  It deliberately knows no algorithm
names and no graph topology.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from memorax.building import BuildContext, ComponentBuilder
from memorax.environments import make
from memorax.runtime import BuiltAlgorithm, Program


@dataclass(frozen=True)
class EnvironmentSpec:
    """What a build says about the environment, in assembly's own vocabulary.

    ``kwargs`` reaches the namespace adapter's ``make`` unchanged. It carries
    the arguments the environment is *constructed* with, which for some tasks
    is what says which task it is; the other three say what the deployment
    does to whatever was constructed.
    """

    id: str
    backend: str | None
    observed: Any
    episode_length: int
    # Typed as a mapping so a specification does not advertise a dictionary
    # its holder may write into, and defaulted from a factory so two builds
    # that name no arguments do not share one.
    kwargs: Mapping[str, Any] = field(default_factory=dict)


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
        **specification.kwargs,
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
            open_evaluation=graph.open_evaluation,
            evaluate=graph.evaluate,
            interact=graph.interact,
        ),
        # The built graph's schema rather than the class's. They are the same
        # object for a graph whose readings do not depend on what it selected;
        # where they do -- an optimizer that carries state worth reading and
        # one that does not -- only the built graph knows which names are going
        # to arrive, and a schema naming the others fails the run on a series
        # that was never going to exist.
        observations=graph.observations.recording(request.record),
    )
