"""Small assembly adapters shared by tests that start a real algorithm graph."""

from types import MethodType
from typing import Any

from memorax.algorithms.r2d2 import R2D2
from memorax.algorithms.stream_ac import StreamAC
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble
from memorax.runtime import BuiltAlgorithm


def graph_of(built: BuiltAlgorithm) -> Any:
    """The algorithm object an assembled program's arrows are bound to.

    ``Program`` is four callables and says nothing about where they came from.
    A test that wants to see what assembly selected reaches back through one of
    them, and this is the one place that says the arrows are bound methods.
    """

    init = built.program.init
    assert isinstance(init, MethodType), "program.init is not a bound method"
    return init.__self__


def assemble_r2d2(parameters, environment, *, num_envs):
    return assemble(
        R2D2,
        BuildRequest(
            parameters=parameters,
            environment=EnvironmentSpec(
                id=environment.id,
                backend=environment.backend,
                observed=environment.observed,
                episode_length=environment.episode_length,
            ),
            num_envs=num_envs,
        ),
    ).program


def assemble_stream_ac(parameters, environment, *, num_envs):
    return assemble(
        StreamAC,
        BuildRequest(
            parameters=parameters,
            environment=EnvironmentSpec(
                id=environment.id,
                backend=environment.backend,
                observed=environment.observed,
                episode_length=environment.episode_length,
            ),
            num_envs=num_envs,
        ),
    ).program
