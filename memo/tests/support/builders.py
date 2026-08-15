"""Small assembly adapters shared by tests that start a real algorithm graph."""

from memorax.algorithms.r2d2 import R2D2
from memorax.algorithms.stream_ac import StreamAC
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble


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
