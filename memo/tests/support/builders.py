"""Small assembly adapters shared by tests that start a real algorithm graph."""

from memorax.algorithms.stream_ac import StreamAC
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble


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
