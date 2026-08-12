"""Deployment composition for the StreamAC algorithm."""

from __future__ import annotations

import sys

from memorax.algorithms.stream_ac import METRICS as METRICS
from memorax.algorithms.stream_ac import PARAMETERS as PARAMETERS
from memorax.algorithms.stream_ac import TRAINING_METRICS, StreamAC
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble
from memorax.runtime import Runtime
from worker.reporter import Reporter


def build_request(config) -> BuildRequest:
    """Project the deployment run document onto assembly's input."""

    environment = config.environment
    return BuildRequest(
        parameters=config.params,
        environment=EnvironmentSpec(
            id=environment.id,
            backend=environment.backend,
            observed=environment.observed,
            episode_length=environment.episode_length,
        ),
        num_envs=config.training.num_envs,
    )


def run(reporter, config) -> None:
    program = assemble(StreamAC, build_request(config))
    Runtime.from_config(program, config, series=TRAINING_METRICS).run(reporter)


def main(argv: list[str] | None = None) -> int:
    del argv
    with Reporter.from_env() as reporter:
        run(reporter, reporter.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
