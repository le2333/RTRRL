"""Deployment composition for the StreamAC algorithm."""

from __future__ import annotations

import sys

from memorax.algorithms.stream_ac import METRICS as METRICS
from memorax.algorithms.stream_ac import PARAMETERS as PARAMETERS
from memorax.algorithms.stream_ac import StreamAC
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble
from memorax.runtime import Runtime, RuntimeConfig

from ._observability import build_reporter, load_run


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


def runtime_config(config) -> RuntimeConfig:
    """Project the deployment run document onto Runtime's input."""

    return RuntimeConfig(
        total_steps=config.training.total_steps,
        epoch_steps=config.training.epoch_steps,
        eval_steps=config.evaluation.steps,
        num_envs=config.training.num_envs,
        seed=config.environment.seed,
    )


def run(reporter, config) -> None:
    algorithm = assemble(StreamAC, build_request(config))
    Runtime(algorithm=algorithm, config=runtime_config(config)).run(reporter)


def main(argv: list[str] | None = None) -> int:
    del argv
    config, scratch = load_run()
    with build_reporter(config, scratch) as reporter:
        run(reporter, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
