"""Deployment composition for the RTRRL algorithm."""

from __future__ import annotations

import sys

from memorax.algorithms.rtrrl_aaai import METRICS as METRICS
from memorax.algorithms.rtrrl_aaai import OBSERVATIONS
from memorax.algorithms.rtrrl_aaai import PARAMETERS as PARAMETERS
from memorax.algorithms.rtrrl_aaai import RTRRL
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble
from memorax.runtime import Runtime, RuntimeConfig

from ._observability import build_reporter, load_run
from ._schedule import sample_steps, trajectory_record


def build_request(config) -> BuildRequest:
    """Project the deployment run document onto assembly's input."""

    algorithm = config.algorithm
    environment = algorithm.environment
    return BuildRequest(
        parameters=algorithm.parameters,
        environment=EnvironmentSpec(
            id=environment.id,
            backend=environment.backend,
            observed=environment.observed,
            episode_length=environment.episode_length,
        ),
        num_envs=algorithm.num_envs,
        record=trajectory_record(config, OBSERVATIONS),
    )


def runtime_config(config) -> RuntimeConfig:
    """Project the deployment run document onto Runtime's input."""

    runtime = config.runtime
    return RuntimeConfig(
        total_steps=runtime.total_steps,
        epoch_steps=runtime.epoch_steps,
        eval_steps=runtime.evaluation_steps,
        num_envs=config.algorithm.num_envs,
        seed=runtime.seed,
        max_episode_steps=config.algorithm.environment.episode_length,
        sample_steps=sample_steps(config),
    )


def run(reporter, config) -> None:
    algorithm = assemble(RTRRL, build_request(config))
    Runtime(algorithm=algorithm, config=runtime_config(config)).run(reporter)


def main(argv: list[str] | None = None) -> int:
    del argv
    config, scratch = load_run()
    with build_reporter(config, scratch) as reporter:
        run(reporter, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
