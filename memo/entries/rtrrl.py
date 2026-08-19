"""Deployment composition for the RTRRL algorithm."""

from __future__ import annotations

import sys

from memorax.algorithms.rtrrl_aaai import METRICS as METRICS
from memorax.algorithms.rtrrl_aaai import OBSERVATIONS
from memorax.algorithms.rtrrl_aaai import PARAMETERS as PARAMETERS
from memorax.algorithms.rtrrl_aaai import RTRRL
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble
from memorax.runtime import Runtime, RuntimeConfig

from ._checkpoint import checkpoint_directory, checkpoint_every_steps, resume
from ._observability import build_reporter, load_run
from ._schedule import trajectory_at_steps, trajectory_record


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

    training = config.training
    return RuntimeConfig(
        total_steps=training.total_steps,
        chunk_steps=training.chunk_steps,
        max_episode_steps=config.algorithm.environment.episode_length,
        evaluate_every_steps=config.evaluation.every_steps,
        rollout_steps=config.evaluation.rollout_steps,
        num_envs=config.algorithm.num_envs,
        seed=training.seed,
        trajectory_at_steps=trajectory_at_steps(config),
        checkpoint_every_steps=checkpoint_every_steps(config),
    )


def run(reporter, config, scratch) -> None:
    """Assemble, restore if this is a branch, and run to the declared budget.

    ``scratch`` is required rather than defaulted because it is where the
    checkpoints go: a caller that did not have one would run a document asking
    for checkpoints and quietly file none, and the run it was the parent of
    would have nothing to fork from.
    """

    algorithm = assemble(RTRRL, build_request(config))
    Runtime(
        algorithm=algorithm,
        config=runtime_config(config),
        checkpoints=checkpoint_directory(config, scratch),
    ).run(reporter, resume=resume(config, algorithm.program))


def main(argv: list[str] | None = None) -> int:
    del argv
    config, scratch = load_run()
    with build_reporter(config, scratch) as reporter:
        run(reporter, config, scratch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
