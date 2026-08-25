"""Deployment composition for the StreamAC algorithm."""

from __future__ import annotations

import sys

from memorax.algorithms.stream_ac import METRICS as METRICS
from memorax.algorithms.stream_ac import OBSERVATIONS
from memorax.algorithms.stream_ac import PARAMETERS as PARAMETERS
from memorax.algorithms.stream_ac import StreamAC
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble
from memorax.runtime import Runtime, RuntimeConfig

from ._observability import build_reporter, load_run
from ._schedule import trajectory_at_steps, trajectory_record
from ._snapshot import resuming


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
            kwargs=environment.kwargs,
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
        evaluation_episodes=config.evaluation.episodes,
        evaluation_chunk_steps=config.evaluation.chunk_steps,
        evaluation_seed=config.evaluation.seed,
        num_envs=config.algorithm.num_envs,
        seed=training.seed,
        trajectory_at_steps=trajectory_at_steps(config),
        snapshot_every_steps=training.snapshot_every_steps,
    )


def run(reporter, config, snapshots=None) -> None:
    algorithm = assemble(StreamAC, build_request(config))
    Runtime(
        algorithm=algorithm, config=runtime_config(config), snapshots=snapshots
    ).run(reporter)


def main(argv: list[str] | None = None) -> int:
    del argv
    config, scratch = load_run()
    # Outside the reporter, and first: the archive brings back the artifact
    # directory the reporter is about to append to, and a sink that had
    # already opened a file would be holding the one this replaces.
    with resuming([(config, scratch)]) as snapshots:
        with build_reporter(config, scratch) as reporter:
            run(reporter, config, snapshots)
    return 0


if __name__ == "__main__":
    sys.exit(main())
