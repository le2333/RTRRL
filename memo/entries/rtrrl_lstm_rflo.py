"""Deployment composition for RTRRL over an LSTM torso credited by RFLO.

A separate entry from ``rtrrl`` and from ``rtrrl_ctrnn_rflo`` rather than a
value either could be given: a different recurrence, a different trace, and --
against ``drqn``, which is where an LSTM appears in this repository today -- a
different way of getting a gradient out of it altogether. Its parameter schema
says so, so a run document cannot be moved between the entries by editing the
entry name. See ``memorax.algorithms.rtrrl_lstm_rflo``.
"""

from __future__ import annotations

import sys

from memorax.algorithms.rtrrl_lstm_rflo import METRICS as METRICS
from memorax.algorithms.rtrrl_lstm_rflo import OBSERVATIONS
from memorax.algorithms.rtrrl_lstm_rflo import PARAMETERS as PARAMETERS
from memorax.algorithms.rtrrl_lstm_rflo import RTRRLLstmRflo
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble
from memorax.runtime import Runtime, RuntimeConfig

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
    )


def run(reporter, config) -> None:
    algorithm = assemble(RTRRLLstmRflo, build_request(config))
    Runtime(algorithm=algorithm, config=runtime_config(config)).run(reporter)


def main(argv: list[str] | None = None) -> int:
    del argv
    config, scratch = load_run()
    with build_reporter(config, scratch) as reporter:
        run(reporter, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
