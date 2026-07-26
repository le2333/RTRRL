"""Run one training job: build from the run config, train, report, evaluate.

This is the whole of what the infrastructure knows about algorithms and what
algorithms know about the infrastructure. The loop below never asks which
topology it is driving; it reads ``AgentProgram`` and the shape of what comes
back, so a new algorithm reaches Aim and Rerun by being registered.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from typing import Any, Protocol

import jax
from training_sdk.episode import Episode
from training_sdk.reporter import Reporter

from .episodes import complete_episodes
from .metrics import scalar_metrics
from .registry import topology


class Destination(Protocol):
    """All of the reporter that the loop below touches."""

    def report(self, step: int, metrics: Mapping[str, float]) -> None: ...

    def log_episode(self, episode: Episode) -> None: ...


def _int_param(params: Mapping[str, Any], name: str, default: int) -> int:
    value = params.get(name, default)
    return int(value)


def run(reporter: Destination, entry: str, params: Mapping[str, Any]) -> None:
    """Train to the step budget, reporting once per epoch."""

    program = topology(entry).build(params)

    num_envs = _int_param(params, "num_envs", 1)
    total_steps = _int_param(params, "total_steps", 1024)
    epoch_steps = _int_param(params, "epoch_steps", max(num_envs, total_steps // 16))
    epoch_steps = max(epoch_steps - epoch_steps % num_envs, num_envs)
    eval_steps = _int_param(params, "eval_steps", 0)

    train_epoch = jax.jit(program.train_epoch_fn, static_argnums=2)
    evaluate = jax.jit(program.evaluate_fn, static_argnums=2)

    key = jax.random.key(_int_param(params, "seed", 0))
    key, init_key = jax.random.split(key)
    state = jax.jit(program.init_fn)(init_key)

    episode_number = 1
    completed = 0
    while completed < total_steps:
        steps = min(epoch_steps, total_steps - completed)
        steps -= steps % num_envs
        if steps <= 0:
            break
        key, epoch_key = jax.random.split(key)
        state, metrics = train_epoch(epoch_key, state, steps)
        completed += steps

        report = scalar_metrics(metrics, steps=steps // num_envs, prefix="train/")
        reporter.report(completed, report)

        if not eval_steps:
            continue

        key, eval_key = jax.random.split(key)
        _, summary = evaluate(eval_key, state, eval_steps)
        report = scalar_metrics(summary, steps=eval_steps, prefix="eval/")

        returns: list[float] = []
        lengths: list[int] = []
        for episode in complete_episodes(
            summary,
            phase="eval",
            start_env_steps=completed,
            num_envs=num_envs,
            first_number=episode_number,
        ):
            reporter.log_episode(episode)
            episode_number = episode.number + 1
            returns.append(float(sum(episode.rewards)))
            lengths.append(len(episode.actions))

        # Only report an episode summary when a whole episode was seen. A mean
        # over nothing would be reported as zero and read as a score.
        if returns:
            report["eval/episode_return"] = sum(returns) / len(returns)
            report["eval/episode_length"] = sum(lengths) / len(lengths)
        reporter.report(completed, report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    with Reporter.from_env() as reporter:
        run(reporter, reporter.config.entry, reporter.config.params)
    return 0


if __name__ == "__main__":
    sys.exit(main())
