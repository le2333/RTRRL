"""Schedule a closed algorithm and report the complete episodes it produces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax

from memorax.runtime.episode import Episode
from memorax.runtime.rollout import complete_episodes

from .program import BuiltAlgorithm


class Destination(Protocol):
    """All of the reporter Runtime touches."""

    def log_episode(self, episode: Episode) -> None: ...


def whole_epochs(*, total_steps: int, epoch_steps: int, num_envs: int) -> range:
    """Return every epoch boundary, refusing a budget that is ragged."""

    if epoch_steps % num_envs:
        raise ValueError(f"epoch_steps {epoch_steps} is not {num_envs} streams' worth")
    if total_steps % epoch_steps:
        raise ValueError(
            f"total_steps {total_steps} is not whole epochs of {epoch_steps}"
        )
    return range(epoch_steps, total_steps + 1, epoch_steps)


@dataclass(frozen=True)
class RuntimeConfig:
    """Only the scheduling values Runtime consumes."""

    total_steps: int
    epoch_steps: int
    eval_steps: int
    num_envs: int
    seed: int


@dataclass(frozen=True)
class Runtime:
    """One run: a closed algorithm, an execution budget, and episode output."""

    algorithm: BuiltAlgorithm
    config: RuntimeConfig

    def run(self, reporter: Destination) -> None:
        """Run the algorithm to its budget, reporting every complete episode."""

        config = self.config
        epochs = whole_epochs(
            total_steps=config.total_steps,
            epoch_steps=config.epoch_steps,
            num_envs=config.num_envs,
        )

        program = self.algorithm.program
        train = jax.jit(program.train, static_argnums=2)
        evaluate = jax.jit(program.evaluate, static_argnums=2)

        key = jax.random.key(config.seed)
        key, init_key = jax.random.split(key)
        state = jax.jit(program.init)(init_key)

        numbers = {"train": 1, "eval": 1}
        for done in epochs:
            key, epoch_key = jax.random.split(key)
            state, observations = train(epoch_key, state, config.epoch_steps)
            numbers["train"] = self._report(
                reporter,
                observations,
                phase="train",
                start_env_steps=done - config.epoch_steps,
                stride=config.num_envs,
                first_number=numbers["train"],
            )

            if not config.eval_steps:
                continue
            key, eval_key = jax.random.split(key)
            observations = evaluate(eval_key, state, config.eval_steps)
            numbers["eval"] = self._report(
                reporter,
                observations,
                phase="eval",
                start_env_steps=done,
                stride=0,
                first_number=numbers["eval"],
            )

    def _report(self, reporter: Destination, chunk, **cut) -> int:
        number = cut["first_number"]
        for episode in complete_episodes(
            chunk,
            observations=self.algorithm.observations,
            num_envs=self.config.num_envs,
            **cut,
        ):
            reporter.log_episode(episode)
            number = episode.number + 1
        return number
