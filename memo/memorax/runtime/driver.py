"""Schedule a closed algorithm and report the complete episodes it produces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax

from memorax.runtime.episode import Episode, SampledTrajectory
from memorax.runtime.rollout import complete_episodes
from memorax.runtime.tracker import EpisodeTracker, TrackingResult

from .program import BuiltAlgorithm


class Destination(Protocol):
    """All of the reporter Runtime touches."""

    def log_episode(self, episode: Episode) -> None: ...
    def log_trajectory(self, trajectory: SampledTrajectory) -> None: ...


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
    """Only the scheduling values Runtime consumes.

    ``epoch_steps`` is when to evaluate and report, not how much to ask for at
    once: a train call is bounded by ``max_episode_steps`` so what a chunk costs
    stays the same however long a reporting interval is.  ``sample_steps`` names
    the environment steps whose training episode is to be kept whole.
    """

    total_steps: int
    epoch_steps: int
    eval_steps: int
    num_envs: int
    seed: int
    max_episode_steps: int
    sample_steps: tuple[int, ...] = ()


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
        interact = jax.jit(program.interact)

        key = jax.random.key(config.seed)
        key, init_key = jax.random.split(key)
        state = jax.jit(program.init)(init_key)

        tracker = EpisodeTracker(
            observations=self.algorithm.observations,
            num_envs=config.num_envs,
            max_episode_steps=config.max_episode_steps,
            sample_steps=tuple(config.sample_steps),
        )
        # One chunk holds at most one episode's worth of every stream, so what a
        # train call allocates is a property of the environment rather than of
        # how often the run was asked to report.
        chunk_limit = config.max_episode_steps * config.num_envs
        trained_steps = 0
        eval_number = 1

        for epoch_end in epochs:
            while trained_steps < epoch_end:
                chunk_steps = min(chunk_limit, epoch_end - trained_steps)
                key, chunk_key = jax.random.split(key)
                state, chunk = train(chunk_key, state, chunk_steps)
                self._publish(
                    reporter, tracker.consume(chunk, start_env_steps=trained_steps)
                )
                trained_steps += chunk_steps

            if not config.eval_steps:
                continue
            key, eval_key = jax.random.split(key)
            observations = evaluate(eval_key, state, config.eval_steps)
            eval_number = self._report(
                reporter,
                observations,
                phase="eval",
                start_env_steps=epoch_end,
                stride=0,
                first_number=eval_number,
            )

        key, tail_key = jax.random.split(key)
        self._finish_samples(reporter, tracker, interact, tail_key, state)

    def _finish_samples(
        self,
        reporter: Destination,
        tracker: EpisodeTracker,
        interact,
        key,
        state,
    ) -> None:
        """Carry the last requested episode to its end without learning from it.

        The budget is spent, so these transitions are taken from a copy of the
        final state and marked post-budget.  Nothing here reaches the training
        state, the scalar statistics, or the step count.
        """

        continued = state
        continued_steps = self.config.total_steps
        for _ in range(self.config.max_episode_steps):
            if not tracker.pending_sample_steps:
                return
            key, step_key = jax.random.split(key)
            continued, metrics = interact(step_key, continued)
            self._publish(
                reporter,
                tracker.consume(
                    jax.tree.map(lambda leaf: leaf[None], metrics),
                    start_env_steps=continued_steps,
                    post_budget=True,
                    report_completed=False,
                ),
            )
            continued_steps += self.config.num_envs

        if tracker.pending_sample_steps:
            raise ValueError("sampled episode exceeded maximum episode length")

    def _publish(self, reporter: Destination, result: TrackingResult) -> None:
        for episode in result.completed:
            reporter.log_episode(episode)
        for trajectory in result.sampled:
            reporter.log_trajectory(trajectory)

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
