"""Reduce completed episodes once and fan out each representation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from memorax.runtime.episode import Episode, SampledTrajectory

from .metrics import statistics
from .protocols import EpisodeSink, ScalarSink, TrajectorySink


class Reporter:
    """Two scalar destinations, because they answer different questions.

    ``scalar_sinks`` receive every reduction. That is the run's record, and it
    is what a question asked afterwards is answered from.

    ``sampled_sinks`` receive every evaluation and only every so many steps of
    training. A run ends an episode every few dozen steps before its policy is
    any good; a dashboard given all of them is given tens of millions of points
    that draw as a solid band, at a cost in memory and throughput that the run
    cannot afford. ``training_every_steps`` of zero sends it no training at all.
    """

    def __init__(
        self,
        *,
        scalar_sinks: Sequence[ScalarSink] = (),
        sampled_sinks: Sequence[ScalarSink] = (),
        episode_sinks: Sequence[EpisodeSink] = (),
        trajectory_sinks: Sequence[TrajectorySink] = (),
        training_every_steps: int = 0,
    ) -> None:
        self._scalar_sinks = tuple(scalar_sinks)
        self._sampled_sinks = tuple(sampled_sinks)
        self._episode_sinks = tuple(episode_sinks)
        self._trajectory_sinks = tuple(trajectory_sinks)
        self._training_every_steps = training_every_steps
        self._next_training_step = training_every_steps

    def report(self, step: int, metrics: Mapping[str, float]) -> None:
        for sink in (*self._scalar_sinks, *self._sampled_sinks):
            sink.report(step, metrics)

    def log_episode(self, episode: Episode) -> None:
        values = statistics(episode)
        step = episode.end_env_steps
        for sink in self._scalar_sinks:
            sink.report(step, values)
        if self._due(episode.phase, step):
            for sink in self._sampled_sinks:
                sink.report(step, values)
        for sink in self._episode_sinks:
            sink.log_episode(episode)

    def _due(self, phase: str, step: int) -> bool:
        """Every evaluation, and the first training episode past each mark."""

        if phase != "train":
            return True
        if not self._training_every_steps or step < self._next_training_step:
            return False
        # Skip whole marks rather than firing repeatedly for one long episode.
        passed = step - self._next_training_step
        self._next_training_step += self._training_every_steps * (
            passed // self._training_every_steps + 1
        )
        return True

    def log_trajectory(self, trajectory: SampledTrajectory) -> None:
        """Fan out one requested walk.

        Its episode was reduced when it completed, so nothing is reduced here.
        """

        for sink in self._trajectory_sinks:
            sink.log_trajectory(trajectory)

    def close(self) -> None:
        for sink in (
            *self._scalar_sinks,
            *self._sampled_sinks,
            *self._episode_sinks,
            *self._trajectory_sinks,
        ):
            sink.close()

    def __enter__(self) -> Reporter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
