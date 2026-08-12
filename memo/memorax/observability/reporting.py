"""Reduce completed episodes once and fan out each representation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from memorax.runtime.episode import Episode

from .metrics import statistics
from .protocols import EpisodeSink, ScalarSink


class Reporter:
    def __init__(
        self,
        *,
        scalar_sinks: Sequence[ScalarSink] = (),
        episode_sinks: Sequence[EpisodeSink] = (),
    ) -> None:
        self._scalar_sinks = tuple(scalar_sinks)
        self._episode_sinks = tuple(episode_sinks)

    def report(self, step: int, metrics: Mapping[str, float]) -> None:
        for sink in self._scalar_sinks:
            sink.report(step, metrics)

    def log_episode(self, episode: Episode) -> None:
        self.report(episode.end_env_steps, statistics(episode))
        for sink in self._episode_sinks:
            sink.log_episode(episode)

    def close(self) -> None:
        for sink in (*self._scalar_sinks, *self._episode_sinks):
            sink.close()

    def __enter__(self) -> Reporter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
