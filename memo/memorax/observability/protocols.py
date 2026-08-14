"""The three observation shapes a backend can consume."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from memorax.runtime.episode import Episode, SampledTrajectory


class ScalarSink(Protocol):
    def report(self, step: int, metrics: Mapping[str, float]) -> None: ...
    def close(self) -> None: ...


class EpisodeSink(Protocol):
    def log_episode(self, episode: Episode) -> None: ...
    def close(self) -> None: ...


class TrajectorySink(Protocol):
    """A backend for the few episodes Runtime was asked to keep whole.

    Every completed episode reaches an ``EpisodeSink``; only a requested sample
    reaches this one, so a backend here never decides what to keep.
    """

    def log_trajectory(self, trajectory: SampledTrajectory) -> None: ...
    def close(self) -> None: ...
