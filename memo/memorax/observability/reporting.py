"""Reduce completed episodes once and fan out each representation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from memorax.runtime.episode import Episode, SampledTrajectory
from memorax.runtime.snapshot import resume as resume_subject
from memorax.runtime.snapshot import suspend as suspend_subject

from .metrics import statistics
from .protocols import EpisodeSink, ScalarSink, TrajectorySink
from .scopes import Reading, Scope


class Reporter:
    """Two scalar destinations, because they answer different questions.

    ``scalar_sinks`` receive every episode's own reduction. That is the run's
    record, and it is what a question asked afterwards is answered from.

    ``sampled_sinks`` receive every evaluation -- it is what the run is scored
    on -- and whatever ``training_scopes`` the run asked for. A run ends an
    episode every few dozen steps before its policy is any good; a dashboard
    given all of them is given tens of millions of points that draw as a solid
    band, at a cost in memory and throughput that the run cannot afford. No
    scope at all sends it no training, only evaluation.
    """

    def __init__(
        self,
        *,
        scalar_sinks: Sequence[ScalarSink] = (),
        sampled_sinks: Sequence[ScalarSink] = (),
        episode_sinks: Sequence[EpisodeSink] = (),
        trajectory_sinks: Sequence[TrajectorySink] = (),
        training_scopes: Sequence[Scope] = (),
    ) -> None:
        self._scalar_sinks = tuple(scalar_sinks)
        self._sampled_sinks = tuple(sampled_sinks)
        self._episode_sinks = tuple(episode_sinks)
        self._trajectory_sinks = tuple(trajectory_sinks)
        self._training_scopes = tuple(training_scopes)

    def report(self, step: int, metrics: Mapping[str, float]) -> None:
        for sink in (*self._scalar_sinks, *self._sampled_sinks):
            sink.report(step, metrics)

    def log_episode(self, episode: Episode) -> None:
        values = statistics(episode)
        for sink in self._scalar_sinks:
            sink.report(episode.end_env_steps, values)
        self._sample(self._due(episode, values))
        for sink in self._episode_sinks:
            sink.log_episode(episode)

    def _due(
        self, episode: Episode, values: Mapping[str, float]
    ) -> tuple[Reading, ...]:
        """What this episode brought due on the dashboard.

        An evaluation is the measurement the run is scored on and there are as
        many of them as the run asked for, so it arrives whole and on no
        schedule. A training episode arrives only through a scope, which says
        in its own name what its number was reduced over.
        """

        if episode.phase != "train":
            return ((episode.end_env_steps, values),)
        return tuple(
            reading
            for scope in self._training_scopes
            for reading in scope.take(episode)
        )

    def _sample(self, readings: tuple[Reading, ...]) -> None:
        for step, values in readings:
            for sink in self._sampled_sinks:
                sink.report(step, values)

    def log_trajectory(self, trajectory: SampledTrajectory) -> None:
        """Fan out one requested walk.

        Its episode was reduced when it completed, so nothing is reduced here.
        """

        for sink in self._trajectory_sinks:
            sink.log_trajectory(trajectory)

    def _carried(self) -> tuple[tuple[str, tuple[object, ...]], ...]:
        """Everything under this reporter that a run leaves mid-way through."""

        return (
            ("scalar", self._scalar_sinks),
            ("sampled", self._sampled_sinks),
            ("episode", self._episode_sinks),
            ("trajectory", self._trajectory_sinks),
            ("scopes", self._training_scopes),
        )

    def suspend(self) -> dict[str, tuple[Any, ...]]:
        """What each destination and scope had already made of the run.

        Position is the identity here: a reporter is rebuilt from the same run
        document, so its third scalar sink is the same backend as before. That
        is checked on the way back in rather than assumed, because a document
        that changed between the two processes would otherwise hand one sink's
        state to another.
        """

        return {
            name: tuple(suspend_subject(subject) for subject in subjects)
            for name, subjects in self._carried()
        }

    def resume(self, state: Mapping[str, Sequence[Any]]) -> None:
        """Put the run's own record back before a single row is added to it."""

        for name, subjects in self._carried():
            carried = state.get(name, ())
            if len(carried) != len(subjects):
                raise ValueError(
                    f"the snapshot holds {len(carried)} {name} destinations and "
                    f"this run has {len(subjects)}"
                )
            for subject, one in zip(subjects, carried, strict=True):
                resume_subject(subject, one)

    def close(self) -> None:
        for scope in self._training_scopes:
            self._sample(scope.close())
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
