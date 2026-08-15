"""The closed execution surface Runtime accepts from an algorithm build."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class Program:
    """The four compiled arrows Runtime schedules.

    ``interact`` is one vectorized behavior-policy transition that learns
    nothing: Runtime schedules it only to finish a sampled episode that the
    training budget cut short.
    """

    init: Callable[..., Any]
    train: Callable[..., Any]
    evaluate: Callable[..., Any]
    interact: Callable[..., Any]


@dataclass(frozen=True)
class ObservationSchema:
    """Paths Runtime reads from one algorithm's stacked step observations.

    Transition identity is explicit because Runtime owns episode cutting but
    must not know how an algorithm groups its readings.  Trajectory paths are
    optional observations: a graph may omit their values when no configured
    sink needs a walk without changing the execution contract.
    """

    reward: str
    done: str
    terminal: str | None
    series: tuple[str, ...] = ()
    observation: str | None = None
    next_observation: str | None = None
    action: str | None = None

    @property
    def episode_fields(self) -> frozenset[str]:
        """What cutting an episode and reducing it to scalars always needs."""

        return frozenset(
            (self.reward, self.done, *self.series)
            + ((self.terminal,) if self.terminal is not None else ())
        )

    @property
    def trajectory_fields(self) -> frozenset[str]:
        """The per-step walk, which only a trajectory backend asks for."""

        return frozenset(
            path
            for path in (self.observation, self.next_observation, self.action)
            if path is not None
        )

    def recording(self, record: frozenset[str]) -> ObservationSchema:
        """This schema as the graph will answer it, having been asked for ``record``.

        A run with no trajectory backend does not pay for the walk, so its
        graph leaves those fields empty and its schema must say so rather than
        name paths nothing will fill.
        """

        wanted = self.trajectory_fields
        if wanted <= record:
            return self
        if record & wanted:
            raise ValueError(
                "trajectory fields must be requested together or not at all"
            )
        return replace(self, observation=None, next_observation=None, action=None)


@dataclass(frozen=True)
class BuiltAlgorithm:
    """One resolved graph together with the readings Runtime may consume."""

    program: Program
    observations: ObservationSchema
