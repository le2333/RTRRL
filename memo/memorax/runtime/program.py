"""The closed execution surface Runtime accepts from an algorithm build."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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
    def required_fields(self) -> frozenset[str]:
        """Fields the graph must retain for cutting and declared reductions."""

        return frozenset(
            (self.reward, self.done, *self.series)
            + ((self.terminal,) if self.terminal is not None else ())
        )


@dataclass(frozen=True)
class BuiltAlgorithm:
    """One resolved graph together with the readings Runtime may consume."""

    program: Program
    observations: ObservationSchema
