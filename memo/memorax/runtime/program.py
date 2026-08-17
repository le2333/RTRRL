"""The closed execution surface Runtime accepts from an algorithm build."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class Program:
    """The five arrows Runtime schedules, as the graph's own methods.

    Not compiled. ``assemble`` puts the graph's bound methods here and
    ``Driver`` is what wraps them in ``jax.jit``, with ``num_steps`` static
    because it is a scan length. Anything that reaches past Runtime and calls
    these directly -- a script, a benchmark -- gets a fresh trace and a fresh
    compilation of the whole scan on every call, which is slow, holds the
    compiled artefacts of every call, and makes any timing a measurement of
    the caller rather than of the algorithm.

    ``interact`` is one vectorized behavior-policy transition that learns
    nothing: Runtime schedules it only to finish a sampled episode that the
    training budget cut short.

    Measuring the policy is two arrows rather than one because a checkpoint is
    scored on a number of *episodes* and a scan is a number of *steps*: how
    long the requested episodes take is not known until they end.
    ``open_evaluation`` opens one rollout on the trained parameters, and
    ``evaluate`` advances that rollout by a bounded number of steps and hands
    it back, so Runtime can keep asking until the episodes it needs have
    completed. The state it passes back and forth is the evaluation's own; a
    caller that has one cannot reach the training state through it, which is
    what keeps a measurement from becoming an update.
    """

    init: Callable[..., Any]
    train: Callable[..., Any]
    open_evaluation: Callable[..., Any]
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
