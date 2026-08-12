"""A running estimate of a mean and a spread, held by whoever needs one.

Normalising an observation and normalising a reward are the same estimator with
two settings. An observation is centred; a reward is only scaled, because
subtracting a mean from it would move the point at which the agent is
indifferent. And a reward is fed the discounted accumulation of itself rather
than its own value, so that its spread is a return's rather than a step's.

Written as two named halves this component said where it was called from, and a
kernel wanting a third stream had nowhere to put it. The kernel names its
streams, being the thing that holds them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax.numpy as jnp
from flax import struct

from memorax.building import ComponentFamily
from memorax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
)
from memorax.parameters import param

COLD_STARTS = ("seeded", "first_sample")
VARIANCES = ("population", "sample")


@dataclass(frozen=True)
class RunningNormalization:
    center: bool = param(valid=[False, True], search=[True])
    cold_start: str = param(valid=list(COLD_STARTS), search=list(COLD_STARTS))
    variance: str = param(valid=list(VARIANCES), search=list(VARIANCES))
    eps: float = param(valid=(1e-12, 1e-2), search=[1e-8], log=True)
    reset_on_start: bool = param(valid=[False, True], search=[False])
    update_during_eval: bool = param(valid=[False, True], search=[True])


@dataclass(frozen=True)
class DiscountedNormalization(RunningNormalization):
    center: bool = param(valid=[False, True], search=[False])
    reset_on_done: bool = param(valid=[False, True], search=[True])


NORMALIZATION_BRANCHES = {"none": (), "running": RunningNormalization}
DISCOUNTED_NORMALIZATION_BRANCHES = {"none": (), "running": DiscountedNormalization}


@dataclass(frozen=True)
class NormalizationConfig:
    """What one estimator reads.

    ``discount`` is the algorithm's, not a setting of its own: the bootstrap and
    this accumulation both read it and have to agree, and there is no way to say
    that between two components. Absent, the estimator sees each value.
    """

    center: bool = True
    cold_start: str = "seeded"
    variance: str = "population"
    eps: float = 1e-8
    reset_on_start: bool = True
    update_during_eval: bool = True
    discount: float | None = None
    reset_on_done: bool = True


@struct.dataclass
class Statistics:
    mean: Any
    M2: Any
    count: Any
    trace: Any = None


@struct.dataclass
class NormalizationMetrics:
    mean: Any = None
    std: Any = None


def _expand_for(value, target):
    return value[(slice(None),) + (None,) * (target.ndim - value.ndim)]


# ------------------------------------------------------------- the two pieces
class Accumulation:
    """A value and everything discounted that came before it."""

    def __init__(self, discount: float, reset_on_done: bool) -> None:
        self.discount = discount
        self.reset_on_done = reset_on_done

    def initial(self, statistics: Statistics, streams: int) -> Statistics:
        return replace(statistics, trace=jnp.zeros((streams,), dtype=jnp.float32))

    def dropped(self, statistics: Statistics) -> Statistics:
        return replace(statistics, trace=jnp.zeros_like(statistics.trace))

    def advance(self, statistics: Statistics, sample, done):
        counted = sample + self.discount * statistics.trace * (1 - done)
        return counted, replace(
            statistics,
            trace=counted * (1 - done) if self.reset_on_done else counted,
        )


class Spread:
    """A running mean and second moment, and what they scale a value by."""

    def __init__(self, config: NormalizationConfig) -> None:
        self.config = config

    def initial(self, sample) -> Statistics:
        streams = sample.shape[0]
        seeded = self.config.cold_start == "seeded"
        return Statistics(
            mean=jnp.zeros_like(sample),
            M2=jnp.ones_like(sample) if seeded else jnp.zeros_like(sample),
            count=(
                jnp.ones((streams,), dtype=jnp.float32)
                if seeded
                else jnp.zeros((streams,), dtype=jnp.float32)
            ),
        )

    def advance(self, statistics: Statistics, value) -> Statistics:
        mean, M2 = self._open(statistics, value)
        count = statistics.count + 1
        delta = value - mean
        mean = mean + delta / _expand_for(count, value)
        return replace(
            statistics, mean=mean, M2=M2 + delta * (value - mean), count=count
        )

    def read(self, statistics: Statistics, sample):
        count = _expand_for(statistics.count, statistics.M2)
        spread = self._spread(statistics.M2, count)
        centred = sample - statistics.mean if self.config.center else sample
        return centred / jnp.sqrt(spread + self.config.eps)

    def _open(self, statistics, value):
        if self.config.cold_start == "seeded":
            return statistics.mean, statistics.M2
        first = statistics.count == 0
        return (
            jnp.where(_expand_for(first, statistics.mean), value, statistics.mean),
            jnp.where(
                _expand_for(first, statistics.M2),
                jnp.zeros_like(statistics.M2),
                statistics.M2,
            ),
        )

    def _spread(self, M2, count):
        if self.config.variance == "population":
            return M2 / count
        return jnp.where(count < 2, jnp.ones_like(M2), M2 / jnp.maximum(count - 1, 1.0))


# ----------------------------------------------------------------- the assembly
class Normalizer:
    """One estimator: accumulate, advance, read."""

    def __init__(self, config: NormalizationConfig):
        self.config = config
        self.spread = Spread(config)
        self.accumulation = (
            None
            if config.discount is None
            else Accumulation(config.discount, config.reset_on_done)
        )

    def initial(self, sample) -> Statistics:
        """Statistics shaped like the stream, one estimate per stream."""

        sample = jnp.asarray(sample, dtype=jnp.float32)
        statistics = self.spread.initial(sample)
        if self.accumulation is None:
            return statistics
        return self.accumulation.initial(statistics, sample.shape[0])

    def begin(self, sample, statistics=None, *, update=True):
        """The value an episode opens on, and the accumulation dropped with it."""

        sample = jnp.asarray(sample, dtype=jnp.float32)
        if statistics is None:
            statistics = self.initial(sample)
        if update:
            statistics = self.spread.advance(statistics, sample)
        value = self.spread.read(statistics, sample)
        if self.accumulation is not None:
            statistics = self.accumulation.dropped(statistics)
        return value, statistics

    def observe(self, statistics, sample, *, done, update=True):
        """One value, and the statistics after it."""

        sample = jnp.asarray(sample, dtype=jnp.float32)
        done = jnp.asarray(done)
        counted = sample
        if self.accumulation is not None:
            counted, statistics = self.accumulation.advance(statistics, sample, done)
        if update:
            statistics = self.spread.advance(statistics, counted)
        return self.spread.read(statistics, sample), statistics


def normalization_metrics(statistics, eps):
    """What one estimator currently believes, per stream."""

    if statistics is None:
        return NormalizationMetrics()
    axes = tuple(range(1, statistics.mean.ndim))
    spread = statistics.M2 / _expand_for(statistics.count, statistics.M2) + eps
    return NormalizationMetrics(
        mean=statistics.mean.mean(axis=axes) if axes else statistics.mean,
        std=jnp.sqrt(spread).mean(axis=axes) if axes else jnp.sqrt(spread),
    )


def make_normalizer(config: NormalizationConfig) -> Normalizer:
    """Build one estimator, refusing settings that cannot mean anything."""

    if config.reset_on_start and not config.update_during_eval:
        raise ValueError("reset_on_start=True requires update_during_eval=True")
    if config.cold_start not in COLD_STARTS:
        raise ValueError(
            f"unknown cold start {config.cold_start!r}; use {', '.join(COLD_STARTS)}"
        )
    if config.variance not in VARIANCES:
        raise ValueError(
            f"unknown variance {config.variance!r}; use {', '.join(VARIANCES)}"
        )
    return Normalizer(config)


def declared_normalizer(component, *, discount=None) -> Normalizer | None:
    """The estimator a declared component asks for, or none if it asked for none."""

    if component is None:
        return None
    return make_normalizer(
        NormalizationConfig(
            center=component.center,
            cold_start=component.cold_start,
            variance=component.variance,
            eps=component.eps,
            reset_on_start=component.reset_on_start,
            update_during_eval=component.update_during_eval,
            discount=discount,
            reset_on_done=getattr(component, "reset_on_done", True),
        )
    )


def _construct_normalizer(selection, builder, *, discount=None):
    del builder
    normalizer = declared_normalizer(selection.parameters, discount=discount)
    return None if normalizer is None else normalizer.config


NORMALIZATION_FAMILY = ComponentFamily(
    branches=NORMALIZATION_BRANCHES,
    construct=_construct_normalizer,
)
DISCOUNTED_NORMALIZATION_FAMILY = ComponentFamily(
    branches=DISCOUNTED_NORMALIZATION_BRANCHES,
    construct=_construct_normalizer,
)


def environment_owns_normalization(env) -> bool:
    """Detect the existing normalization wrappers through wrapper chains."""
    current = env
    while current is not None:
        if isinstance(
            current,
            (NormalizeObservationWrapper, NormalizeRewardWrapper),
        ):
            return True
        current = getattr(current, "_env", None)
    return False
