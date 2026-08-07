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


class Normalizer:
    def __init__(self, config: NormalizationConfig):
        self.config = config

    def initial(self, sample) -> Statistics:
        """Statistics shaped like the stream, one estimate per stream."""

        sample = jnp.asarray(sample, dtype=jnp.float32)
        streams = sample.shape[0]
        seeded = self.config.cold_start == "seeded"
        counted = jnp.ones((streams,), dtype=jnp.float32)
        return Statistics(
            mean=jnp.zeros_like(sample),
            M2=jnp.ones_like(sample) if seeded else jnp.zeros_like(sample),
            count=counted if seeded else jnp.zeros((streams,), dtype=jnp.float32),
            trace=(
                jnp.zeros((streams,), dtype=jnp.float32)
                if self.config.discount is not None
                else None
            ),
        )

    def begin(self, sample, statistics=None, *, update=True):
        """The value an episode opens on, and the accumulation dropped with it."""

        sample = jnp.asarray(sample, dtype=jnp.float32)
        if statistics is None:
            statistics = self.initial(sample)
        if update:
            statistics = self._accumulate(statistics, sample)
        return self._scaled(statistics, sample), replace(
            statistics,
            trace=(
                None if statistics.trace is None else jnp.zeros_like(statistics.trace)
            ),
        )

    def observe(self, statistics, sample, *, done, update=True):
        """One value, and the statistics after it."""

        sample = jnp.asarray(sample, dtype=jnp.float32)
        done = jnp.asarray(done)
        counted = sample
        if statistics.trace is not None:
            counted = sample + self.config.discount * statistics.trace * (1 - done)
            statistics = replace(
                statistics,
                trace=counted * (1 - done) if self.config.reset_on_done else counted,
            )
        if update:
            statistics = self._accumulate(statistics, counted)
        return self._scaled(statistics, sample), statistics

    def _accumulate(self, statistics, sample) -> Statistics:
        mean, M2 = self._open(statistics, sample)
        count = statistics.count + 1
        delta = sample - mean
        mean = mean + delta / _expand_for(count, sample)
        return replace(
            statistics, mean=mean, M2=M2 + delta * (sample - mean), count=count
        )

    def _scaled(self, statistics, sample):
        count = _expand_for(statistics.count, statistics.M2)
        spread = self._spread(statistics.M2, count)
        centred = sample - statistics.mean if self.config.center else sample
        return centred / jnp.sqrt(spread + self.config.eps)

    def _open(self, statistics, sample):
        if self.config.cold_start == "seeded":
            return statistics.mean, statistics.M2
        first = statistics.count == 0
        return (
            jnp.where(_expand_for(first, statistics.mean), sample, statistics.mean),
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
