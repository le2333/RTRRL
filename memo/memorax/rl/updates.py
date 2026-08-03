"""Interchangeable rules that turn ascent directions into parameter updates.

A rule owns one parameter group. Which parameters form a group, and what
scaling an algorithm applies to a trace before handing it over, stays with the
algorithm; a rule only decides how far to step along the direction it is given.

Both rules receive the eligibility trace and the TD error separately rather
than a finished gradient. Adam does not need the split, but OBGD does: its
step-size bound reads |delta| and the trace norm together, so delta cannot be
folded in beforehand.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from training_sdk.parameters import param


@struct.dataclass
class ObjectiveDirections:
    """What an objective asks for, split by how the ascent reaches a parameter.

    Traced directions are accumulated into an eligibility trace and later
    weighted by the TD error; direct ones apply on the step they arise. Which
    domains exist, and which parameter group each maps onto, is the
    algorithm's own routing decision.
    """

    traced_by_domain: Any
    direct_by_domain: Any
    metrics: Any


@struct.dataclass
class RuleOutput:
    """Parameter updates, the rule's carried state, and its diagnostics."""

    updates: Any
    state: Any
    metrics: Any


@dataclass(frozen=True)
class UpdateRule:
    """Build-time pairing of a rule's state constructor and its step.

    ``init`` takes both the parameter tree and the trace tree because the two
    rules carry different shapes: Adam's moments follow the parameters, which
    the env axis has already been averaged out of, while OBGD's second moment
    follows the trace and stays per-env, since its bound is measured per-env.
    """

    init: Callable[..., Any]
    apply: Callable[..., RuleOutput]


def _broadcast_env(values, leaf):
    return values[(slice(None),) + (None,) * (leaf.ndim - 1)]


def _combine(traced, direct, delta):
    """Weight the trace by the TD error and add the untraced directions.

    ``direct`` may be None when an objective routes everything through the
    trace, which saves carrying a tree of zeros through every step.
    """

    if direct is None:
        return jax.tree.map(lambda trace: _broadcast_env(delta, trace) * trace, traced)
    return jax.tree.map(
        lambda trace, immediate: _broadcast_env(delta, trace) * trace + immediate,
        traced,
        direct,
    )


def make_optax_rule(transform, *, rate=None) -> UpdateRule:
    """Step along delta * trace through any optax transformation.

    The transformation sees one finished ascent direction per parameter, so
    whatever an algorithm expresses through optax -- Adam, clipping, freezing
    a subtree -- it expresses by composing the transformation it passes in.

    ``rate`` is the scalar the transformation multiplies the ascent by, when the
    caller knows it. Given, the rule reports it as the step size it took, which
    is the same reading a bound reports and lets a caller read one name whether
    or not a bound is in the way.
    """

    def init(*, params, traces):
        del traces
        return transform.init(params)

    def apply(traced, direct, state, *, delta, step, params):
        del step
        ascent = jax.tree.map(
            lambda leaf: jnp.mean(leaf, axis=0),
            _combine(traced, direct, delta),
        )
        updates, state = transform.update(ascent, state, params)
        metrics = {} if rate is None else {"step_size": jnp.full_like(delta, rate)}
        return RuleOutput(updates=updates, state=state, metrics=metrics)

    return UpdateRule(init=init, apply=apply)


@dataclass(frozen=True)
class ObBound:
    kappa: float = param(valid=(0.0, 100.0), search=(0.5, 10.0), placeholder=2.0)


@dataclass(frozen=True)
class AdaptiveObBound:
    kappa: float = param(valid=(0.0, 100.0), search=(0.5, 10.0), placeholder=2.0)
    beta2: float = param(valid=(0.0, 1.0), search=(0.9, 0.9999), placeholder=0.999)
    eps: float = param(valid=(1e-12, 1e-2), search=[1e-8], placeholder=1e-8, log=True)


@dataclass(frozen=True)
class AdaptiveObBoundFixed(AdaptiveObBound):
    """The published placement of eps: inside the root rather than beside it.

    Identical to declare and identical to read, and a different component all
    the same, because it is a different denominator. A branch that shared a
    class with another could not be told apart by what it hands back.
    """


@dataclass(frozen=True)
class Sgd:
    lr: float = param(valid=(1e-9, 10.0), search=(1e-5, 1.0), placeholder=0.1, log=True)


@dataclass(frozen=True)
class Adam:
    lr: float = param(
        valid=(1e-9, 10.0), search=(1e-5, 1e-2), placeholder=0.001, log=True
    )
    b1: float = param(valid=(0.0, 1.0), search=[0.9], placeholder=0.9)
    b2: float = param(valid=(0.0, 1.0), search=[0.999], placeholder=0.999)
    eps: float = param(valid=(1e-12, 1e-2), search=[1e-8], placeholder=1e-8, log=True)


def base_transform(base):
    """The optax transform a base names, for the paths that have no bound."""

    import optax

    if isinstance(base, Adam):
        return optax.adam(base.lr, b1=base.b1, b2=base.b2, eps=base.eps)
    return optax.sgd(base.lr)


BOUND_BRANCHES = {
    "none": (),
    "ob": ObBound,
    "adaptive_ob": AdaptiveObBound,
    "adaptive_ob_fixed": AdaptiveObBoundFixed,
}

BASE_BRANCHES = {"sgd": Sgd, "adam": Adam}


def make_bounded_rule(*, bound, base) -> UpdateRule:
    """Step along delta * trace under an overshooting bound on the step size.

    Two axes rather than one name. The bound shrinks the step whenever
    ``|delta| * ||z||_1 * lr * kappa`` exceeds one, which keeps a single update
    from crossing the TD target; the base is what it is written over. One name
    for the pair meant nothing could ask for a different bound without also
    changing the base, or the other way round.

    ``bound=None``
        No bound: the base alone, which is what an experiment asks for by
        naming ``optimizer_bound: none``.
    ``ObBound``
        No second moment and no eps at all, as published.
    ``AdaptiveObBound``
        Divides by ``sqrt(v_hat) + eps``, which is what memorax and everything
        forked from it compute. Kept exactly as it is because the recorded runs
        and the golden snapshot answer to it.
    ``AdaptiveObBoundFixed``
        Divides by ``sqrt(v_hat + eps)``, which is what AdaptiveObGD as
        published computes. The difference is not a rounding. While the second
        moment is near eps the two denominators stand a factor apart, and the
        step with it: measured against the published optimiser, this bound and
        the one above are about a factor of two apart there. They agree again
        once the second moment clears eps, where both denominators are
        ``sqrt(v_hat)``.
    """

    if bound is None:
        return make_optax_rule(base_transform(base), rate=base.lr)
    if not isinstance(base, Sgd):
        raise ValueError(
            "the overshooting bound is written over a plain rate; putting it "
            f"over {type(base).__name__.lower()} is a different rule and none "
            "is published, so it is refused rather than guessed at"
        )
    learning_rate = base.lr
    kappa = bound.kappa
    adaptive = isinstance(bound, AdaptiveObBound)
    beta2 = getattr(bound, "beta2", 0.0)
    eps = getattr(bound, "eps", 1e-8)
    eps_inside_root = isinstance(bound, AdaptiveObBoundFixed)

    def denominator(second):
        if eps_inside_root:
            return jnp.sqrt(second + eps)
        return jnp.sqrt(second) + eps

    def init(*, params, traces):
        del params
        return jax.tree.map(jnp.zeros_like, traces)

    def apply(traced, direct, state, *, delta, step, params):
        del params
        # An unbounded second moment is only carried where something reads it.
        # Under the plain bound the step never does, and accumulating one there
        # meant the rule needed a decay rate that changed nothing.
        moment = (
            jax.tree.map(
                lambda old, trace: (
                    beta2 * old
                    + (1 - beta2) * jnp.square(_broadcast_env(delta, trace) * trace)
                ),
                state,
                traced,
            )
            if adaptive
            else state
        )

        if adaptive:
            corrected = jax.tree.map(lambda leaf: leaf / (1.0 - beta2**step), moment)
            normalized = jax.tree.map(
                lambda trace, second: jnp.abs(trace) / denominator(second),
                traced,
                corrected,
            )
        else:
            corrected = None
            normalized = jax.tree.map(jnp.abs, traced)

        trace_sum = sum(
            jnp.sum(leaf, axis=tuple(range(1, leaf.ndim)))
            for leaf in jax.tree.leaves(normalized)
        )
        delta_bar = jnp.maximum(jnp.abs(delta), 1.0)
        step_size = learning_rate / jnp.maximum(
            1.0,
            delta_bar * trace_sum * learning_rate * kappa,
        )

        # The bounded step size reaches the TD error first, and their product
        # multiplies the trace. That is the order StreamAC multiplies in, and
        # multiplying in any other order is the same number differently
        # rounded, which would leave this rule permanently a last bit away from
        # the implementation it has to answer to. So the combined direction
        # ``_combine`` builds for optax is not reused here.
        ascent = jax.tree.map(
            lambda trace: (
                (_broadcast_env(step_size, trace) * _broadcast_env(delta, trace))
                * trace
            ),
            traced,
        )
        if direct is not None:
            ascent = jax.tree.map(
                lambda leaf, immediate: (
                    leaf + _broadcast_env(step_size, immediate) * immediate
                ),
                ascent,
                direct,
            )
        if adaptive:
            ascent = jax.tree.map(
                lambda leaf, second: leaf / denominator(second),
                ascent,
                corrected,
            )
        updates = jax.tree.map(lambda leaf: jnp.mean(leaf, axis=0), ascent)
        return RuleOutput(
            updates=updates,
            state=moment,
            metrics={"step_size": step_size},
        )

    return UpdateRule(init=init, apply=apply)
