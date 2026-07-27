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


def make_optax_rule(transform) -> UpdateRule:
    """Step along delta * trace through any optax transformation.

    The transformation sees one finished ascent direction per parameter, so
    whatever an algorithm expresses through optax -- Adam, clipping, freezing
    a subtree -- it expresses by composing the transformation it passes in.
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
        return RuleOutput(updates=updates, state=state, metrics={})

    return UpdateRule(init=init, apply=apply)


def make_obgd_rule(
    *,
    learning_rate,
    kappa,
    beta2=0.0,
    eps=1e-8,
    adaptive=False,
) -> UpdateRule:
    """Step along delta * trace under an overshooting bound on the step size.

    The bound shrinks the step whenever ``|delta| * ||z||_1 * lr * kappa``
    exceeds one, which keeps a single update from crossing the TD target.
    ``adaptive`` normalises the trace by a second moment before measuring its
    norm, giving the AdaOBGD variant.
    """

    def init(*, params, traces):
        del params
        return jax.tree.map(jnp.zeros_like, traces)

    def apply(traced, direct, state, *, delta, step, params):
        del params
        moment = jax.tree.map(
            lambda old, trace: (
                beta2 * old
                + (1 - beta2) * jnp.square(_broadcast_env(delta, trace) * trace)
            ),
            state,
            traced,
        )

        if adaptive:
            corrected = jax.tree.map(lambda leaf: leaf / (1.0 - beta2**step), moment)
            normalized = jax.tree.map(
                lambda trace, second: jnp.abs(trace) / (jnp.sqrt(second) + eps),
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
                lambda leaf, second: leaf / (jnp.sqrt(second) + eps),
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
