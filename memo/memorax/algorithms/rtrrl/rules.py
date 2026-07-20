"""Pure numerical update rules for the strict AAAI25 RTRRL path."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from memorax.online_ac.traces import TraceDirections


@struct.dataclass
class EmphasisOrAverageReward:
    """State leaves updated after one strict RTRRL transition."""

    emphasis: Any
    average_reward: Any


def _broadcast_environment(values, leaf):
    return values[(slice(None),) + (None,) * (leaf.ndim - 1)]


def _validate_environment_tree(tree, *, kind, domain, environment_count):
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        location = jax.tree_util.keystr(path)
        if leaf.ndim == 0:
            raise ValueError(
                f"{kind} leaf {location} in domain {domain} must have a "
                f"non-scalar leading environment dimension; got shape {leaf.shape}"
            )
        if leaf.shape[0] != environment_count:
            raise ValueError(
                f"{kind} leaf {location} in domain {domain} has environment "
                f"size {leaf.shape[0]}, but delta environment size "
                f"{environment_count}"
            )


def td_error(
    *,
    reward,
    value,
    next_value,
    terminated,
    gamma,
    average_reward: Any = 0.0,
):
    """Return per-environment differential TD(0) errors."""

    return (
        reward
        + gamma * next_value * (1 - terminated)
        - average_reward
        - value
    )


def _accumulated_trace(old, gradient, *, decay, emphasis, terminated):
    reset_old = (1 - _broadcast_environment(terminated, old)) * old
    return (
        decay * reset_old
        + _broadcast_environment(emphasis, gradient) * gradient
    )


def _dutch_trace(old, gradient, *, decay, learning_rate, terminated):
    reset_old = (1 - _broadcast_environment(terminated, old)) * old
    parameter_axes = tuple(range(1, old.ndim))
    contraction = jnp.sum(reset_old * gradient, axis=parameter_axes)
    correction = 1 - learning_rate * decay * contraction
    return (
        decay * reset_old
        + _broadcast_environment(correction, gradient) * gradient
    )


def _dutch_update_direction(
    trace,
    direct,
    *,
    delta,
    learning_rate,
    value_difference,
):
    delta_trace = _broadcast_environment(delta, trace) * trace
    return jnp.mean(
        delta_trace
        + learning_rate
        * _broadcast_environment(value_difference, trace)
        * (trace - delta_trace)
        + direct,
        axis=0,
    )


def update_traces(
    incoming,
    gradients,
    *,
    gamma,
    lambda_actor,
    lambda_critic,
    lambda_rnn,
    trace_mode,
    critic_learning_rate,
    emphasis,
    terminated,
    timing,
):
    """Update strict actor, critic, and recurrent eligibility traces."""

    if trace_mode not in {"accumulate", "dutch"}:
        raise ValueError(f"unknown trace mode: {trace_mode}")
    if timing not in {"incoming", "fresh"}:
        raise ValueError(f"unknown trace timing: {timing}")

    decays = {
        "actor": gamma * lambda_actor,
        "critic": gamma * lambda_critic,
        "recurrent": gamma * lambda_rnn,
    }
    carried = {}
    for domain in incoming:
        if domain not in decays:
            raise ValueError(f"unknown trace domain: {domain}")
        if domain == "recurrent" and lambda_rnn == 0:
            carried[domain] = gradients[domain]
        elif domain == "critic" and trace_mode == "dutch":
            carried[domain] = jax.tree.map(
                lambda old, gradient: _dutch_trace(
                    old,
                    gradient,
                    decay=decays[domain],
                    learning_rate=critic_learning_rate,
                    terminated=terminated,
                ),
                incoming[domain],
                gradients[domain],
            )
        else:
            carried[domain] = jax.tree.map(
                lambda old, gradient: _accumulated_trace(
                    old,
                    gradient,
                    decay=decays[domain],
                    emphasis=emphasis,
                    terminated=terminated,
                ),
                incoming[domain],
                gradients[domain],
            )

    return TraceDirections(
        carried=carried,
        update=carried if timing == "fresh" else incoming,
    )


def combine_update_directions(
    traces,
    direct_gradients,
    *,
    delta,
    recurrent_scale,
    trace_mode="accumulate",
    critic_learning_rate=None,
    critic_value_difference=None,
):
    """Combine per-environment traced and preweighted direct directions.

    ``direct_gradients`` is an API boundary: objective coefficients, including
    entropy coefficients, have already been applied. This rule routes each
    direct leaf unchanged; it never delta-weights or domain-rescales it.
    """

    if traces.keys() != direct_gradients.keys():
        raise ValueError("traced and direct gradient domains must match")
    if trace_mode not in {"accumulate", "dutch"}:
        raise ValueError(f"unknown trace mode: {trace_mode}")
    delta = jnp.asarray(delta)
    if delta.ndim != 1:
        raise ValueError(
            "delta must have one environment dimension; "
            f"got shape {delta.shape}"
        )
    environment_count = delta.shape[0]
    for domain in traces:
        if (
            jax.tree_util.tree_structure(traces[domain])
            != jax.tree_util.tree_structure(direct_gradients[domain])
        ):
            raise ValueError(
                f"traced and direct gradient trees must match in domain {domain}"
            )
        _validate_environment_tree(
            traces[domain],
            kind="trace",
            domain=domain,
            environment_count=environment_count,
        )
        _validate_environment_tree(
            direct_gradients[domain],
            kind="direct",
            domain=domain,
            environment_count=environment_count,
        )
    if trace_mode == "dutch" and (
        critic_learning_rate is None or critic_value_difference is None
    ):
        raise ValueError(
            "Dutch critic updates require learning rate and value difference"
        )

    combined = {}
    for domain in traces:
        traced_scale = recurrent_scale if domain == "recurrent" else 1.0
        if domain == "critic" and trace_mode == "dutch":
            assert critic_learning_rate is not None
            assert critic_value_difference is not None
            combined[domain] = jax.tree.map(
                lambda trace, direct: _dutch_update_direction(
                    trace,
                    direct,
                    delta=delta,
                    learning_rate=critic_learning_rate,
                    value_difference=critic_value_difference,
                ),
                traces[domain],
                direct_gradients[domain],
            )
        else:
            combined[domain] = jax.tree.map(
                lambda trace, direct: jnp.mean(
                    traced_scale
                    * _broadcast_environment(delta, trace)
                    * trace
                    + direct,
                    axis=0,
                ),
                traces[domain],
                direct_gradients[domain],
            )
    return combined


def update_emphasis_or_average_reward(
    *,
    emphasis,
    average_reward,
    delta,
    terminated,
    gamma,
    eta,
):
    """Apply the episodic-emphasis or continuing-average-reward branch."""

    if eta is None:
        return EmphasisOrAverageReward(
            emphasis=gamma * emphasis * (1 - terminated) + terminated,
            average_reward=average_reward,
        )
    return EmphasisOrAverageReward(
        emphasis=emphasis,
        average_reward=average_reward + eta * jnp.mean(delta),
    )


def update_slow_target(
    *,
    fast_parameters,
    previous_slow_parameters,
    period,
):
    """Polyak-update a slow parameter tree from post-update fast parameters."""

    if period == 1.0:
        return fast_parameters
    return jax.tree.map(
        lambda fast, slow: period * fast + (1 - period) * slow,
        fast_parameters,
        previous_slow_parameters,
    )
