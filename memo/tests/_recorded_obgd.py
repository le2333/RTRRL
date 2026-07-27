"""The bounded step exactly as the code that recorded the snapshot wrote it.

Lifted from ``memo/memorax/online_ac/updates.py`` at commit 5f7ff4e, the commit
that added ``tests/golden/stream_ac_rtu``. That implementation was deleted, and
the snapshot is all that is left of what it computed -- except that its
arithmetic is still in the history, so it can be asked directly instead of
inferred from four bias vectors.

It is here to answer one question: whether the last last-bit between our rule
and the recording is inside the bounded step or upstream of it. Delete this file
once the answer is written down somewhere better.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _broadcast_env(values, leaf):
    return values[(slice(None),) + (None,) * (leaf.ndim - 1)]


def make_whole_tree_obgd(config):
    """Build StreamAC OBGD over one complete actor or critic parameter tree."""

    beta2 = config.beta2
    eps = config.eps
    adaptive = config.adaptive

    def obgd(traces, v, *, delta, learning_rate, kappa, step):
        new_v = jax.tree.map(
            lambda old_v, trace: (
                beta2 * old_v
                + (1 - beta2) * jnp.square(_broadcast_env(delta, trace) * trace)
            ),
            v,
            traces,
        )

        if adaptive:
            v_hat = jax.tree.map(
                lambda moment: moment / (1.0 - beta2**step),
                new_v,
            )
            norm_leaves = jax.tree.leaves(
                jax.tree.map(
                    lambda trace, moment: (jnp.abs(trace) / (jnp.sqrt(moment) + eps)),
                    traces,
                    v_hat,
                )
            )
            trace_sum = sum(
                jnp.sum(leaf, axis=tuple(range(1, leaf.ndim))) for leaf in norm_leaves
            )
        else:
            v_hat = None
            trace_sum = sum(
                jnp.sum(jnp.abs(leaf), axis=tuple(range(1, leaf.ndim)))
                for leaf in jax.tree.leaves(traces)
            )

        delta_bar = jnp.maximum(jnp.abs(delta), 1.0)
        step_size = learning_rate / jnp.maximum(
            1.0,
            delta_bar * trace_sum * learning_rate * kappa,
        )

        if adaptive:
            updates = jax.tree.map(
                lambda trace, moment: jnp.mean(
                    _broadcast_env(step_size, trace)
                    * _broadcast_env(delta, trace)
                    * trace
                    / (jnp.sqrt(moment) + eps),
                    axis=0,
                ),
                traces,
                v_hat,
            )
        else:
            updates = jax.tree.map(
                lambda trace: jnp.mean(
                    _broadcast_env(step_size, trace)
                    * _broadcast_env(delta, trace)
                    * trace,
                    axis=0,
                ),
                traces,
            )

        return updates, new_v

    return obgd
