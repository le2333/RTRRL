"""Pure optimizer builders for RTRRL and StreamAC."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

_RECURRENT_DOMAINS = frozenset({"feature_extractor", "torso"})


def make_grouped_adam(config, abstract_params):
    """Build legacy-compatible grouped Adam with positive ascent scales."""

    td_transform = optax.chain(
        optax.scale_by_adam(b1=config.b1, b2=config.b2, eps=config.eps),
        optax.scale(config.td_lr),
    )
    recurrent_transforms = []
    if config.rnn_grad_clip:
        recurrent_transforms.append(optax.clip_by_global_norm(config.rnn_grad_clip))
    recurrent_transforms.extend(
        (
            optax.scale_by_adam(
                b1=config.b1,
                b2=config.b2,
                eps=config.eps,
            ),
            optax.scale(config.rnn_lr),
        )
    )

    def label(path, _leaf):
        top = getattr(path[0], "key", None)
        if top not in _RECURRENT_DOMAINS:
            return "td"
        if config.freeze_gamma and any(
            getattr(part, "key", None) == "gamma_log" for part in path
        ):
            return "frozen"
        return "rnn"

    labels = jax.tree_util.tree_map_with_path(label, abstract_params)
    return optax.multi_transform(
        {
            "td": td_transform,
            "rnn": optax.chain(*recurrent_transforms),
            "frozen": optax.set_to_zero(),
        },
        labels,
    )


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
