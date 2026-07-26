"""Pytree readings used by diagnostics."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def tree_norm(tree):
    """The L2 norm over every leaf of a tree, zero when it has none."""

    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.asarray(0.0)
    return jnp.sqrt(sum(jnp.sum(jnp.abs(leaf) ** 2) for leaf in leaves))


def find_leaf(tree, name):
    """The first leaf whose path passes through ``name``, or None."""

    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        if any(getattr(key, "key", None) == name for key in path):
            return leaf
    return None
