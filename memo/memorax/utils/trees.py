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


def tree_cosine(left, right):
    """The cosine of the angle between two trees, or NaN if either is zero."""

    pairs = list(zip(jax.tree.leaves(left), jax.tree.leaves(right)))
    if not pairs:
        return jnp.asarray(jnp.nan)
    dot = sum(jnp.sum(one * other) for one, other in pairs)
    scale = tree_norm(left) * tree_norm(right)
    return jnp.where(scale > 0, dot / jnp.where(scale > 0, scale, 1.0), jnp.nan)


def stream_norm(tree):
    """One Euclidean norm per stream over every leaf of one unit.

    The env axis is axis 0 of every streamed leaf, so the sum runs over
    everything but it, and a unit is a whole subtree rather than a leaf:
    measuring leaf by leaf would keep only each leaf's own length and throw
    away the direction the leaves hold between them, which is a different
    quantity and, where something is normalized by it, a different algorithm.
    """

    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.asarray(0.0)
    return jnp.sqrt(
        sum(
            jnp.sum(jnp.square(leaf.reshape(leaf.shape[0], -1)), axis=1)
            for leaf in leaves
        )
    )


def subtree_norms(tree, *, streams: bool = False) -> dict:
    """One L2 norm per top-level subtree, keyed by its name.

    Split rather than summed because a network's parts are not credited alike:
    with exact recurrent credit the cell's own parameters see the whole stream
    while everything before it sees one step, and a single norm over the tree
    averages that distinction away.

    With ``streams``, each leaf's leading axis is a parallel environment and is
    kept, so the reading is per stream rather than over all of them at once.
    """

    parameter_tree = tree.get("params", tree) if hasattr(tree, "get") else tree

    def norm(subtree):
        if not jax.tree.leaves(subtree):
            return jnp.asarray(0.0)
        return stream_norm(subtree) if streams else tree_norm(subtree)

    return {name: norm(subtree) for name, subtree in parameter_tree.items()}


def find_leaf(tree, name):
    """The first leaf whose path passes through ``name``, or None."""

    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        if any(getattr(key, "key", None) == name for key in path):
            return leaf
    return None
