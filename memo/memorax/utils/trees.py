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
    """The cosine of the angle between two trees, or NaN if either is zero.

    Both trees are flattened into one vector, so the reading is about the whole
    tree's direction and not any leaf's, and every axis is flattened with the
    rest: a leading axis of parallel environments is folded in rather than
    reported per stream, which is what ``tree_norm`` beside this does too.

    NaN and not zero when a side vanishes, because zero has no direction and
    reporting a right angle would invent one. Epoch averages are ``nanmean``, so
    an undefined step drops out instead of pulling the average sideways.
    """

    pairs = list(zip(jax.tree.leaves(left), jax.tree.leaves(right)))
    if not pairs:
        return jnp.asarray(jnp.nan)
    dot = sum(jnp.sum(one * other) for one, other in pairs)
    scale = tree_norm(left) * tree_norm(right)
    return jnp.where(scale > 0, dot / jnp.where(scale > 0, scale, 1.0), jnp.nan)


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
        leaves = jax.tree.leaves(subtree)
        if not leaves:
            return jnp.asarray(0.0)
        if not streams:
            return tree_norm(subtree)
        squares = [
            jnp.sum(jnp.square(leaf.reshape(leaf.shape[0], -1)), axis=1)
            for leaf in leaves
        ]
        return jnp.sqrt(sum(squares))

    return {name: norm(subtree) for name, subtree in parameter_tree.items()}


def find_leaf(tree, name):
    """The first leaf whose path passes through ``name``, or None."""

    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        if any(getattr(key, "key", None) == name for key in path):
            return leaf
    return None
