"""Readings taken off a parameter tree, which a diagnostic is only as good as."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from memorax.utils.trees import subtree_norms, tree_cosine, tree_norm


def test_a_norm_is_taken_per_part_and_per_stream():
    """Two things a single norm would blur: which part, and which stream.

    Parts, because with exact recurrent credit the cell sees the whole stream
    and what feeds it sees one step, and a norm over the whole network averages
    that away. Streams, because every leaf carries a leading axis of parallel
    environments and one of them diverging is exactly what a norm is watched
    for.
    """

    grads = {
        "params": {
            "torso": {"kernel": jnp.asarray([[3.0, 4.0], [0.0, 0.0]])},
            "head": {"bias": jnp.asarray([[0.0], [5.0]])},
        }
    }

    assert subtree_norms(grads, streams=True).keys() == {"torso", "head"}
    per_stream = subtree_norms(grads, streams=True)
    assert jnp.allclose(per_stream["torso"], jnp.asarray([5.0, 0.0]))
    assert jnp.allclose(per_stream["head"], jnp.asarray([0.0, 5.0]))

    whole = subtree_norms(grads)
    assert jnp.allclose(whole["torso"], tree_norm(grads["params"]["torso"]))


def test_a_part_with_no_parameters_reads_zero_rather_than_failing():
    """A memoryless torso has parameters; a bare one would have none."""

    assert subtree_norms({"params": {"torso": {}}})["torso"] == 0.0


def test_a_cosine_says_which_way_two_trees_point():
    """Parallel reads 1, opposed -1, orthogonal 0, and scale does not move it."""

    left = {"torso": {"kernel": jnp.asarray([[3.0, 4.0]])}, "bias": jnp.asarray([0.0])}
    longer = jax.tree.map(lambda leaf: leaf * 2.0, left)
    against = jax.tree.map(lambda leaf: -leaf, left)
    across = {
        "torso": {"kernel": jnp.asarray([[4.0, -3.0]])},
        "bias": jnp.asarray([0.0]),
    }

    assert jnp.allclose(tree_cosine(left, longer), 1.0)
    assert jnp.allclose(tree_cosine(left, against), -1.0)
    assert jnp.allclose(tree_cosine(left, across), 0.0)


def test_a_cosine_against_nothing_is_undefined_rather_than_orthogonal():
    """A zero side gives NaN, not 0.0."""

    left = {"kernel": jnp.asarray([3.0, 4.0])}

    assert jnp.isnan(tree_cosine(left, jax.tree.map(jnp.zeros_like, left)))
