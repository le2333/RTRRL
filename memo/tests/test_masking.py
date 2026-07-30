"""Zeroing an observation dimension against deleting it."""

from __future__ import annotations

from typing import cast

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array

from memorax.environments.brax import masks

KEPT = 5
WIDTH = 11


def test_a_zeroed_dimension_is_a_deleted_one_to_the_layer_that_reads_it():
    """A wide layer over a masked input equals a narrow one over the survivors."""

    mask = masks["hopper"]["P"]
    assert int(mask.sum()) == KEPT and mask.size == WIDTH
    assert not bool(mask[KEPT:].any()), "the P mask is not the prefix assumed here"

    observation = jax.random.normal(jax.random.key(0), (WIDTH,))
    wide = nn.Dense(7)
    variables = wide.init(jax.random.key(1), observation)
    kernel = variables["params"]["kernel"]

    narrow = nn.Dense(7)
    sliced = {"params": {"kernel": kernel[:KEPT], "bias": variables["params"]["bias"]}}

    theirs = cast(Array, narrow.apply(sliced, observation[:KEPT]))
    ours = cast(Array, wide.apply(variables, observation * mask))

    assert jnp.array_equal(ours, theirs), "a zero and a deletion differ at the layer"


def test_a_zeroed_dimension_never_moves_the_weights_that_read_it():
    """The masked columns take exactly zero gradient."""

    mask = masks["hopper"]["P"]
    observation = jax.random.normal(jax.random.key(0), (WIDTH,))
    layer = nn.Dense(7)
    variables = layer.init(jax.random.key(1), observation)

    def loss(params):
        return jnp.sum(cast(Array, layer.apply(params, observation * mask)) ** 2)

    gradient = jax.grad(loss)(variables)["params"]["kernel"]

    assert jnp.array_equal(gradient[KEPT:], jnp.zeros_like(gradient[KEPT:]))
    assert jnp.any(gradient[:KEPT] != 0.0), "the surviving columns learned nothing"


def test_the_two_shapes_do_not_start_from_the_same_scale():
    """Fan-in differs, so the surviving columns are drawn sqrt(11/5) apart."""

    def kernel_of(fan_in: int) -> Array:
        drawn = nn.Dense(4096).init(jax.random.key(0), jnp.zeros(fan_in))
        return cast(Array, drawn["params"]["kernel"])

    wide = kernel_of(WIDTH)
    narrow = kernel_of(KEPT)

    ratio = jnp.std(narrow) / jnp.std(wide[:KEPT])

    assert jnp.allclose(ratio, jnp.sqrt(WIDTH / KEPT), rtol=0.05), float(ratio)
