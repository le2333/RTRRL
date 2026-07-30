"""What a zeroed observation dimension is, to the network that reads it.

We mask a partially observed task by multiplying the observation by zero. The
reference implementation deletes the dimension instead, so its network reads a
five-vector where ours reads an eleven-vector with six zeros in it, and whether
those are the same task is a precondition for comparing the two at all: if the
agent can still see the velocities through a zero, our Hopper is not partially
observed and no curve drawn against theirs means anything.

These say it is the same task and name the one place the two do part company,
which is the starting point rather than the task.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from memorax.environments.brax import masks

KEPT = 5
WIDTH = 11


def test_a_zeroed_dimension_is_a_deleted_one_to_the_layer_that_reads_it():
    """The masked columns multiply zero, so they contribute exactly nothing.

    Not approximately nothing: a dense layer is a sum of columns weighted by
    the input, and a weight times zero is zero before any rounding. So the wide
    layer over a masked observation computes the narrow layer's function of the
    surviving dimensions, and the network is as blind either way.
    """

    mask = masks["hopper"]["P"]
    assert int(mask.sum()) == KEPT and mask.size == WIDTH
    assert not bool(mask[KEPT:].any()), "the P mask is not the prefix assumed here"

    observation = jax.random.normal(jax.random.key(0), (WIDTH,))
    wide = nn.Dense(7)
    variables = wide.init(jax.random.key(1), observation)
    kernel = variables["params"]["kernel"]

    narrow = nn.Dense(7)
    sliced = {"params": {"kernel": kernel[:KEPT], "bias": variables["params"]["bias"]}}

    theirs = narrow.apply(sliced, observation[:KEPT])
    ours = wide.apply(variables, observation * mask)

    assert jnp.array_equal(ours, theirs), "a zero and a deletion differ at the layer"


def test_a_zeroed_dimension_never_moves_the_weights_that_read_it():
    """And it stays deleted, rather than being learned around later.

    The gradient of a weight on a zeroed input is that input times the incoming
    error, so it is exactly zero at every step. The columns behind the mask are
    dead for the whole run: they cost memory and initialisation draws and they
    never enter the function.
    """

    mask = masks["hopper"]["P"]
    observation = jax.random.normal(jax.random.key(0), (WIDTH,))
    layer = nn.Dense(7)
    variables = layer.init(jax.random.key(1), observation)

    def loss(params):
        return jnp.sum(layer.apply(params, observation * mask) ** 2)

    gradient = jax.grad(loss)(variables)["params"]["kernel"]

    assert jnp.array_equal(gradient[KEPT:], jnp.zeros_like(gradient[KEPT:]))
    assert jnp.any(gradient[:KEPT] != 0.0), "the surviving columns learned nothing"


def test_the_two_shapes_do_not_start_from_the_same_scale():
    """The one real difference, and it is an initialisation and not a task.

    Flax draws a dense kernel with a standard deviation of one over the root of
    its fan-in, and the fan-in is eleven for us and five for them. So the
    surviving columns start about ``sqrt(11/5)`` times smaller here even though
    they compute the same function of the same inputs.

    This is worth stating rather than fixing: it is the same kind of difference
    as every other initialisation gap between the two codebases, and closing it
    alone would not make the two comparable in any sense the rest are not.
    """

    observation = jnp.zeros(WIDTH)
    wide = nn.Dense(4096).init(jax.random.key(0), observation)["params"]["kernel"]
    narrow = nn.Dense(4096).init(jax.random.key(0), jnp.zeros(KEPT))["params"]["kernel"]

    ratio = jnp.std(narrow) / jnp.std(wide[:KEPT])

    assert jnp.allclose(ratio, jnp.sqrt(WIDTH / KEPT), rtol=0.05), float(ratio)
