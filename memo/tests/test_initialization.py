"""How weights are drawn is a structure, so both ways are selectable.

``sparse`` is streaming-drl's: ``initialize_weights`` puts
``sparse_init(sparsity=0.9)`` on every ``nn.Linear`` including the heads and
zeroes every bias.

``lecun`` is the framework default rather than anyone's choice -- flax's
``nn.Dense`` initialises its kernel with ``lecun_normal()`` unless told
otherwise, and memorax writes that same default down in its blocks and heads
while its own StreamAC example passes a bare ``nn.Dense``.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest
from training_sdk.contract import StructureSpec

from entries import stream_ac
from memorax.networks.components import FFN
from memorax.networks.initialization import (
    INITIALIZATION_BRANCHES,
    Sparse,
    declared_initializer,
)

SHAPE = (16, 8)


def drawn(component) -> jax.Array:
    return declared_initializer(component)(jax.random.key(0), SHAPE, jnp.float32)


def test_initialization_is_declared_as_a_structure():
    node = stream_ac.PARAMETERS["initialization"]

    assert isinstance(node, StructureSpec)
    assert set(node.branches) == set(INITIALIZATION_BRANCHES)


def test_sparse_zeroes_the_declared_share_of_every_output_unit():
    """Their loop zeroes ``ceil(sparsity * fan_in)`` inputs per output unit."""

    fan_in, fan_out = SHAPE
    weights = drawn(Sparse(sparsity=0.9))
    zeros = (weights == 0.0).sum(axis=0)

    assert zeros.shape == (fan_out,)
    assert jnp.all(zeros == math.ceil(0.9 * fan_in))


def test_the_other_branch_leaves_every_weight_in_place():
    """``lecun`` carries no component, so it reads back as ``None``."""

    assert INITIALIZATION_BRANCHES["lecun"] == ()

    weights = drawn(None)

    assert jnp.count_nonzero(weights) == weights.size


@pytest.mark.parametrize("sparsity", (0.5, 0.9))
def test_the_share_is_the_one_that_was_declared(sparsity):
    fan_in, _ = SHAPE
    weights = drawn(Sparse(sparsity=sparsity))

    assert jnp.all((weights == 0.0).sum(axis=0) == math.ceil(sparsity * fan_in))


def test_a_component_takes_the_initialiser_it_was_given():
    layer = FFN(features=4, kernel_init=declared_initializer(Sparse(sparsity=0.9)))
    x = jnp.ones((2, 1, 16), dtype=jnp.float32)
    params = layer.init(jax.random.key(0), x)
    kernel = params["params"]["Dense_0"]["kernel"]

    assert jnp.all((kernel == 0.0).sum(axis=0) == math.ceil(0.9 * 16))
