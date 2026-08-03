"""How weights are drawn belongs to the layers that have weights to draw.

Declared under the backbone branch that has any, so a branch with none does not
offer the choice and the value it would have needed is not in the manifest.

``sparse`` is chosen on both sides at 0.9: streaming-drl's
``initialize_weights`` puts ``sparse_init(sparsity=0.9)`` on every ``nn.Linear``
including the heads and zeroes every bias, and memorax's MinAtar example passes
``sparse(sparsity=0.9)`` to its convolution, its dense layer and both heads.

``lecun`` is the framework default rather than anyone's choice -- flax's
``nn.Dense`` initialises its kernel with ``lecun_normal()`` unless told
otherwise, and memorax's other StreamAC example passes a bare ``nn.Dense``.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest
from training_sdk.contract import StructureSpec
from training_sdk.parameters import expand, flatten

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


def test_the_branch_with_layers_declares_it_and_the_branch_without_does_not():
    backbone = stream_ac.PARAMETERS["backbone"]
    mlp = backbone.branches["mlp"]["initialization"]

    assert isinstance(mlp, StructureSpec)
    assert set(mlp.branches) == set(INITIALIZATION_BRANCHES)
    # ``rtu`` is the cell and a head; the cell draws its own and memorax fixes
    # how, so there is nothing here for a branch to choose between.
    assert "initialization" not in backbone.branches["rtu"]
    assert "initialization" not in stream_ac.PARAMETERS


def test_the_key_only_exists_under_the_branch_that_has_it():
    flat = flatten(expand(stream_ac.PARAMETERS))

    assert "backbone.mlp.initialization" in flat
    assert "backbone.mlp.initialization.sparse.sparsity" in flat
    assert not [key for key in flat if key.startswith("backbone.rtu.initialization")]


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
