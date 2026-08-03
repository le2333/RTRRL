"""Which policy head, as a structure with a branch per parameterisation.

Three exist and they differ in where the scale comes from:

- ``global_std`` is memorax's: one ``Dense`` for the mean and a learnable
  ``log_std`` that no observation reaches.
- ``state_std`` is streaming-drl's: two ``Linear``s, the second one's output
  through ``softplus``.
- ``bounded`` is the one this repository added: loc and log-scale both squashed
  into an interval before ``softplus``.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest
from training_sdk.contract import StructureSpec

from entries import stream_ac
from memorax.networks import heads
from memorax.networks.policy import ACTOR_HEAD_BRANCHES, actor_head

ACTIONS = 3
WIDTH = 6
BRANCHES = tuple(ACTOR_HEAD_BRANCHES)


def scales(name: str) -> tuple[jax.Array, jax.Array]:
    """The scale each of two different observations is given."""

    head = actor_head(name, action_dim=ACTIONS)
    left = jnp.linspace(-1.0, 1.0, WIDTH, dtype=jnp.float32)[None]
    right = jnp.linspace(2.0, 9.0, WIDTH, dtype=jnp.float32)[None]
    params = head.init(jax.random.key(0), left)
    return tuple(head.apply(params, x)[0].scale_diag for x in (left, right))


@pytest.mark.parametrize("name", BRANCHES)
def test_every_branch_builds_and_answers_with_a_distribution(name):
    head = actor_head(name, action_dim=ACTIONS)
    x = jnp.zeros((1, WIDTH), dtype=jnp.float32)
    params = head.init(jax.random.key(0), x)
    distribution, _ = head.apply(params, x)

    assert distribution.loc.shape == (1, ACTIONS)
    assert jnp.all(distribution.scale_diag > 0.0)


def test_the_global_scale_is_the_same_whatever_was_observed():
    left, right = scales("global_std")

    assert jnp.allclose(left, right)


@pytest.mark.parametrize("name", ("state_std", "bounded"))
def test_the_other_two_read_the_scale_off_the_observation(name):
    left, right = scales(name)

    assert not jnp.allclose(left, right)


def test_the_bounded_one_keeps_both_inside_its_interval():
    head = actor_head("bounded", action_dim=ACTIONS)
    x = jnp.full((1, WIDTH), 50.0, dtype=jnp.float32)
    params = head.init(jax.random.key(0), x)
    distribution, _ = head.apply(params, x)

    low, high = heads.BoundedGaussian.loc_bounds
    assert jnp.all(distribution.loc >= low) and jnp.all(distribution.loc <= high)
    assert jnp.all(distribution.scale_diag <= jax.nn.softplus(1.0) * 3)


def test_the_head_a_parameterisation_is_its_own_component():
    """``Gaussian`` carried both by flag, which made it two components in one."""

    named = [field.name for field in dataclasses.fields(heads.Gaussian)]

    assert "bound" not in named
    assert "loc_bounds" not in named


def test_the_actor_head_is_declared_as_a_structure():
    node = stream_ac.PARAMETERS["actor_head"]

    assert isinstance(node, StructureSpec)
    assert set(node.branches) == set(ACTOR_HEAD_BRANCHES)
