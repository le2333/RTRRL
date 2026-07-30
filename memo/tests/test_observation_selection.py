"""Selecting observation dimensions by index."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from memorax.environments import make

HOPPER_WIDTH = 11
KEPT = (0, 1, 2, 3, 4)


def test_the_environment_reports_only_the_selected_dimensions():
    env, params = make("brax::hopper", observed=KEPT, backend="spring")

    observation, _ = env.reset(jax.random.key(0), params)

    assert observation.shape == (len(KEPT),)
    assert env.observation_space(params).shape == (len(KEPT),)


def test_a_full_observation_is_what_the_task_reports():
    env, params = make("brax::hopper", backend="spring")

    observation, _ = env.reset(jax.random.key(0), params)

    assert observation.shape == (HOPPER_WIDTH,)


def test_the_selected_dimensions_are_the_ones_asked_for():
    wide, params = make("brax::hopper", backend="spring")
    narrow, _ = make("brax::hopper", observed=KEPT, backend="spring")
    key = jax.random.key(0)

    whole, _ = wide.reset(key, params)
    part, _ = narrow.reset(key, params)

    assert jnp.array_equal(part, whole[jnp.asarray(KEPT)])
