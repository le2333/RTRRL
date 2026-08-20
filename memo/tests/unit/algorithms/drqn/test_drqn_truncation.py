"""How far back a window's gradient reaches, which is what the sweep sweeps.

The truncation search only means something if t is the number of steps the
gradient actually crosses. These tests hold that: the whole window is
differentiated, nothing before it is, and the two branches differ in the window
rather than in the loss.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from memorax.algorithms.drqn import (
    ZERO_MEMORY,
    Core,
    LearnerSequence,
    QFunction,
    RecurrentInputs,
)


def core(**overrides):
    settings = {
        "q_function": QFunction(
            action_dim=2,
            observation_dim=2,
            hidden_dim=3,
            feature_dim=4,
            core_kind="lru",
        ),
        "optimizer": optax.adam(0.05),
        "gamma": 0.9,
        "target_update_period": 4,
    }
    settings.update(overrides)
    return Core(**settings)


def window(key, transitions):
    walk = jax.random.normal(key, (1, transitions + 1, 2))
    return LearnerSequence(
        inputs=RecurrentInputs(
            observation=walk[:, :-1],
            episode_start=jnp.asarray([[True] + [False] * (transitions - 1)]),
        ),
        bootstrap_inputs=RecurrentInputs(
            observation=walk[:, 1:],
            episode_start=jnp.zeros((1, transitions), dtype=jnp.bool_),
        ),
        actions=jnp.zeros((1, transitions), dtype=jnp.int32),
        rewards=jnp.ones((1, transitions)),
        dones=jnp.zeros((1, transitions), dtype=jnp.bool_),
        terminals=jnp.zeros((1, transitions), dtype=jnp.bool_),
        valid=jnp.ones((1, transitions), dtype=jnp.bool_),
        batch_valid=jnp.ones((1,), dtype=jnp.bool_),
    )


@pytest.mark.parametrize("truncation", [1, 4, 8])
def test_the_gradient_reaches_every_input_in_the_window(truncation):
    """Each step in the window moves the loss, including the earliest one.

    A gradient that stopped short would leave the first inputs unable to change
    it, and the sweep would be measuring a shorter truncation than it named.
    """

    learner = core()
    drawn = window(jax.random.key(truncation), truncation)
    state = learner.init(jax.random.key(0), drawn.inputs)

    def loss_of(observation):
        moved = drawn.replace(inputs=drawn.inputs.replace(observation=observation))
        value, _ = learner._loss(state.params, state.target_params, moved)
        return value

    sensitivity = jax.grad(loss_of)(drawn.inputs.observation)

    reached = np.abs(np.asarray(sensitivity)).sum(axis=(0, 2))
    assert reached.shape == (truncation,)
    assert np.all(reached > 0.0), reached


def test_a_longer_window_carries_credit_the_shorter_one_cannot():
    """t is not a formality: the same data scored at two t gives two gradients."""

    learner = core()
    long_window = window(jax.random.key(3), 8)
    state = learner.init(jax.random.key(1), long_window.inputs)
    short_window = long_window.replace(
        inputs=jax.tree.map(lambda value: value[:, :1], long_window.inputs),
        bootstrap_inputs=jax.tree.map(
            lambda value: value[:, :1], long_window.bootstrap_inputs
        ),
        actions=long_window.actions[:, :1],
        rewards=long_window.rewards[:, :1],
        dones=long_window.dones[:, :1],
        terminals=long_window.terminals[:, :1],
        valid=long_window.valid[:, :1],
        batch_valid=long_window.batch_valid,
    )

    def gradient(sample):
        return jax.grad(
            lambda params: learner._loss(params, state.target_params, sample)[0]
        )(state.params)

    long_gradient = jax.tree.leaves(gradient(long_window))
    short_gradient = jax.tree.leaves(gradient(short_window))

    assert any(
        not np.allclose(np.asarray(left), np.asarray(right))
        for left, right in zip(long_gradient, short_gradient)
    )


def test_both_branches_open_their_window_on_the_same_zero_state():
    """Full BPTT is the branch whose window is the episode, and nothing else.

    Neither branch carries a stored recurrence in, so the loss has no branch to
    take: whatever window it is handed, it opens on zero memory.
    """

    learner = core()
    drawn = window(jax.random.key(5), 3)
    state = learner.init(jax.random.key(2), drawn.inputs)

    scored, _ = learner._loss(state.params, state.target_params, drawn)

    start = learner.q_function.reset(ZERO_MEMORY, 1)
    _, online_q, _ = learner.q_function.unroll(state.params, drawn.inputs, start)
    _, successor_q, _ = learner.q_function.unroll(
        state.target_params, drawn.bootstrap_inputs, start
    )
    q_value = jnp.take_along_axis(online_q, drawn.actions[..., None], axis=-1).squeeze(
        axis=-1
    )
    target = drawn.rewards + 0.9 * jnp.max(successor_q, axis=-1)
    expected = 0.5 * jnp.mean(jnp.square(target - q_value))

    np.testing.assert_allclose(float(scored), float(expected), rtol=1e-5)
