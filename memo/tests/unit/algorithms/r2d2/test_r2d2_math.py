import jax
import jax.numpy as jnp
import numpy as np
import pytest

from memorax.algorithms.r2d2 import (
    double_q_n_step_targets,
    masked_sequence_loss,
    sequence_priorities,
    signed_hyperbolic,
    signed_parabolic,
)


def identity(value):
    return value


def test_signed_transforms_round_trip_and_remain_strictly_monotonic():
    values = jnp.asarray([-100.0, -2.0, 0.0, 3.0, 100.0])

    transformed = signed_hyperbolic(values)

    np.testing.assert_allclose(signed_parabolic(transformed), values, rtol=2e-5)
    assert np.all(np.diff(np.asarray(transformed)) > 0)


def test_double_q_n_step_targets_select_online_actions_and_keep_tail_starts():
    rewards = jnp.asarray([[1.0, 2.0, 3.0]])
    terminals = jnp.asarray([[False, False, False]])
    valid = jnp.asarray([[True, True, True]])
    online_q = jnp.asarray(
        [
            [
                [0.0, 0.0],
                [0.0, 2.0],
                [3.0, 1.0],
                [2.0, 4.0],
            ]
        ]
    )
    target_q = jnp.asarray(
        [
            [
                [0.0, 0.0],
                [8.0, 1.0],
                [5.0, 9.0],
                [9.0, 7.0],
            ]
        ]
    )

    targets = double_q_n_step_targets(
        rewards,
        terminals,
        online_q,
        target_q,
        valid,
        gamma=0.5,
        n_step=2,
        transform=identity,
        inverse_transform=identity,
    )

    expected_t0 = 1.0 + 0.5 * 2.0 + 0.25 * 5.0
    expected_t1 = 2.0 + 0.5 * 3.0 + 0.25 * 7.0
    expected_t2 = 3.0 + 0.5 * 7.0
    np.testing.assert_allclose(targets, [[expected_t0, expected_t1, expected_t2]])


def test_double_q_n_step_targets_invert_bootstrap_and_transform_return():
    targets = double_q_n_step_targets(
        rewards=jnp.asarray([[1.0]]),
        terminals=jnp.asarray([[False]]),
        online_q=jnp.asarray([[[0.0, 0.0], [0.0, 1.0]]]),
        target_q=jnp.asarray([[[0.0, 0.0], [2.3266249, 2.008]]]),
        valid=jnp.asarray([[True]]),
        gamma=0.5,
        n_step=1,
        transform=signed_hyperbolic,
        inverse_transform=signed_parabolic,
    )

    np.testing.assert_allclose(targets, [[1.4544897]], rtol=2e-5)


def test_double_q_n_step_targets_stop_target_gradients():
    rewards = jnp.asarray([[1.0]])
    terminals = jnp.asarray([[False]])
    online_q = jnp.asarray([[[0.0, 0.0], [0.0, 1.0]]])
    target_q = jnp.asarray([[[0.0, 0.0], [10.0, 8.0]]])
    valid = jnp.asarray([[True]])

    gradient = jax.grad(
        lambda candidate: jnp.sum(
            double_q_n_step_targets(
                rewards,
                terminals,
                online_q,
                candidate,
                valid,
                gamma=0.5,
                n_step=1,
                transform=identity,
                inverse_transform=identity,
            )
        )
    )(target_q)

    np.testing.assert_array_equal(gradient, jnp.zeros_like(target_q))


def test_double_q_n_step_targets_shorten_at_valid_padding_boundary():
    rewards = jnp.asarray([[1.0, 2.0, 100.0]])
    terminals = jnp.asarray([[False, False, False]])
    valid = jnp.asarray([[True, True, False]])
    online_q = jnp.asarray([[[0.0, 0.0], [0.0, 2.0], [3.0, 1.0], [2.0, 4.0]]])
    target_q = jnp.asarray([[[0.0, 0.0], [8.0, 1.0], [5.0, 9.0], [9.0, 7.0]]])

    targets = double_q_n_step_targets(
        rewards,
        terminals,
        online_q,
        target_q,
        valid,
        gamma=0.5,
        n_step=2,
        transform=identity,
        inverse_transform=identity,
    )

    expected_t0 = 1.0 + 0.5 * 2.0 + 0.25 * 5.0
    expected_t1 = 2.0 + 0.5 * 5.0
    np.testing.assert_allclose(targets, [[expected_t0, expected_t1, 0.0]])


def test_double_q_n_step_targets_stop_at_terminals_but_not_truncations():
    rewards = jnp.asarray([[1.0, 2.0, 3.0]])
    valid = jnp.asarray([[True, True, True]])
    online_q = jnp.asarray([[[0.0, 0.0], [0.0, 2.0], [3.0, 1.0], [2.0, 4.0]]])
    target_q = jnp.asarray([[[0.0, 0.0], [8.0, 1.0], [5.0, 9.0], [9.0, 7.0]]])

    terminal_targets = double_q_n_step_targets(
        rewards,
        jnp.asarray([[False, True, False]]),
        online_q,
        target_q,
        valid,
        gamma=0.5,
        n_step=2,
        transform=identity,
        inverse_transform=identity,
    )
    truncation_targets = double_q_n_step_targets(
        rewards,
        jnp.asarray([[False, False, False]]),
        online_q,
        target_q,
        valid,
        gamma=0.5,
        n_step=2,
        transform=identity,
        inverse_transform=identity,
    )

    np.testing.assert_allclose(terminal_targets, [[2.0, 2.0, 6.5]])
    np.testing.assert_allclose(truncation_targets, [[3.25, 5.25, 6.5]])
    with pytest.raises(TypeError):
        double_q_n_step_targets(
            rewards,
            terminals=jnp.asarray([[False, False, False]]),
            online_q=online_q,
            target_q=target_q,
            valid=valid,
            done=jnp.asarray([[True, True, True]]),
            gamma=0.5,
            n_step=2,
            transform=identity,
            inverse_transform=identity,
        )


def test_masked_sequence_loss_and_priorities_ignore_invalid_timesteps():
    td_error = jnp.asarray([[1.0, -3.0, 100.0], [2.0, 4.0, 6.0]])
    valid = jnp.asarray([[1, 1, 0], [1, 1, 1]])
    importance_weights = jnp.asarray([0.5, 1.0])

    loss = masked_sequence_loss(td_error, valid, importance_weights)
    priorities = sequence_priorities(td_error, valid, max_weight=0.75)

    per_sequence_half_mse = [
        0.5 * (1.0**2 + 3.0**2) / 2,
        0.5 * (2.0**2 + 4.0**2 + 6.0**2) / 3,
    ]
    expected_loss = np.mean(np.array(per_sequence_half_mse) * [0.5, 1.0])
    expected_priority = [0.75 * 3.0 + 0.25 * 2.0, 0.75 * 6.0 + 0.25 * 4.0]

    np.testing.assert_allclose(loss, expected_loss)
    np.testing.assert_allclose(priorities, expected_priority)
