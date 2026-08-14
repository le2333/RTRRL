import jax
import jax.numpy as jnp
import numpy as np
import pytest

from memorax.algorithms.r2d2 import (
    DuelingQHead,
    QFunction,
    RecurrentInputs,
    encode_recurrent_inputs,
)
from memorax.networks.heads import DiscreteQNetwork


def _inputs(*, episode_start=None):
    if episode_start is None:
        episode_start = [[True, False, False]]
    return RecurrentInputs(
        observation=jnp.asarray([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),
        previous_action=jnp.asarray([[0, 1, 0]]),
        previous_reward=jnp.asarray([[0.5, -1.0, 2.0]]),
        episode_start=jnp.asarray(episode_start),
    )


def _assert_tree_allclose(left, right):
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves):
        np.testing.assert_allclose(left_leaf, right_leaf, rtol=1e-5, atol=1e-6)


def test_encoder_orders_observation_action_reward_and_episode_start():
    inputs = RecurrentInputs(
        observation=jnp.asarray([[[1, 2], [3, 4]]]),
        previous_action=jnp.asarray([[0, 1]]),
        previous_reward=jnp.asarray([[0.5, -1.0]]),
        episode_start=jnp.asarray([[True, False]]),
    )

    encoded = encode_recurrent_inputs(inputs, action_dim=2)

    np.testing.assert_array_equal(
        encoded,
        [[[1.0, 2.0, 1.0, 0.0, 0.5, 1.0],
          [3.0, 4.0, 0.0, 1.0, -1.0, 0.0]]],
    )
    assert jnp.issubdtype(encoded.dtype, jnp.floating)


def test_linear_and_dueling_heads_produce_one_q_per_action():
    hidden = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    linear = DiscreteQNetwork(action_dim=3)
    linear_variables = linear.init(jax.random.key(0), hidden)
    linear_q, _ = linear.apply(linear_variables, hidden)

    dueling = DuelingQHead(action_dim=3)
    dueling_variables = dueling.init(jax.random.key(1), hidden)
    dueling_q = dueling.apply(dueling_variables, hidden)
    params = dueling_variables["params"]
    value = hidden @ params["value"]["kernel"] + params["value"]["bias"]
    advantage = (
        hidden @ params["advantage"]["kernel"] + params["advantage"]["bias"]
    )

    assert linear_q.shape == (2, 3, 3)
    assert dueling_q.shape == (2, 3, 3)
    np.testing.assert_allclose(
        dueling_q,
        value + advantage - advantage.mean(axis=-1, keepdims=True),
    )


@pytest.mark.parametrize("backbone_kind", ["lru", "rtu"])
def test_apply_unroll_and_recurrence_trajectory_share_one_graph(backbone_kind):
    q_function = QFunction(
        action_dim=2,
        feature_dim=4,
        hidden_dim=3,
        backbone_kind=backbone_kind,
        head_kind="dueling",
    )
    inputs = _inputs()
    params, recurrence = q_function.init(jax.random.key(2), inputs)
    one_step = jax.tree.map(lambda value: value[:, :1], inputs)

    applied_recurrence, applied_q = q_function.apply(params, one_step, recurrence)
    unrolled_recurrence, unrolled_q = q_function.unroll(
        params, one_step, recurrence
    )
    final_recurrence, full_q, post_recurrences = (
        q_function._unroll_with_recurrences(params, inputs, recurrence)
    )
    direct_final, direct_q = q_function.unroll(params, inputs, recurrence)

    assert applied_q.shape == (1, 1, 2)
    _assert_tree_allclose(applied_recurrence, unrolled_recurrence)
    np.testing.assert_allclose(applied_q, unrolled_q, rtol=1e-5, atol=1e-6)
    _assert_tree_allclose(final_recurrence, direct_final)
    np.testing.assert_allclose(full_q, direct_q, rtol=1e-5, atol=1e-6)
    for leaf in jax.tree.leaves(post_recurrences):
        assert leaf.shape[:2] == (1, 3)


@pytest.mark.parametrize("backbone_kind", ["lru", "rtu"])
def test_episode_start_resets_before_consuming_the_input(backbone_kind):
    q_function = QFunction(
        action_dim=2,
        feature_dim=4,
        hidden_dim=3,
        backbone_kind=backbone_kind,
        head_kind="linear",
    )
    inputs = _inputs(episode_start=[[False, False, False]])
    params, recurrence = q_function.init(jax.random.key(3), inputs)
    first = jax.tree.map(lambda value: value[:, :1], inputs)
    second = jax.tree.map(lambda value: value[:, 1:2], inputs)
    advanced, _ = q_function.apply(params, first, recurrence)

    reset_second = second.replace(episode_start=jnp.asarray([[True]]))
    _, reset_q = q_function.apply(params, reset_second, advanced)
    fresh = q_function.reset(jax.random.key(4), advanced)
    _, fresh_q = q_function.apply(params, reset_second, fresh)

    np.testing.assert_allclose(reset_q, fresh_q, rtol=1e-5, atol=1e-6)
