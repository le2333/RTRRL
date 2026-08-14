import inspect

import jax
import jax.numpy as jnp
import numpy as np
import optax

from memorax.algorithms.r2d2 import (
    Core,
    LearnerSequence,
    QFunction,
    RecurrentInputs,
    signed_hyperbolic,
    signed_parabolic,
)


def _tree_equal(left, right):
    return all(
        np.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right))
    )


def _core():
    return Core(
        q_function=QFunction(
            action_dim=2,
            feature_dim=4,
            hidden_dim=3,
            backbone_kind="rtu",
            head_kind="linear",
        ),
        optimizer=optax.adam(0.05),
        gamma=0.5,
        n_step=1,
        burn_in_length=0,
        unroll_length=2,
        importance_sampling_exponent=0.4,
        max_priority_weight=0.75,
        target_update_period=2,
        transform=signed_hyperbolic,
        inverse_transform=signed_parabolic,
    )


def _inputs(time):
    return RecurrentInputs(
        observation=jnp.arange(time * 2, dtype=jnp.float32).reshape(1, time, 2),
        previous_action=jnp.zeros((1, time), dtype=jnp.int32),
        previous_reward=jnp.zeros((1, time)),
        episode_start=jnp.asarray([[True] + [False] * (time - 1)]),
    )


def _sample(initial_recurrence):
    inputs = _inputs(3)
    return LearnerSequence(
        inputs=inputs,
        bootstrap_inputs=RecurrentInputs(
            observation=inputs.observation[:, 1:],
            previous_action=jnp.zeros((1, 2), dtype=jnp.int32),
            previous_reward=jnp.asarray([[1.0, 2.0]]),
            episode_start=jnp.zeros((1, 2), dtype=jnp.bool_),
        ),
        actions=jnp.asarray([[0, 1]]),
        rewards=jnp.asarray([[1.0, 2.0]]),
        dones=jnp.asarray([[False, True]]),
        terminals=jnp.asarray([[False, True]]),
        valid=jnp.asarray([[True, True]]),
        initial_recurrence=initial_recurrence,
        probabilities=jnp.asarray([1.0]),
        indices=jnp.asarray([3]),
        buffer_size=jnp.asarray(8),
    )


def test_act_only_advances_recurrence():
    core = _core()
    timestep = _inputs(1)
    state = core.init(jax.random.key(0), timestep)

    recurrence, action, metrics = core.act(
        jax.random.key(1), state, timestep, epsilon=jnp.asarray(0.25)
    )

    assert action.shape == (1,)
    assert metrics.selected_q.shape == (1,)
    assert float(metrics.epsilon) == 0.25
    assert not _tree_equal(recurrence, state.recurrence)
    assert _tree_equal(state.params, state.target_params)
    assert int(state.update_step) == 0
    assert tuple(inspect.signature(Core.update_parameters).parameters) == (
        "self",
        "key",
        "state",
        "sample",
        "step",
    )


def test_update_changes_only_learner_state_and_targets_on_update_two():
    core = _core()
    state = core.init(jax.random.key(2), _inputs(1))
    sample = _sample(state.recurrence)

    first, first_metrics, first_priorities = core.update_parameters(
        jax.random.key(3), state, sample, step=jnp.asarray(1)
    )
    second, second_metrics, second_priorities = core.update_parameters(
        jax.random.key(4), first, sample, step=jnp.asarray(2)
    )

    assert bool(first_metrics.applied)
    assert bool(second_metrics.applied)
    assert first_priorities.shape == (1,)
    assert second_priorities.shape == (1,)
    assert int(first.update_step) == 1
    assert int(second.update_step) == 2
    assert not _tree_equal(first.params, state.params)
    assert not _tree_equal(first.optimizer_state, state.optimizer_state)
    assert _tree_equal(first.target_params, state.target_params)
    assert _tree_equal(second.target_params, second.params)
    assert _tree_equal(first.recurrence, state.recurrence)
    assert _tree_equal(second.recurrence, state.recurrence)
