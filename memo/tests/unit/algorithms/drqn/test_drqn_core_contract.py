"""What one DRQN update is allowed to move, and what it must leave alone."""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import optax

from memorax.algorithms.drqn import Core, LearnerSequence, QFunction, RecurrentInputs


def _tree_equal(left, right):
    return all(
        np.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right))
    )


def _core(**overrides):
    settings = {
        "q_function": QFunction(
            action_dim=2,
            observation_dim=2,
            hidden_dim=3,
            feature_dim=4,
            core_kind="lru",
        ),
        "optimizer": optax.adam(0.05),
        "gamma": 0.5,
        "target_update_period": 2,
    }
    settings.update(overrides)
    return Core(**settings)


def _inputs(time):
    return RecurrentInputs(
        observation=jnp.arange(time * 2, dtype=jnp.float32).reshape(1, time, 2),
        episode_start=jnp.asarray([[True] + [False] * (time - 1)]),
    )


def _sample():
    inputs = _inputs(3)
    return LearnerSequence(
        inputs=inputs,
        bootstrap_inputs=RecurrentInputs(
            observation=inputs.observation[:, 1:],
            episode_start=jnp.zeros((1, 2), dtype=jnp.bool_),
        ),
        actions=jnp.asarray([[0, 1]]),
        rewards=jnp.asarray([[1.0, 2.0]]),
        dones=jnp.asarray([[False, True]]),
        terminals=jnp.asarray([[False, True]]),
        valid=jnp.asarray([[True, True]]),
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


def test_the_update_is_not_told_the_environment_step():
    """Nothing in this update is scheduled against environment steps.

    Epsilon is the actor's, and the target period counts learner updates, which
    the core already holds. Being handed the step count would let a later
    change schedule something on it without saying so.
    """

    assert tuple(inspect.signature(Core.update_parameters).parameters) == (
        "self",
        "key",
        "state",
        "sample",
    )


def test_update_moves_the_learner_and_copies_the_target_on_the_period():
    core = _core()
    state = core.init(jax.random.key(2), _inputs(1))
    sample = _sample()

    first, first_metrics = core.update_parameters(jax.random.key(3), state, sample)
    second, second_metrics = core.update_parameters(jax.random.key(4), first, sample)

    assert bool(first_metrics.applied)
    assert bool(second_metrics.applied)
    assert int(first.update_step) == 1
    assert int(second.update_step) == 2
    assert not _tree_equal(first.params, state.params)
    assert not _tree_equal(first.optimizer_state, state.optimizer_state)
    # Period two: the first update leaves the target where it was, and the
    # second replaces it outright rather than averaging towards it.
    assert _tree_equal(first.target_params, state.target_params)
    assert _tree_equal(second.target_params, second.params)
    # The behaviour policy's recurrence is not the learner's business.
    assert _tree_equal(first.recurrence, state.recurrence)
    assert _tree_equal(second.recurrence, state.recurrence)


def test_the_copy_is_hard_and_not_an_average():
    """A Polyak factor below one would land between the two parameter sets."""

    core = _core(target_update_period=1)
    state = core.init(jax.random.key(5), _inputs(1))

    updated, _ = core.update_parameters(jax.random.key(6), state, _sample())

    for copied, online, before in zip(
        jax.tree.leaves(updated.target_params),
        jax.tree.leaves(updated.params),
        jax.tree.leaves(state.target_params),
    ):
        assert np.array_equal(np.asarray(copied), np.asarray(online))
        if not np.allclose(np.asarray(online), np.asarray(before)):
            assert not np.allclose(np.asarray(copied), np.asarray(before))


def test_the_reported_update_carries_no_priority_or_importance_weight():
    """There is nothing to report: uniform replay corrects for nothing."""

    core = _core()
    state = core.init(jax.random.key(7), _inputs(1))

    _, metrics = core.update_parameters(jax.random.key(8), state, _sample())

    fields = set(type(metrics).__dataclass_fields__)
    assert fields == {"applied", "loss", "td_error", "q_value", "gradient_norm"}
    assert float(metrics.gradient_norm) > 0.0
