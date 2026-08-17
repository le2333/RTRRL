from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flashbax.buffers import sum_tree

from memorax.algorithms.r2d2 import (
    METRICS,
    OBSERVATIONS,
    R2D2,
    REPORTS,
    TRAINING_METRICS,
    Core,
    QFunction,
    R2D2Config,
    RecurrentInputs,
    Reports,
    signed_hyperbolic,
    signed_parabolic,
    tbptt_starts,
)
from memorax.buffers import make_prioritised_episode_buffer
from tests.support.environments import TinyDiscreteEnv
from tests.support.replay import start_flags


def _tree_allclose(left, right):
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    return len(left_leaves) == len(right_leaves) and all(
        np.allclose(left_leaf, right_leaf)
        for left_leaf, right_leaf in zip(left_leaves, right_leaves)
    )


@dataclass(frozen=True)
class TruncatingTinyDiscreteEnv(TinyDiscreteEnv):
    def step(self, key, state, action, params):
        observation, state, reward, _, info = super().step(key, state, action, params)
        return (
            observation,
            state,
            reward,
            jnp.asarray(True),
            {
                **info,
                "terminal": jnp.asarray(False),
            },
        )


def _algorithm(
    *,
    num_envs=1,
    minimum_size=8,
    evaluation_epsilon=0.0,
    env=None,
    reports=REPORTS,
    record=OBSERVATIONS.trajectory_fields,
):
    env = env or TinyDiscreteEnv()
    buffer = make_prioritised_episode_buffer(
        max_length=16,
        min_length=minimum_size,
        sample_batch_size=1,
        sample_sequence_length=2,
        get_start_flags=start_flags(tbptt_starts, burn_in_length=0),
        add_sequences=False,
        add_batch_size=num_envs,
    )
    core = Core(
        q_function=QFunction(
            action_dim=2,
            feature_dim=4,
            hidden_dim=3,
            backbone_kind="rtu",
            head_kind="linear",
        ),
        optimizer=optax.adam(0.01),
        gamma=0.9,
        n_step=1,
        burn_in_length=0,
        unroll_length=2,
        importance_sampling_exponent=0.4,
        max_priority_weight=0.9,
        target_update_period=2,
        transform=signed_hyperbolic,
        inverse_transform=signed_parabolic,
    )
    return R2D2(
        cfg=R2D2Config(
            num_envs=num_envs,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_steps=4,
            evaluation_epsilon=evaluation_epsilon,
        ),
        env=env,
        env_params=env.default_params,
        core=core,
        buffer=buffer,
        reports=reports,
        record=record,
    )


def test_train_step_records_the_executed_interaction_and_actor_recurrence():
    algorithm = _algorithm()
    initial = algorithm.init(jax.random.key(0))

    state, metrics = algorithm.train_step(initial, jax.random.key(1))

    stored = jax.tree.map(lambda value: value[:, 0], state.buffer_state.experience)
    np.testing.assert_array_equal(stored.observation, [[-0.25, 0.5]])
    np.testing.assert_array_equal(stored.previous_action, [0])
    np.testing.assert_array_equal(stored.previous_reward, [0.0])
    np.testing.assert_array_equal(stored.episode_start, [True])
    np.testing.assert_array_equal(stored.action, metrics.interaction.action)
    np.testing.assert_allclose(stored.reward, metrics.interaction.reward)
    np.testing.assert_allclose(
        stored.next_observation, metrics.interaction.next_observation
    )
    np.testing.assert_array_equal(stored.done, [False])
    np.testing.assert_array_equal(stored.terminal, [False])
    assert _tree_allclose(stored.actor_recurrence, initial.core.recurrence)
    assert int(state.step) == 1


def test_update_runs_once_at_the_first_sampleable_step_and_writes_priority():
    algorithm = _algorithm(minimum_size=2)
    initial = algorithm.init(jax.random.key(2))

    collecting, collecting_metrics = algorithm.train_step(initial, jax.random.key(3))

    assert not bool(algorithm.buffer.can_sample(collecting.buffer_state))
    assert not bool(collecting_metrics.update.applied)
    for item in fields(collecting_metrics.update):
        if item.name != "applied":
            assert float(getattr(collecting_metrics.update, item.name)) == 0.0
    assert _tree_allclose(collecting.core.params, initial.core.params)
    assert _tree_allclose(collecting.core.target_params, initial.core.target_params)
    assert _tree_allclose(collecting.core.optimizer_state, initial.core.optimizer_state)
    assert int(collecting.core.update_step) == 0

    updated, update_metrics = algorithm.train_step(collecting, jax.random.key(4))

    assert bool(update_metrics.update.applied)
    assert int(updated.core.update_step) == 1
    stored_priority = sum_tree.get(
        updated.buffer_state.sum_tree_state, jnp.asarray([0])
    )[0]
    expected_priority = (update_metrics.update.priority + 1e-6) ** 0.6
    np.testing.assert_allclose(stored_priority, expected_priority, rtol=1e-5)


def test_episode_end_clears_feedback_and_resets_before_the_reset_observation():
    algorithm = _algorithm()
    state = algorithm.init(jax.random.key(5))
    for seed in (6, 7, 8):
        state, _ = algorithm.train_step(state, jax.random.key(seed))

    assert bool(state.timestep.done[0])
    assert bool(state.episode_start[0])
    np.testing.assert_array_equal(state.timestep.action, [0])
    np.testing.assert_array_equal(state.timestep.reward, [0.0])

    step_key = jax.random.key(9)
    _, action_key, _, _ = jax.random.split(step_key, 4)
    reset_timestep = state.timestep.replace(
        obs=jnp.asarray([[-0.25, 0.5]], dtype=jnp.float32)
    )
    fresh_core = algorithm.core.reset(jax.random.key(10), state.core)
    expected_recurrence, _, _ = algorithm.core.act(
        action_key,
        fresh_core,
        RecurrentInputs(
            observation=reset_timestep.obs[:, None],
            previous_action=jnp.zeros((1, 1), dtype=jnp.int32),
            previous_reward=jnp.zeros((1, 1), dtype=jnp.float32),
            episode_start=jnp.ones((1, 1), dtype=jnp.bool_),
        ),
        epsilon=jnp.asarray(0.0),
    )

    restarted, metrics = algorithm.train_step(state, step_key)

    stored = jax.tree.map(lambda value: value[:, 3], restarted.buffer_state.experience)
    np.testing.assert_allclose(metrics.interaction.observation, [[-0.25, 0.5]])
    np.testing.assert_array_equal(stored.previous_action, [0])
    np.testing.assert_array_equal(stored.previous_reward, [0.0])
    np.testing.assert_array_equal(stored.episode_start, [True])
    assert _tree_allclose(restarted.core.recurrence, expected_recurrence)


def test_replay_preserves_time_limit_done_without_terminal():
    algorithm = _algorithm(env=TruncatingTinyDiscreteEnv())
    initial = algorithm.init(jax.random.key(11))

    state, metrics = algorithm.train_step(initial, jax.random.key(12))

    stored = jax.tree.map(lambda value: value[:, 0], state.buffer_state.experience)
    np.testing.assert_array_equal(stored.done, [True])
    np.testing.assert_array_equal(stored.terminal, [False])
    np.testing.assert_array_equal(metrics.interaction.done, [True])
    np.testing.assert_array_equal(metrics.interaction.terminal, [False])


def test_evaluation_uses_fresh_interaction_state_and_leaves_training_unchanged():
    algorithm = _algorithm(minimum_size=2, evaluation_epsilon=0.375)
    state = algorithm.init(jax.random.key(13))
    for seed in (14, 15):
        state, _ = algorithm.train_step(state, jax.random.key(seed))
    assert int(state.core.update_step) == 1

    snapshot = {
        "params": jax.tree.map(np.array, state.core.params),
        "target_params": jax.tree.map(np.array, state.core.target_params),
        "optimizer_state": jax.tree.map(np.array, state.core.optimizer_state),
        "buffer_state": jax.tree.map(np.array, state.buffer_state),
        "step": np.array(state.step),
        "update_step": np.array(state.core.update_step),
    }
    altered = state.replace(
        core=state.core.replace(
            recurrence=jax.tree.map(lambda value: value + 100.0, state.core.recurrence)
        )
    )

    key = jax.random.key(16)
    _, metrics = algorithm.evaluate(
        key, algorithm.open_evaluation(key, state), num_steps=2
    )
    _, altered_metrics = algorithm.evaluate(
        key, algorithm.open_evaluation(key, altered), num_steps=2
    )

    assert metrics.update is None
    np.testing.assert_allclose(metrics.forward.epsilon, [0.375, 0.375])
    np.testing.assert_allclose(metrics.interaction.observation[0], [[-0.25, 0.5]])
    np.testing.assert_allclose(
        metrics.forward.selected_q, altered_metrics.forward.selected_q
    )
    np.testing.assert_array_equal(
        metrics.interaction.action, altered_metrics.interaction.action
    )
    assert _tree_allclose(state.core.params, snapshot["params"])
    assert _tree_allclose(state.core.target_params, snapshot["target_params"])
    assert _tree_allclose(state.core.optimizer_state, snapshot["optimizer_state"])
    assert _tree_allclose(state.buffer_state, snapshot["buffer_state"])
    np.testing.assert_array_equal(state.step, snapshot["step"])
    np.testing.assert_array_equal(state.core.update_step, snapshot["update_step"])


def test_readings_declare_and_gate_optional_training_series():
    expected_series = (
        "forward.selected_q",
        "forward.epsilon",
        "update.loss",
        "update.td_error",
        "update.q_value",
        "update.gradient_norm",
        "update.importance_weight",
        "update.priority",
    )
    assert TRAINING_METRICS == expected_series
    assert OBSERVATIONS.reward == "interaction.reward"
    assert OBSERVATIONS.done == "interaction.done"
    assert OBSERVATIONS.terminal == "interaction.terminal"
    assert OBSERVATIONS.series == expected_series
    assert OBSERVATIONS.episode_fields == frozenset(
        (
            OBSERVATIONS.reward,
            OBSERVATIONS.done,
            OBSERVATIONS.terminal,
            *OBSERVATIONS.series,
        )
    )
    assert OBSERVATIONS.trajectory_fields == frozenset(
        (
            OBSERVATIONS.observation,
            OBSERVATIONS.next_observation,
            OBSERVATIONS.action,
        )
    )
    assert "train/episode/selected_q" not in METRICS
    assert "train/episode/forward.selected_q" in METRICS

    reports = Reports(
        selected_q=False,
        epsilon=True,
        loss=False,
        td_error=True,
        q_value=False,
        gradient_norm=False,
        importance_weight=False,
        priority=True,
    )
    algorithm = _algorithm(
        minimum_size=2,
        reports=reports,
        record=frozenset(),
    )
    state = algorithm.init(jax.random.key(17))
    collecting, collecting_metrics = algorithm.train_step(state, jax.random.key(18))
    _, update_metrics = algorithm.train_step(collecting, jax.random.key(19))

    assert collecting_metrics.interaction.observation is None
    assert collecting_metrics.forward.selected_q is None
    assert float(collecting_metrics.forward.epsilon) == 0.0
    assert collecting_metrics.update.loss is None
    assert float(collecting_metrics.update.td_error) == 0.0
    assert bool(update_metrics.update.applied)
    assert update_metrics.update.loss is None
    assert update_metrics.update.td_error is not None
    assert update_metrics.update.q_value is None
    assert update_metrics.update.gradient_norm is None
    assert update_metrics.update.importance_weight is None
    assert update_metrics.update.priority is not None
