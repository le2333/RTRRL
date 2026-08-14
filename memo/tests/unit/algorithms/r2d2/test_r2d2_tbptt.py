import jax
import jax.numpy as jnp
import numpy as np
import optax

from memorax.algorithms.r2d2 import (
    Core,
    CoreState,
    LearnerSequence,
    RecurrentInputs,
    _burn_in,
)


class ScalarQFunction:
    def unroll(self, params, inputs, recurrence):
        values = jnp.swapaxes(inputs.observation[..., 0], 0, 1)

        def step(hidden, value):
            hidden = jnp.tanh(params * hidden + value)
            return hidden, jnp.stack((hidden, -hidden), axis=-1)

        final, q_values = jax.lax.scan(step, recurrence, values)
        return final, jnp.swapaxes(q_values, 0, 1)


def _inputs(values):
    batch, time = values.shape
    return RecurrentInputs(
        observation=values[..., None],
        previous_action=jnp.zeros((batch, time), dtype=jnp.int32),
        previous_reward=jnp.zeros((batch, time)),
        episode_start=jnp.zeros((batch, time), dtype=jnp.bool_),
    )


def test_burn_in_matches_the_complete_forward_unroll():
    q_function = ScalarQFunction()
    inputs = _inputs(jnp.asarray([[0.2, -0.1, 0.4, 0.3]]))
    params = jnp.asarray(0.7)
    recurrence = jnp.zeros((1,))
    full_final, full_q = q_function.unroll(params, inputs, recurrence)

    warmed, target_warmed, suffix = _burn_in(
        q_function,
        params,
        params,
        inputs,
        recurrence,
        recurrence,
        burn_in_length=2,
    )
    suffix_final, suffix_q = q_function.unroll(params, suffix, warmed)

    np.testing.assert_allclose(
        warmed, jnp.tanh(0.7 * jnp.tanh(0.2) - 0.1), rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(target_warmed, warmed)
    np.testing.assert_allclose(suffix_q, full_q[:, 2:], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(suffix_final, full_final, rtol=1e-6, atol=1e-6)


def test_burn_in_stops_input_gradients_but_not_recurrent_parameter_gradients():
    q_function = ScalarQFunction()
    values = jnp.asarray([[0.2, -0.1, 0.4, 0.3]])

    def loss(weight, candidate_values):
        inputs = _inputs(candidate_values)
        warmed, _, suffix = _burn_in(
            q_function,
            weight,
            weight,
            inputs,
            jnp.zeros((1,)),
            jnp.zeros((1,)),
            burn_in_length=2,
        )
        _, q_values = q_function.unroll(weight, suffix, warmed)
        return jnp.sum(q_values[..., 0])

    weight_gradient, input_gradient = jax.grad(loss, argnums=(0, 1))(
        jnp.asarray(0.7), values
    )

    np.testing.assert_array_equal(input_gradient[:, :2], 0.0)
    assert np.all(np.abs(np.asarray(input_gradient[:, 2:])) > 0)
    assert abs(float(weight_gradient)) > 0


class IdentifiableQFunction:
    action_dim = 2

    def apply(self, params, inputs, recurrence):
        recurrence = jnp.where(inputs.episode_start[:, 0], 0.0, recurrence)
        recurrence = recurrence + inputs.observation[:, 0, 0]
        value = recurrence + params
        return recurrence, jnp.stack((value, -value), axis=-1)[:, None]

    def _unroll_with_recurrences(self, params, inputs, recurrence):
        time_inputs = jax.tree.map(
            lambda value: jnp.swapaxes(value, 0, 1), inputs
        )

        def step(carry, timestep):
            timestep = jax.tree.map(
                lambda value: jnp.expand_dims(value, axis=1), timestep
            )
            next_carry, q_values = self.apply(params, timestep, carry)
            return next_carry, (q_values[:, 0], next_carry)

        final, (q_values, post_recurrences) = jax.lax.scan(
            step, recurrence, time_inputs
        )
        return (
            final,
            jnp.swapaxes(q_values, 0, 1),
            jnp.swapaxes(post_recurrences, 0, 1),
        )

    def unroll(self, params, inputs, recurrence):
        final, q_values, _ = self._unroll_with_recurrences(
            params, inputs, recurrence
        )
        return final, q_values


def _identity(value):
    return value


def _identifiable_core(*, gamma=0.5, beta=0.4):
    return Core(
        q_function=IdentifiableQFunction(),
        optimizer=optax.sgd(0.1),
        gamma=gamma,
        n_step=1,
        burn_in_length=0,
        unroll_length=2,
        importance_sampling_exponent=beta,
        max_priority_weight=0.75,
        target_update_period=2,
        transform=_identity,
        inverse_transform=_identity,
    )


def _learner_sample(observations, rewards, dones, terminals):
    batch = observations.shape[0]
    transition_count = rewards.shape[1]
    return LearnerSequence(
        inputs=RecurrentInputs(
            observation=observations[..., None],
            previous_action=jnp.zeros(
                observations.shape, dtype=jnp.int32
            ),
            previous_reward=jnp.zeros(observations.shape),
            episode_start=jnp.zeros(observations.shape, dtype=jnp.bool_),
        ),
        bootstrap_inputs=RecurrentInputs(
            observation=jnp.full((batch, transition_count, 1), 9.0),
            previous_action=jnp.zeros(
                (batch, transition_count), dtype=jnp.int32
            ),
            previous_reward=rewards,
            episode_start=jnp.zeros(
                (batch, transition_count), dtype=jnp.bool_
            ),
        ),
        actions=jnp.zeros((batch, transition_count), dtype=jnp.int32),
        rewards=rewards,
        dones=dones,
        terminals=terminals,
        valid=jnp.ones((batch, transition_count), dtype=jnp.bool_),
        initial_recurrence=jnp.zeros((batch,)),
        probabilities=jnp.full((batch,), 1.0 / batch),
        indices=jnp.arange(batch),
        buffer_size=jnp.asarray(8),
    )


def test_alignment_uses_shifted_q_and_history_preserving_truncation_bootstrap():
    core = _identifiable_core()
    sample = _learner_sample(
        observations=jnp.asarray([[1.0, -9.0, 3.0]]),
        rewards=jnp.zeros((1, 2)),
        dones=jnp.asarray([[True, True]]),
        terminals=jnp.asarray([[False, True]]),
    )
    sample = sample.replace(
        inputs=sample.inputs.replace(
            episode_start=jnp.asarray([[True, True, False]])
        )
    )

    current, online_next, target_next, *_ = core._aligned_unroll(
        jnp.asarray(0.0),
        jnp.asarray(100.0),
        sample,
        sample.inputs,
        jnp.zeros((1,)),
        jnp.zeros((1,)),
        transition_start=0,
    )

    np.testing.assert_array_equal(
        current, jnp.asarray([[[1.0, -1.0], [-9.0, 9.0]]])
    )
    np.testing.assert_array_equal(
        online_next[:, 0], jnp.asarray([[1.0, -1.0]])
    )
    np.testing.assert_array_equal(
        target_next[:, 0], jnp.asarray([[101.0, -101.0]])
    )
    np.testing.assert_array_equal(
        online_next[:, 1], jnp.asarray([[10.0, -10.0]])
    )
    np.testing.assert_array_equal(
        target_next[:, 1], jnp.asarray([[110.0, -110.0]])
    )
    np.testing.assert_array_equal(
        online_next[:, 2], jnp.asarray([[-6.0, 6.0]])
    )
    np.testing.assert_array_equal(
        target_next[:, 2], jnp.asarray([[94.0, -94.0]])
    )
    assert float(online_next[0, 1, 0]) not in (9.0, -9.0)


def test_importance_weights_apply_once_at_sequence_reduction():
    core = _identifiable_core(gamma=0.0, beta=1.0)
    sample = _learner_sample(
        observations=jnp.zeros((2, 3)),
        rewards=jnp.asarray([[1.0, 1.0], [2.0, 2.0]]),
        dones=jnp.asarray([[False, True], [False, True]]),
        terminals=jnp.asarray([[False, True], [False, True]]),
    ).replace(
        probabilities=jnp.asarray([0.5, 0.25]),
    )
    state = CoreState(
        update_step=jnp.asarray(0),
        recurrence=jnp.zeros((2,)),
        params=jnp.asarray(0.0),
        target_params=jnp.asarray(0.0),
        optimizer_state=core.optimizer.init(jnp.asarray(0.0)),
    )

    _, metrics, priorities = core.update_parameters(
        jax.random.key(0),
        state,
        sample,
        step=jnp.asarray(1),
    )

    np.testing.assert_allclose(metrics.loss, 1.125)
    np.testing.assert_allclose(metrics.importance_weight, 0.75)
    np.testing.assert_allclose(priorities, jnp.asarray([1.0, 2.0]))


def test_update_metrics_ignore_invalid_padded_transitions():
    core = _identifiable_core(gamma=0.0)
    sample = _learner_sample(
        observations=jnp.asarray([[1.0, 100.0, 0.0]]),
        rewards=jnp.zeros((1, 2)),
        dones=jnp.asarray([[False, True]]),
        terminals=jnp.asarray([[False, True]]),
    ).replace(valid=jnp.asarray([[True, False]]))
    state = CoreState(
        update_step=jnp.asarray(0),
        recurrence=jnp.zeros((1,)),
        params=jnp.asarray(0.0),
        target_params=jnp.asarray(0.0),
        optimizer_state=core.optimizer.init(jnp.asarray(0.0)),
    )

    _, metrics, _ = core.update_parameters(
        jax.random.key(1),
        state,
        sample,
        step=jnp.asarray(1),
    )

    np.testing.assert_allclose(metrics.td_error, 1.0)
    np.testing.assert_allclose(metrics.q_value, 1.0)


def test_core_uses_online_action_selection_and_target_action_evaluation():
    core = _identifiable_core(gamma=0.5)
    sample = _learner_sample(
        observations=jnp.asarray([[1.0, 0.0, 0.0]]),
        rewards=jnp.zeros((1, 2)),
        dones=jnp.asarray([[False, True]]),
        terminals=jnp.asarray([[False, True]]),
    )

    _, readings = core._tbptt_loss(
        jnp.asarray(-5.0),
        jnp.asarray(100.0),
        sample,
        jnp.ones((1,)),
    )

    np.testing.assert_allclose(readings.q_value[:, 0], -4.0)
    np.testing.assert_allclose(readings.td_error[:, 0], 46.5)
