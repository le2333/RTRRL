import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from memorax.algorithms.r2d2 import (
    Core,
    LearnerSequence,
    QFunction,
    RecurrentInputs,
)


def recurrence(weight, inputs):
    def step(hidden, value):
        hidden = jnp.tanh(weight * hidden + value)
        return hidden, hidden

    return jax.lax.scan(step, 0.0, inputs)[1]


class ScalarQFunction:
    action_dim = 2

    def __init__(self, initial_recurrence=0.0):
        self.initial_recurrence = initial_recurrence

    def reset(self, key, batch_size):
        del key
        return jnp.full((batch_size,), self.initial_recurrence)

    def _q_values(self, hidden):
        return jnp.stack((hidden, -hidden), axis=-1)

    def apply(self, params, inputs, recurrence):
        value = inputs.observation[:, 0, 0]
        hidden = jnp.tanh(params * recurrence + value)
        return hidden, self._q_values(hidden)[:, None]

    def _unroll_with_recurrences(self, params, inputs, recurrence):
        values = jnp.swapaxes(inputs.observation[..., 0], 0, 1)

        def step(hidden, value):
            hidden = jnp.tanh(params * hidden + value)
            return hidden, (self._q_values(hidden), hidden)

        final, (q_values, post_recurrences) = jax.lax.scan(step, recurrence, values)
        return (
            final,
            jnp.swapaxes(q_values, 0, 1),
            jnp.swapaxes(post_recurrences, 0, 1),
        )

    def unroll(self, params, inputs, recurrence):
        final, q_values, _ = self._unroll_with_recurrences(params, inputs, recurrence)
        return final, q_values


class WideScalarQFunction(ScalarQFunction):
    action_dim = 128

    def _q_values(self, hidden):
        offsets = jnp.arange(self.action_dim, dtype=hidden.dtype) * 0.01
        return hidden[..., None] + offsets


def _identity(value):
    return value


def _core(
    q_function,
    *,
    learning_kind="full_bptt",
    gamma=0.5,
    unroll_length=3,
):
    return Core(
        q_function=q_function,
        optimizer=optax.sgd(0.01),
        gamma=gamma,
        n_step=1,
        burn_in_length=0,
        unroll_length=unroll_length,
        importance_sampling_exponent=0.4,
        max_priority_weight=0.75,
        target_update_period=2,
        transform=_identity,
        inverse_transform=_identity,
        learning_kind=learning_kind,
    )


def _inputs(values):
    batch_size, input_count = values.shape
    return RecurrentInputs(
        observation=values[..., None],
        previous_action=jnp.zeros((batch_size, input_count), dtype=jnp.int32),
        previous_reward=jnp.zeros((batch_size, input_count)),
        episode_start=jnp.asarray([[True] + [False] * (input_count - 1)] * batch_size),
    )


def _sample(
    values,
    rewards,
    *,
    actions=None,
    dones=None,
    terminals=None,
    valid=None,
    initial_recurrence=None,
):
    batch_size, input_count = values.shape
    transition_count = input_count - 1
    if actions is None:
        actions = jnp.zeros((batch_size, transition_count), dtype=jnp.int32)
    if dones is None:
        dones = jnp.asarray([[False] * (transition_count - 1) + [True]] * batch_size)
    if terminals is None:
        terminals = dones
    if valid is None:
        valid = jnp.ones((batch_size, transition_count), dtype=jnp.bool_)
    inputs = _inputs(values)
    return LearnerSequence(
        inputs=inputs,
        bootstrap_inputs=RecurrentInputs(
            observation=values[:, 1:, None],
            previous_action=actions,
            previous_reward=rewards,
            episode_start=jnp.zeros((batch_size, transition_count), dtype=jnp.bool_),
        ),
        actions=actions,
        rewards=rewards,
        dones=dones,
        terminals=terminals,
        valid=valid,
        initial_recurrence=initial_recurrence,
        probabilities=jnp.full((batch_size,), 1.0 / batch_size),
        indices=jnp.arange(batch_size),
        buffer_size=jnp.asarray(8),
    )


def test_full_bptt_gradient_matches_an_explicit_full_unroll():
    core = _core(ScalarQFunction(), gamma=0.5, unroll_length=3)
    values = jnp.asarray([[0.2, -0.1, 0.4, 0.3]])
    rewards = jnp.asarray([[0.1, -0.2, 0.3]])
    terminals = jnp.asarray([[False, False, True]])
    sample = _sample(values, rewards, terminals=terminals)
    weight = jnp.asarray(0.7)
    target_weight = jnp.asarray(0.4)

    def direct_loss(candidate_weight):
        online_hidden = recurrence(candidate_weight, values[0])
        target_hidden = recurrence(target_weight, values[0])
        selected_target = jnp.where(
            online_hidden[1:] >= 0.0,
            target_hidden[1:],
            -target_hidden[1:],
        )
        targets = rewards[0] + 0.5 * (~terminals[0]) * selected_target
        td_error = online_hidden[:-1] - jax.lax.stop_gradient(targets)
        return 0.5 * jnp.mean(jnp.square(td_error))

    expected_gradient = jax.grad(direct_loss)(weight)
    actual_gradient = jax.grad(
        lambda candidate: core._full_bptt_loss(
            candidate,
            target_weight,
            sample,
            jnp.ones((1,)),
        )[0]
    )(weight)

    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-7)
    assert abs(float(actual_gradient)) > 0.0


def test_full_bptt_padding_changes_neither_loss_priority_nor_gradient():
    core = _core(WideScalarQFunction(), gamma=0.5, unroll_length=4)
    values = jnp.asarray([[0.2, -0.1, 0.0, 0.0, 0.0]])
    rewards = jnp.asarray([[0.3, -0.2, 0.0, 0.0]])
    dones = jnp.asarray([[False, True, True, True]])
    terminals = jnp.asarray([[False, True, False, False]])
    valid = jnp.asarray([[True, True, False, False]])
    base = _sample(
        values,
        rewards,
        dones=dones,
        terminals=terminals,
        valid=valid,
    )
    changed = base.replace(
        inputs=base.inputs.replace(
            observation=base.inputs.observation.at[:, 2:].set(1000.0),
            previous_action=base.inputs.previous_action.at[:, 2:].set(99),
            previous_reward=base.inputs.previous_reward.at[:, 2:].set(1000.0),
        ),
        bootstrap_inputs=base.bootstrap_inputs.replace(
            observation=base.bootstrap_inputs.observation.at[:, 2:].set(-1000.0)
        ),
        actions=base.actions.at[:, 2:].set(99),
        rewards=base.rewards.at[:, 2:].set(1000.0),
    )
    params = jnp.asarray(0.7)
    target_params = jnp.asarray(0.4)

    def evaluate(sample):
        (loss, readings), gradient = jax.value_and_grad(
            lambda candidate: core._full_bptt_loss(
                candidate,
                target_params,
                sample,
                jnp.ones((1,)),
            ),
            has_aux=True,
        )(params)
        return loss, readings.priority, gradient

    base_result = evaluate(base)
    changed_result = evaluate(changed)
    for base_value, changed_value in zip(base_result, changed_result):
        np.testing.assert_allclose(base_value, changed_value, rtol=1e-6, atol=1e-7)


def test_full_bptt_uses_fresh_recurrence_instead_of_actor_recurrence():
    values = jnp.asarray([[0.2, -0.1, 0.4, 0.3]])
    sample = _sample(
        values,
        jnp.asarray([[0.1, -0.2, 0.3]]),
        initial_recurrence=jnp.asarray([50.0]),
    )
    params = jnp.asarray(0.7)
    target_params = jnp.asarray(0.4)

    def evaluate(core, learner_sample):
        return core._full_bptt_loss(
            params,
            target_params,
            learner_sample,
            jnp.ones((1,)),
        )

    fresh_core = _core(ScalarQFunction(initial_recurrence=0.0))
    changed_actor = sample.replace(initial_recurrence=jnp.asarray([-75.0]))
    original = evaluate(fresh_core, sample)
    actor_changed = evaluate(fresh_core, changed_actor)
    initialized_changed = evaluate(
        _core(ScalarQFunction(initial_recurrence=0.5)), sample
    )

    np.testing.assert_allclose(original[0], actor_changed[0])
    np.testing.assert_allclose(original[1].q_value, actor_changed[1].q_value)
    assert not np.allclose(original[0], initialized_changed[0])
    assert not np.allclose(original[1].q_value, initialized_changed[1].q_value)


def test_full_bptt_equals_zero_burn_in_tbptt_over_a_complete_episode():
    core = _core(ScalarQFunction(), gamma=0.5, unroll_length=3)
    sample = _sample(
        jnp.asarray([[0.2, -0.1, 0.4, 0.3]]),
        jnp.asarray([[0.1, -0.2, 0.3]]),
        initial_recurrence=jnp.zeros((1,)),
    )
    params = jnp.asarray(0.7)
    target_params = jnp.asarray(0.4)

    def evaluate(loss_method):
        return jax.value_and_grad(
            lambda candidate: loss_method(
                candidate,
                target_params,
                sample,
                jnp.ones((1,)),
            ),
            has_aux=True,
        )(params)

    (full_loss, full_readings), full_gradient = evaluate(core._full_bptt_loss)
    (tbptt_loss, tbptt_readings), tbptt_gradient = evaluate(core._tbptt_loss)

    np.testing.assert_allclose(full_loss, tbptt_loss, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        full_readings.q_value, tbptt_readings.q_value, rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        full_readings.priority,
        tbptt_readings.priority,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(full_gradient, tbptt_gradient, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("backbone_kind", ["lru", "rtu"])
def test_core_reset_derives_the_actor_batch_width(backbone_kind):
    q_function = QFunction(
        action_dim=2,
        feature_dim=4,
        hidden_dim=3,
        backbone_kind=backbone_kind,
        head_kind="linear",
    )
    core = _core(q_function)
    inputs = RecurrentInputs(
        observation=jnp.asarray([[[0.2, -0.1]], [[0.4, 0.3]]]),
        previous_action=jnp.asarray([[0], [1]]),
        previous_reward=jnp.zeros((2, 1)),
        episode_start=jnp.ones((2, 1), dtype=jnp.bool_),
    )
    state = core.init(jax.random.key(8), inputs)

    reset_state = core.reset(jax.random.key(9), state)

    for actual, expected in zip(
        jax.tree.leaves(reset_state.recurrence),
        jax.tree.leaves(state.recurrence),
    ):
        assert actual.shape[0] == 2
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("backbone_kind", ["lru", "rtu"])
@pytest.mark.parametrize("learning_kind", ["tbptt", "full_bptt"])
def test_every_backbone_and_learning_mode_updates_recurrent_parameters(
    backbone_kind, learning_kind
):
    q_function = QFunction(
        action_dim=2,
        feature_dim=4,
        hidden_dim=3,
        backbone_kind=backbone_kind,
        head_kind="dueling",
    )
    core = _core(
        q_function,
        learning_kind=learning_kind,
        gamma=0.5,
        unroll_length=2,
    )
    inputs = RecurrentInputs(
        observation=jnp.asarray([[[0.2, -0.1], [0.4, 0.3], [-0.2, 0.5]]]),
        previous_action=jnp.asarray([[0, 1, 0]]),
        previous_reward=jnp.asarray([[0.0, 0.7, -0.2]]),
        episode_start=jnp.asarray([[True, False, False]]),
    )
    state = core.init(
        jax.random.key(10),
        jax.tree.map(lambda value: value[:, :1], inputs),
    )
    sample = LearnerSequence(
        inputs=inputs,
        bootstrap_inputs=RecurrentInputs(
            observation=inputs.observation[:, 1:],
            previous_action=inputs.previous_action[:, 1:],
            previous_reward=inputs.previous_reward[:, 1:],
            episode_start=jnp.zeros((1, 2), dtype=jnp.bool_),
        ),
        actions=jnp.asarray([[1, 0]]),
        rewards=jnp.asarray([[0.7, -0.2]]),
        dones=jnp.asarray([[False, True]]),
        terminals=jnp.asarray([[False, True]]),
        valid=jnp.asarray([[True, True]]),
        initial_recurrence=(state.recurrence if learning_kind == "tbptt" else None),
        probabilities=jnp.asarray([1.0]),
        indices=jnp.asarray([0]),
        buffer_size=jnp.asarray(8),
    )

    next_state, metrics, _ = core.update_parameters(
        jax.random.key(11), state, sample, step=jnp.asarray(1)
    )

    assert np.isfinite(float(metrics.loss))
    assert np.isfinite(float(metrics.gradient_norm))
    assert float(metrics.gradient_norm) > 0.0
    before = jax.tree_util.tree_leaves_with_path(state.params)
    after = jax.tree.leaves(next_state.params)
    recurrent_gradients = [
        (before_leaf - after_leaf) / 0.01
        for (path, before_leaf), after_leaf in zip(before, after)
        if f"{backbone_kind.upper()}Cell" in "/".join(map(str, path))
    ]
    assert recurrent_gradients
    assert all(
        np.all(np.isfinite(np.asarray(gradient))) for gradient in recurrent_gradients
    )
    assert any(np.any(np.asarray(gradient) != 0.0) for gradient in recurrent_gradients)
