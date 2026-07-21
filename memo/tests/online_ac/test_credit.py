from dataclasses import FrozenInstanceError, fields

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from golden import assert_tree_allclose

from memorax.networks import RNN, LRUCell, LRUConfig, Memoroid, RTUCell, RTUConfig
from memorax.online_ac import (
    ActionDecision,
    AgentProgram,
    JAXEnvAdapter,
    MetaProgramConfig,
    StandardProgramConfig,
    Transition,
    make_exact_rtrl_credit,
    make_td0,
)
from memorax.utils.axes import add_feature_axis, broadcast_done, reset_carry


def _identity(*args):
    return args


def test_td0_uses_explicit_bootstrap_discount():
    td = make_td0()

    delta = td(
        reward=jnp.array([2.0]),
        value=jnp.array([1.0]),
        next_value=jnp.array([4.0]),
        bootstrap_discount=jnp.array([0.5]),
    )

    np.testing.assert_allclose(delta, [3.0])


def test_agent_program_is_a_frozen_host_only_type():
    program = AgentProgram(
        init_fn=_identity,
        train_epoch_fn=_identity,
        evaluate_fn=_identity,
        state_schema={"step": "int32"},
        metric_schema={"loss": "float32"},
    )

    assert [field.name for field in fields(program)] == [
        "init_fn",
        "train_epoch_fn",
        "evaluate_fn",
        "state_schema",
        "metric_schema",
    ]
    assert jax.tree.leaves(program) == [program]
    with pytest.raises(FrozenInstanceError):
        setattr(program, "state_schema", {})


def test_jax_env_adapter_has_explicit_jax_boundary_fields():
    env_params = jnp.array([3], dtype=jnp.int32)
    adapter = JAXEnvAdapter(
        reset_fn=_identity,
        step_fn=_identity,
        env_params=env_params,
        build_context={"action_shape": (2,)},
    )

    assert [field.name for field in fields(adapter)] == [
        "reset_fn",
        "step_fn",
        "env_params",
        "build_context",
    ]
    assert callable(adapter.reset_fn)
    assert callable(adapter.step_fn)
    assert adapter.env_params is env_params
    assert jax.tree.leaves(adapter) == [adapter]


def test_action_decision_names_all_action_semantics():
    assert [field.name for field in fields(ActionDecision)] == [
        "sampled_action",
        "logprob_action",
        "env_action",
        "bootstrap_feedback_action",
        "persisted_feedback_action",
    ]


def test_runtime_transition_types_are_jax_pytrees():
    actions = [jnp.array(index) for index in range(5)]
    decision = ActionDecision(*actions)
    transition = Transition(
        observation=jnp.array([1.0]),
        action_decision=decision,
        reward=jnp.array(2.0),
        done=jnp.array(False),
        next_observation=jnp.array([3.0]),
        bootstrap_discount=jnp.array(0.9),
        info={"count": jnp.array(1)},
    )

    assert jax.tree.leaves(decision) == actions
    assert len(jax.tree.leaves(transition)) == 11


@pytest.mark.parametrize(
    "type_",
    [ActionDecision, Transition, MetaProgramConfig, StandardProgramConfig],
)
def test_fixed_program_types_are_frozen(type_):
    assert type_.__dataclass_params__.frozen


def _lru_case():
    core = Memoroid(
        cell=LRUCell(config=LRUConfig(features=2, hidden_dim=2, output_dim=2))
    )
    inputs = jnp.array([[[0.2, -0.4]]], dtype=jnp.float32)
    done = jnp.array([[False]])
    carry = core.initialize_carry(jax.random.key(1), (1, 2))
    variables = core.init(jax.random.key(2), inputs, done, carry)
    sensitivity = core.initialize_sensitivity(jax.random.key(3), (1, 2))
    return core, variables["params"], inputs, done, carry, sensitivity


def _rtu_case():
    core = RNN(cell=RTUCell(config=RTUConfig(features=2, hidden_dim=2)))
    inputs = jnp.array([[[0.3, -0.1]]], dtype=jnp.float32)
    done = jnp.array([[False]])
    carry = core.initialize_carry(jax.random.key(4), (1, 2))
    variables = core.init(jax.random.key(5), inputs, done, carry)
    sensitivity = core.initialize_sensitivity(jax.random.key(6), (1, 2))
    return core, variables["params"], inputs, done, carry, sensitivity


def _real_tree_sum(tree):
    total = jnp.asarray(0.0)
    for index, leaf in enumerate(jax.tree.leaves(tree), start=1):
        total = total + index * jnp.real(leaf).sum()
        if jnp.iscomplexobj(leaf):
            total = total + (index + 0.25) * jnp.imag(leaf).sum()
    return total


def test_lru_credit_delegates_to_memoroid_local_jacobian():
    core, params, inputs, done, carry, sensitivity = _lru_case()
    credit = make_exact_rtrl_credit(core)

    actual = credit(params, inputs, done, carry, sensitivity)
    expected = core.apply(
        {"params": params},
        inputs,
        done,
        carry,
        sensitivity=sensitivity,
        method="local_jacobian",
    )

    assert_tree_allclose(actual, expected)
    assert_tree_allclose(credit.initialize(jax.random.key(7), (1, 2)), sensitivity)

    adapter_value, adapter_grad = jax.value_and_grad(
        lambda p: _real_tree_sum(credit(p, inputs, done, carry, sensitivity))
    )(params)
    legacy_value, legacy_grad = jax.value_and_grad(
        lambda p: _real_tree_sum(
            core.apply(
                {"params": p},
                inputs,
                done,
                carry,
                sensitivity=sensitivity,
                method="local_jacobian",
            )
        )
    )(params)
    np.testing.assert_allclose(adapter_value, legacy_value, rtol=1e-6, atol=1e-7)
    assert_tree_allclose(adapter_grad, legacy_grad)
    omitted_credit_grad = jax.grad(
        lambda p: _real_tree_sum(
            core.apply(
                {"params": p},
                inputs,
                done,
                carry,
                sensitivity=None,
                method="local_jacobian",
            )
        )
    )(params)
    with pytest.raises(AssertionError):
        assert_tree_allclose(adapter_grad, omitted_credit_grad)


def test_rtu_credit_delegates_to_rnn_local_jacobian():
    core, params, inputs, done, carry, sensitivity = _rtu_case()
    credit = make_exact_rtrl_credit(core)

    actual = credit(params, inputs, done, carry, sensitivity)
    expected = core.apply(
        {"params": params},
        inputs,
        done,
        carry,
        sensitivity=sensitivity,
        method="local_jacobian",
    )

    assert_tree_allclose(actual, expected)
    assert_tree_allclose(credit.initialize(jax.random.key(8), (1, 2)), sensitivity)

    adapter_value, adapter_grad = jax.value_and_grad(
        lambda p: _real_tree_sum(credit(p, inputs, done, carry, sensitivity))
    )(params)
    legacy_value, legacy_grad = jax.value_and_grad(
        lambda p: _real_tree_sum(
            core.apply(
                {"params": p},
                inputs,
                done,
                carry,
                sensitivity=sensitivity,
                method="local_jacobian",
            )
        )
    )(params)
    np.testing.assert_allclose(adapter_value, legacy_value, rtol=1e-6, atol=1e-7)
    assert_tree_allclose(adapter_grad, legacy_grad)
    omitted_credit_grad = jax.grad(
        lambda p: _real_tree_sum(
            core.apply(
                {"params": p},
                inputs,
                done,
                carry,
                sensitivity=None,
                method="local_jacobian",
            )
        )
    )(params)
    with pytest.raises(AssertionError):
        assert_tree_allclose(adapter_grad, omitted_credit_grad)


def test_phantom_changes_gradient_but_not_forward_value():
    core, params, inputs, done, carry, sensitivity = _lru_case()
    credit = make_exact_rtrl_credit(core)
    carry, _, sensitivity = credit(params, inputs, done, carry, sensitivity)
    next_inputs = jnp.array([[[-0.7, 0.6]]], dtype=jnp.float32)

    def value_with_credit(p):
        _, value, _ = credit(p, next_inputs, done, carry, sensitivity)
        return value.sum()

    def value_without_credit(p):
        _, value, _ = credit(p, next_inputs, done, carry, None)
        return value.sum()

    with_value, with_grad = jax.value_and_grad(value_with_credit)(params)
    without_value, without_grad = jax.value_and_grad(value_without_credit)(params)

    np.testing.assert_allclose(with_value, without_value, rtol=1e-6, atol=1e-7)
    assert any(
        not np.allclose(a, b)
        for a, b in zip(jax.tree.leaves(with_grad), jax.tree.leaves(without_grad))
    )


def _rtu_controlled_reset_forward(
    core, params, inputs, done, carry, sensitivity, *, phantom_before_reset
):
    cell_variables = {"params": params["cell"]}
    initial_carry = core.cell.initialize_carry(jax.random.key(0), (inputs.shape[0], 2))
    done_t = done[:, 0]
    input_t = inputs[:, 0]
    phantom = core.cell.apply(cell_variables, sensitivity, method="compute_phantom")

    if phantom_before_reset:
        carry = core.cell.apply(cell_variables, carry, phantom, method="inject_phantom")
        carry = reset_carry(done_t, carry, initial_carry)
    else:
        carry = reset_carry(done_t, carry, initial_carry)
        carry = core.cell.apply(cell_variables, carry, phantom, method="inject_phantom")

    sensitivity = jax.tree.map(
        lambda value: jnp.where(
            broadcast_done(add_feature_axis(done_t), value), 0, value
        ),
        sensitivity,
    )
    next_carry, output, next_sensitivity = core.cell.apply(
        cell_variables,
        carry,
        input_t,
        sensitivity,
        method="local_jacobian",
    )
    return next_carry, output[:, None], next_sensitivity


def test_rtu_reset_order_matches_legacy():
    core, params, inputs, _, initial_carry, initial_sensitivity = _rtu_case()
    credit = make_exact_rtrl_credit(core)
    done = jnp.array([[True]])
    carry = jax.tree.map(lambda value: value + 0.75, initial_carry)
    sensitivity = jax.tree.map(lambda value: value + 0.5, initial_sensitivity)

    legacy = credit(params, inputs, done, carry, sensitivity)
    phantom_then_reset = _rtu_controlled_reset_forward(
        core,
        params,
        inputs,
        done,
        carry,
        sensitivity,
        phantom_before_reset=True,
    )
    reset_then_phantom = _rtu_controlled_reset_forward(
        core,
        params,
        inputs,
        done,
        carry,
        sensitivity,
        phantom_before_reset=False,
    )

    assert_tree_allclose(legacy, phantom_then_reset)
    assert_tree_allclose(legacy, reset_then_phantom)

    legacy_grad = jax.grad(
        lambda p: _real_tree_sum(credit(p, inputs, done, carry, sensitivity))
    )(params)
    phantom_then_reset_grad = jax.grad(
        lambda p: _real_tree_sum(
            _rtu_controlled_reset_forward(
                core,
                p,
                inputs,
                done,
                carry,
                sensitivity,
                phantom_before_reset=True,
            )
        )
    )(params)
    reset_then_phantom_grad = jax.grad(
        lambda p: _real_tree_sum(
            _rtu_controlled_reset_forward(
                core,
                p,
                inputs,
                done,
                carry,
                sensitivity,
                phantom_before_reset=False,
            )
        )
    )(params)

    assert_tree_allclose(legacy_grad, phantom_then_reset_grad)
    with pytest.raises(AssertionError):
        assert_tree_allclose(legacy_grad, reset_then_phantom_grad)


def test_bootstrap_state_can_be_discarded():
    core, params, inputs, done, pre_acting_carry, pre_acting_credit = _lru_case()
    exact = make_exact_rtrl_credit(core)
    acting_carry, _, acting_credit = exact(
        params, inputs, done, pre_acting_carry, pre_acting_credit
    )
    post_acting_state = (acting_carry, acting_credit)
    preserved_acting_state = jax.tree.map(lambda value: value.copy(), post_acting_state)
    bootstrap_inputs = jnp.array([[[0.9, -0.8]]], dtype=jnp.float32)
    bootstrap_params = jax.lax.stop_gradient(params)

    bootstrap_from_post = exact(
        bootstrap_params,
        bootstrap_inputs,
        done,
        jax.lax.stop_gradient(acting_carry),
        jax.lax.stop_gradient(acting_credit),
    )
    bootstrap_from_pre = exact(
        bootstrap_params,
        bootstrap_inputs,
        done,
        jax.lax.stop_gradient(pre_acting_carry),
        jax.lax.stop_gradient(pre_acting_credit),
    )

    with pytest.raises(AssertionError):
        assert_tree_allclose(bootstrap_from_post, bootstrap_from_pre)
    assert_tree_allclose(post_acting_state, preserved_acting_state)


def test_differentiation_forward_replays_pre_acting_state():
    core, params, inputs, done, pre_acting_carry, pre_acting_credit = _rtu_case()
    exact = make_exact_rtrl_credit(core)

    acting = exact(params, inputs, done, pre_acting_carry, pre_acting_credit)
    differentiation = exact(params, inputs, done, pre_acting_carry, pre_acting_credit)
    from_post_acting = exact(params, inputs, done, acting[0], acting[2])

    assert_tree_allclose(differentiation, acting)
    with pytest.raises(AssertionError):
        assert_tree_allclose(from_post_acting, acting)
