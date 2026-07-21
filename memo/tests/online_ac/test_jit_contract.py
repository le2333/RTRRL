import inspect
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from memorax.networks import RTUCell
from memorax.online_ac.types import AgentProgram, EvalSummary


def _make_program(kind, rtrrl_agent_factory, stream_ac_agent_factory):
    if kind == "meta":
        from memorax.online_ac.meta import make_meta_program

        parts = rtrrl_agent_factory(fresh_trace=False)
        return make_meta_program(parts, parts.cfg)

    from memorax.online_ac.standard import make_standard_program

    parts = stream_ac_agent_factory(adaptive=False)
    return make_standard_program(parts, parts.cfg)


def _signature(tree):
    return (
        jax.tree.structure(tree),
        tuple(
            (np.shape(leaf), np.asarray(leaf).dtype) for leaf in jax.tree.leaves(tree)
        ),
    )


def _block(tree):
    return jax.tree.map(
        lambda leaf: leaf.block_until_ready() if isinstance(leaf, jax.Array) else leaf,
        tree,
    )


def _count_traces(fn: Callable[..., Any]):
    traces = 0

    def counted(*args, **kwargs):
        nonlocal traces
        traces += 1
        return fn(*args, **kwargs)

    def trace_count():
        return traces

    return counted, trace_count


def _assert_mutated(left, right):
    assert any(
        not np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def _assert_array_state(state):
    leaves = jax.tree.leaves(state)
    assert leaves
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)


def _force_evaluation_action(state, direction):
    is_meta = hasattr(state, "params")
    parameters = state.params if is_meta else state.actor_params

    def force_actor_head(path, leaf):
        path_text = jax.tree_util.keystr(path)
        actor_head = "['actor']" in path_text if is_meta else "['head']" in path_text
        if not actor_head:
            return leaf
        if path_text.endswith("['kernel']"):
            return jnp.zeros_like(leaf)
        if not path_text.endswith("['bias']"):
            return leaf
        forced = jnp.zeros_like(leaf)
        if is_meta:
            action_dim = forced.shape[-1] // 2
            return forced.at[:action_dim].set(100.0 * direction)
        return forced.at[0].set(100.0 * direction).at[1].set(-100.0 * direction)

    forced_parameters = jax.tree.map_with_path(force_actor_head, parameters)
    return (
        state.replace(params=forced_parameters)
        if is_meta
        else state.replace(actor_params=forced_parameters)
    )


def _evaluation_derived_output(state):
    return (
        state.timestep.obs,
        state.timestep.action,
        state.env_state.observation,
    )


def _assert_evaluation_computation_sensitive(
    evaluate, key, positive_policy, negative_policy, num_steps
):
    positive_output = evaluate(key, positive_policy, num_steps)
    negative_output = evaluate(key, negative_policy, num_steps)
    _assert_mutated(
        _evaluation_derived_output(positive_output[0]),
        _evaluation_derived_output(negative_output[0]),
    )
    return positive_output, negative_output


def _primitive_names(value):
    names = []
    if hasattr(value, "primitive") and hasattr(value, "params"):
        names.append(value.primitive.name)
        names.extend(_primitive_names(value.params))
    elif hasattr(value, "jaxpr"):
        names.extend(_primitive_names(value.jaxpr))
    elif hasattr(value, "eqns"):
        for equation in value.eqns:
            names.extend(_primitive_names(equation))
    elif isinstance(value, dict):
        for nested in value.values():
            names.extend(_primitive_names(nested))
    elif isinstance(value, (tuple, list)):
        for nested in value:
            names.extend(_primitive_names(nested))
    return names


def _is_numeric_constant(value):
    try:
        return np.asarray(value).dtype != np.dtype("O")
    except (TypeError, ValueError):
        return False


def _assert_jaxpr_host_pure(closed_jaxpr):
    forbidden = (
        "debug_callback",
        "host_callback",
        "io_callback",
        "lox",
        "recorder",
        "config_parser",
        "registry",
        "agentprogram",
        "optimizer transform",
    )
    primitives = {name.lower() for name in _primitive_names(closed_jaxpr)}
    matches = [token for token in forbidden if token in primitives]
    assert not matches, f"forbidden host tokens in JAXPR: {matches}"
    assert all(
        _is_numeric_constant(leaf) for leaf in jax.tree.leaves(closed_jaxpr.consts)
    )


def test_jaxpr_purity_rejects_nested_debug_callback():
    def with_nested_callback(value):
        def callback_branch(operand):
            jax.debug.callback(lambda _: None, operand)
            return operand + 1

        return jax.lax.cond(
            value > 0,
            callback_branch,
            lambda operand: operand - 1,
            value,
        )

    closed_jaxpr = jax.make_jaxpr(with_nested_callback)(jnp.asarray(1))

    with pytest.raises(AssertionError, match="debug_callback"):
        _assert_jaxpr_host_pure(closed_jaxpr)


def test_standard_program_preserves_rtu_subclass_carry_semantics(
    stream_ac_agent_factory,
):
    from memorax.online_ac.standard import make_standard_program

    class OnesCarryRTUCell(RTUCell):
        @nn.nowrap
        def initialize_carry(self, key, input_shape):
            carry = super().initialize_carry(key, input_shape)
            return jax.tree.map(jnp.ones_like, carry)

    parts = stream_ac_agent_factory(adaptive=False)
    actor_network = parts.actor_network.clone(
        torso=parts.actor_network.torso.clone(
            cell=OnesCarryRTUCell(config=parts.actor_network.torso.cell.config)
        )
    )
    program = make_standard_program(
        replace(parts, actor_network=actor_network), parts.cfg
    )

    state = program.init_fn(jax.random.key(0))

    assert all(
        np.all(np.asarray(leaf) == 1) for leaf in jax.tree.leaves(state.actor_carry)
    )


def test_standard_init_uses_no_private_or_manual_jaxpr_interpreter():
    from memorax.online_ac import standard

    source = inspect.getsource(standard)

    assert "jax._src" not in source
    assert "eval_jaxpr" not in source
    assert "_without_lox_effects" not in source


def test_standard_evaluate_depends_on_policy_parameters(
    stream_ac_agent_factory,
):
    from memorax.online_ac.standard import make_standard_program

    parts = stream_ac_agent_factory(adaptive=False)
    program = make_standard_program(parts, parts.cfg)
    state = program.init_fn(jax.random.key(20))
    positive_policy = _force_evaluation_action(state, 1)
    negative_policy = _force_evaluation_action(state, -1)
    evaluate_key = jax.random.key(21)

    positive_output, _ = _assert_evaluation_computation_sensitive(
        program.evaluate_fn,
        evaluate_key,
        positive_policy,
        negative_policy,
        2,
    )

    def ignore_state_mutant(key, state, num_steps):
        del key, state, num_steps
        return positive_output

    with pytest.raises(AssertionError):
        _assert_evaluation_computation_sensitive(
            ignore_state_mutant,
            evaluate_key,
            positive_policy,
            negative_policy,
            2,
        )


@pytest.mark.parametrize("kind", ["meta", "standard"])
def test_jit_lifecycle_has_stable_signatures_and_reuses_fixed_shape_traces(
    kind,
    rtrrl_agent_factory,
    stream_ac_agent_factory,
):
    program = _make_program(kind, rtrrl_agent_factory, stream_ac_agent_factory)

    counted_init, init_trace_count = _count_traces(program.init_fn)
    jitted_init = jax.jit(counted_init)
    initial = _block(jitted_init(jax.random.key(1)))
    second_initial = _block(jitted_init(jax.random.key(2)))
    assert init_trace_count() == 1
    _assert_array_state(initial)
    _assert_array_state(second_initial)
    _assert_mutated(initial, second_initial)

    initial_signature = _signature(initial)
    assert _signature(second_initial) == initial_signature

    counted_train, train_trace_count = _count_traces(program.train_epoch_fn)
    jitted_train = jax.jit(counted_train, static_argnames=("num_steps",))
    train_key = jax.random.key(3)
    trained, metrics = _block(jitted_train(train_key, initial, num_steps=1))
    mutated_initial = initial.replace(
        timestep=initial.timestep.replace(obs=initial.timestep.obs + 0.25)
    )
    trained_again, metrics_again = _block(
        jitted_train(train_key, mutated_initial, num_steps=1)
    )
    assert train_trace_count() == 1
    _assert_array_state(trained)
    _assert_array_state(trained_again)
    assert _signature(trained) == initial_signature
    assert _signature(trained_again) == initial_signature
    assert _signature(metrics_again) == _signature(metrics)
    _assert_mutated(trained, trained_again)

    counted_evaluate, evaluate_trace_count = _count_traces(program.evaluate_fn)
    jitted_evaluate = jax.jit(counted_evaluate, static_argnames=("num_steps",))
    evaluate_key = jax.random.key(5)
    positive_policy = _force_evaluation_action(trained, 1)
    negative_policy = _force_evaluation_action(trained, -1)
    evaluated, summary = _block(
        jitted_evaluate(evaluate_key, positive_policy, num_steps=2)
    )
    evaluated_again, summary_again = _block(
        jitted_evaluate(evaluate_key, negative_policy, num_steps=2)
    )
    assert evaluate_trace_count() == 1
    _assert_array_state(evaluated)
    _assert_array_state(evaluated_again)
    assert isinstance(summary, EvalSummary)
    assert isinstance(summary_again, EvalSummary)
    assert _signature(evaluated) == initial_signature
    assert _signature(evaluated_again) == initial_signature
    assert _signature(summary_again) == _signature(summary)
    _assert_mutated(
        _evaluation_derived_output(evaluated),
        _evaluation_derived_output(evaluated_again),
    )


@pytest.mark.parametrize("kind", ["meta", "standard"])
def test_jaxpr_is_host_pure_and_state_contains_only_array_leaves(
    kind,
    rtrrl_agent_factory,
    stream_ac_agent_factory,
):
    program = _make_program(kind, rtrrl_agent_factory, stream_ac_agent_factory)
    assert isinstance(program, AgentProgram)
    state = program.init_fn(jax.random.key(10))
    assert jax.tree.leaves(state)
    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(state))

    jaxprs = (
        jax.make_jaxpr(program.init_fn)(jax.random.key(11)),
        jax.make_jaxpr(program.train_epoch_fn, static_argnums=(2,))(
            jax.random.key(12), state, 1
        ),
        jax.make_jaxpr(program.evaluate_fn, static_argnums=(2,))(
            jax.random.key(13), state, 2
        ),
    )
    for closed_jaxpr in jaxprs:
        _assert_jaxpr_host_pure(closed_jaxpr)
