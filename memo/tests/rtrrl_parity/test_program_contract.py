"""Closed-program, fixed-shape scan, and JIT contracts for strict RTRRL."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from memorax.algorithms.rtrrl import program as program_module
from memorax.algorithms.rtrrl.program import build_rtrrl_program
from memorax.algorithms.rtrrl.types import RTRRLState
from memorax.online_ac.types import ActionDecision, AgentProgram, EvalSummary

from .test_init_parity import _strict_setup
from .test_step_parity import _ThreeStepEnvironment


def _primitive_names(closed_jaxpr):
    names = set()

    def visit(jaxpr):
        for equation in jaxpr.eqns:
            names.add(equation.primitive.name)
            for value in equation.params.values():
                if hasattr(value, "jaxpr") and hasattr(value.jaxpr, "eqns"):
                    visit(value.jaxpr)
                elif hasattr(value, "eqns"):
                    visit(value)
                elif isinstance(value, (tuple, list)):
                    for item in value:
                        if hasattr(item, "jaxpr") and hasattr(
                            item.jaxpr, "eqns"
                        ):
                            visit(item.jaxpr)
                        elif hasattr(item, "eqns"):
                            visit(item)

    visit(closed_jaxpr.jaxpr)
    return names


def test_builder_selects_components_once_and_declares_stable_schemas(monkeypatch):
    components, config, _ = _strict_setup()
    environment = _ThreeStepEnvironment()
    calls = {"init": 0, "step": 0}
    original_init = program_module.make_init_fn
    original_step = program_module.make_step_fn

    def counted_init(*args, **kwargs):
        calls["init"] += 1
        return original_init(*args, **kwargs)

    def counted_step(*args, **kwargs):
        calls["step"] += 1
        return original_step(*args, **kwargs)

    monkeypatch.setattr(program_module, "make_init_fn", counted_init)
    monkeypatch.setattr(program_module, "make_step_fn", counted_step)

    program = build_rtrrl_program(config, components, environment)

    assert isinstance(program, AgentProgram)
    assert calls == {"init": 1, "step": 1}
    assert program.state_schema is RTRRLState
    assert program.metric_schema is program_module.RTRRLEpochSummary


def test_production_epoch_returns_only_final_state_and_scalar_summary():
    components, config, _ = _strict_setup()
    program = build_rtrrl_program(config, components, _ThreeStepEnvironment())
    state = program.init_fn(jax.random.key(7))
    state_structure = jax.tree.structure(state)

    final_state, summary = program.train_epoch_fn(
        jax.random.key(11), state, 3
    )

    assert jax.tree.structure(final_state) == state_structure
    assert all(jnp.ndim(leaf) == 0 for leaf in jax.tree.leaves(summary))
    assert len(jax.tree.leaves(summary)) < len(jax.tree.leaves(final_state))
    assert int(summary.steps) == 3


def test_train_epoch_has_no_host_callback_and_fixed_shape_does_not_retrace():
    components, config, _ = _strict_setup()
    program = build_rtrrl_program(config, components, _ThreeStepEnvironment())
    state = program.init_fn(jax.random.key(13))

    closed = jax.make_jaxpr(
        lambda key, current: program.train_epoch_fn(key, current, 3)
    )(jax.random.key(17), state)
    names = _primitive_names(closed)
    assert not names & {
        "debug_callback",
        "io_callback",
        "pure_callback",
        "outside_call",
    }

    traces = 0

    def counted(key, current):
        nonlocal traces
        traces += 1
        return program.train_epoch_fn(key, current, 3)

    compiled = jax.jit(counted)
    first_state, first_summary = compiled(jax.random.key(19), state)
    second_state, second_summary = compiled(jax.random.key(23), first_state)
    jax.block_until_ready((second_state, second_summary))

    assert traces == 1
    assert jax.tree.structure(first_state) == jax.tree.structure(second_state)
    assert jax.tree.structure(first_summary) == jax.tree.structure(
        second_summary
    )


def test_evaluation_uses_action_decision_event_schema():
    components, config, _ = _strict_setup()
    program = build_rtrrl_program(config, components, _ThreeStepEnvironment())
    state = program.init_fn(jax.random.key(29))

    returned_state, summary = program.evaluate_fn(
        jax.random.key(31), state, 2
    )

    assert returned_state is state
    assert isinstance(summary, EvalSummary)
    assert isinstance(summary.info["action_decision"], ActionDecision)
