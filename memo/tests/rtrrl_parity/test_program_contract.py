"""Closed-program, fixed-shape scan, and JIT contracts for strict RTRRL."""

# pyright: reportMissingImports=false
# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

WORKTREE_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(WORKTREE_ROOT / "rtrrl"))
sys.path.insert(0, str(WORKTREE_ROOT / "memo" / "experiments" / "base"))
sys.path.insert(0, str(WORKTREE_ROOT / "memo" / "tests" / "online_ac"))

from memorax.algorithms.rtrrl import RTRRL, RTRRLState as PublicRTRRLState
from memorax.algorithms.rtrrl.heads import RTRRLTDHead
from memorax.algorithms.rtrrl.lru import AAAI25LRU
from memorax.algorithms.rtrrl import program as program_module
from memorax.algorithms.rtrrl.program import build_rtrrl_program
from memorax.algorithms.rtrrl.types import RTRRLState
from memorax.online_ac.types import (
    ActionDecision,
    AgentProgram,
    EvalSummary,
)

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


def test_public_state_export_is_the_program_state_schema():
    components, config, _ = _strict_setup()
    program = build_rtrrl_program(config, components, _ThreeStepEnvironment())

    assert PublicRTRRLState is RTRRLState
    assert program.state_schema is PublicRTRRLState


def test_strict_experiment_builder_reaches_only_closed_program(monkeypatch):
    import experiment
    from conftest import TinyContinuousEnv

    env = TinyContinuousEnv()
    captured = {}
    sentinel = AgentProgram(
        init_fn=lambda key: None,
        train_epoch_fn=lambda key, state, steps: (state, None),
        evaluate_fn=lambda key, state, steps: (state, EvalSummary()),
        state_schema=RTRRLState,
        metric_schema=program_module.RTRRLEpochSummary,
    )

    def strict_builder(config, components, environment):
        captured.update(
            config=config,
            components=components,
            environment=environment,
        )
        return sentinel

    monkeypatch.setattr(experiment, "build_rtrrl_program", strict_builder)
    monkeypatch.setattr(
        experiment,
        "build_meta_program",
        lambda *args, **kwargs: pytest.fail(
            "strict profile reached build_meta_program"
        ),
        raising=False,
    )
    cfg = SimpleNamespace(
        profile="aaai25_strict_lru",
        num_envs=2,
        hidden_dim=3,
        meta_rl=True,
        use_encoder=False,
    )

    agent = experiment.build_rtrrl_agent(cfg, env, env.default_params)

    assert isinstance(agent, RTRRL)
    assert agent.profile == "aaai25_strict_lru"
    assert agent.program is sentinel
    assert isinstance(captured["components"].recurrent, AAAI25LRU)
    assert isinstance(captured["components"].head, RTRRLTDHead)
    assert captured["config"].observation_dim == 2
    assert captured["config"].action_dim == 2
    assert captured["config"].num_envs == 2


def test_strict_update_step_executes_one_vector_transition_for_multiple_envs():
    calls = []
    program = AgentProgram(
        init_fn=lambda key: jnp.asarray(0),
        train_epoch_fn=lambda key, state, steps: (
            calls.append(steps) or state + steps,
            None,
        ),
        evaluate_fn=lambda key, state, steps: (state, EvalSummary()),
        state_schema=RTRRLState,
        metric_schema=program_module.RTRRLEpochSummary,
    )
    agent = RTRRL.from_program(
        program,
        profile="aaai25_strict_lru",
        num_envs=8,
    )

    next_state, auxiliary = agent._update_step(
        jnp.asarray(4), jax.random.key(3)
    )

    assert int(next_state) == 5
    assert auxiliary is None
    assert calls == [1]


def test_strict_builder_owns_normalization_without_host_callbacks():
    import experiment
    from conftest import TinyContinuousEnv
    from memorax.environments.wrappers import (
        NormalizeObservationWrapper,
        NormalizeRewardWrapper,
        RecordEpisodeStatistics,
    )

    env = NormalizeRewardWrapper(
        NormalizeObservationWrapper(
            RecordEpisodeStatistics(TinyContinuousEnv())
        )
    )
    cfg = SimpleNamespace(
        profile="aaai25_strict_lru",
        num_envs=2,
        hidden_dim=3,
        meta_rl=True,
        use_encoder=False,
        normalize_obs=True,
        normalize_reward=True,
    )
    agent = experiment.build_rtrrl_agent(cfg, env, env.default_params)
    state = agent.init(jax.random.key(5))

    closed = jax.make_jaxpr(
        lambda key, current: agent.program.train_epoch_fn(key, current, 1)
    )(jax.random.key(7), state)

    assert hasattr(state.environment_state, "observation_mean")
    assert hasattr(state.environment_state, "reward_mean")
    assert not _primitive_names(closed) & {
        "debug_callback",
        "io_callback",
        "pure_callback",
        "outside_call",
    }


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
    assert "environment_state" in summary.info
    assert "returned_episode" in summary.info
    assert "returned_episode_returns" in summary.info
