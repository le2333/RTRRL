from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct
from gymnax.environments import spaces

from memorax.environments.brax import BraxGymnaxWrapper
from memorax.environments.kmemory_chain import KMemoryChain
from memorax.environments.memory_chain import MemoryChain
from memorax.environments.wrappers import MaskObservationWrapper, RecordEpisodeStatistics
from memorax.online_ac.meta import make_meta_program
from memorax.online_ac.standard import make_standard_program
from memorax.online_ac.types import EvalTrace


@struct.dataclass
class EndingState:
    step_count: jax.Array
    observation: jax.Array


@dataclass(frozen=True)
class EndingParams:
    horizon: int = 3


@dataclass(frozen=True)
class TruncatingEnv:
    @property
    def default_params(self):
        return EndingParams()

    def reset(self, key, params):
        del key, params
        observation = jnp.array([0.0, 1.0], dtype=jnp.float32)
        return observation, EndingState(jnp.asarray(0), observation)

    def step(self, key, state, action, params):
        observation, next_state, reward, terminated, truncated, info = self.trace_step(
            key, state, action, params
        )
        return observation, next_state, reward, terminated | truncated, info

    def trace_step(self, key, state, action, params):
        del key
        step_count = state.step_count + 1
        observation = state.observation + jnp.asarray(action, dtype=jnp.float32)
        observation = observation + jnp.asarray([1.0, -1.0], dtype=jnp.float32)
        truncated = step_count == params.horizon
        return (
            observation,
            EndingState(step_count, observation),
            step_count.astype(jnp.float32),
            jnp.asarray(False),
            truncated,
            {"step_count": step_count},
        )

    def action_space(self, params):
        del params
        return spaces.Box(-1.0, 1.0, (2,), dtype=jnp.float32)

    def observation_space(self, params):
        del params
        return spaces.Box(-100.0, 100.0, (2,), dtype=jnp.float32)


def _program(kind, rtrrl_agent_factory, stream_ac_agent_factory, env=None):
    if kind == "meta":
        parts = rtrrl_agent_factory(fresh_trace=False)
        if env is not None:
            parts = replace(parts, env=env, env_params=env.default_params)
        return make_meta_program(parts, parts.cfg)
    parts = stream_ac_agent_factory(
        adaptive=False,
        continuous=True,
        env=env,
    )
    return make_standard_program(parts, parts.cfg)


@pytest.mark.parametrize("kind", ["meta", "standard"])
def test_eval_trace_has_n_plus_one_observations_and_n_transitions(
    kind, rtrrl_agent_factory, stream_ac_agent_factory
):
    program = _program(kind, rtrrl_agent_factory, stream_ac_agent_factory)
    state = program.init_fn(jax.random.key(1))

    _, summary = program.evaluate_fn(jax.random.key(2), state, num_steps=5)

    assert isinstance(summary.trace, EvalTrace)
    trace = summary.trace
    assert trace.observations.shape[0] == trace.actions.shape[0] + 1 == 6
    assert trace.rewards.shape[0] == trace.actions.shape[0] == 5
    assert trace.terminals.shape == trace.truncations.shape == trace.rewards.shape
    assert trace.valid_transitions.shape == (1,)
    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(trace))


@pytest.mark.parametrize("kind", ["meta", "standard"])
def test_natural_terminal_keeps_real_post_transition_endpoint(
    kind, rtrrl_agent_factory, stream_ac_agent_factory
):
    program = _program(kind, rtrrl_agent_factory, stream_ac_agent_factory)
    state = program.init_fn(jax.random.key(3))

    _, summary = program.evaluate_fn(jax.random.key(4), state, num_steps=5)

    trace = summary.trace
    assert int(trace.valid_transitions[0]) == 3
    assert bool(trace.terminals[2, 0])
    assert not bool(trace.truncations[2, 0])
    np.testing.assert_array_equal(
        trace.observations[3, 0],
        trace.environment_states.observation[2, 0],
    )


@pytest.mark.parametrize("kind", ["meta", "standard"])
def test_first_truncation_sets_valid_transition_count(
    kind, rtrrl_agent_factory, stream_ac_agent_factory
):
    program = _program(
        kind,
        rtrrl_agent_factory,
        stream_ac_agent_factory,
        env=TruncatingEnv(),
    )
    state = program.init_fn(jax.random.key(5))

    _, summary = program.evaluate_fn(jax.random.key(6), state, num_steps=5)

    trace = summary.trace
    assert int(trace.valid_transitions[0]) == 3
    assert not bool(trace.terminals[2, 0])
    assert bool(trace.truncations[2, 0])
    assert np.all(~np.asarray(trace.terminals[:3, 0]))


@pytest.mark.parametrize(
    "env",
    [MemoryChain(L=2), KMemoryChain(L=2, K=2)],
    ids=["memory-chain", "kmemory-chain"],
)
def test_memory_chains_report_natural_termination_without_truncation(env):
    observation, state = env.reset(jax.random.key(7), env.default_params)
    del observation

    *_, first_terminated, first_truncated, _ = env.trace_step(
        jax.random.key(8), state, jnp.asarray(0), env.default_params
    )
    _, state, _, _, _, _ = env.trace_step(
        jax.random.key(8), state, jnp.asarray(0), env.default_params
    )
    *_, terminated, truncated, _ = env.trace_step(
        jax.random.key(9), state, jnp.asarray(0), env.default_params
    )

    assert not bool(first_terminated)
    assert not bool(first_truncated)
    assert bool(terminated)
    assert not bool(truncated)


def test_episode_statistics_trace_step_preserves_terminal_observation():
    env = RecordEpisodeStatistics(TruncatingEnv())
    _, state = env.reset(jax.random.key(10), env.default_params)

    observation, state, _, terminated, truncated, info = env.trace_step(
        jax.random.key(11),
        state,
        jnp.zeros((2,), dtype=jnp.float32),
        env.default_params,
    )

    np.testing.assert_array_equal(observation, state.env_state.observation)
    assert not bool(terminated)
    assert not bool(truncated)
    assert set(info) >= {"returned_episode", "returned_episode_lengths"}


def test_observation_mask_is_applied_to_trace_endpoint():
    env = RecordEpisodeStatistics(
        MaskObservationWrapper(
            TruncatingEnv(), mask=jnp.asarray([1.0, 0.0], dtype=jnp.float32)
        )
    )
    _, state = env.reset(jax.random.key(10), env.default_params)

    observation, *_ = env.trace_step(
        jax.random.key(11),
        state,
        jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        env.default_params,
    )

    np.testing.assert_array_equal(observation, jnp.asarray([1.0, 0.0]))


def test_brax_trace_step_hard_fails_without_truncation_signal():
    @struct.dataclass
    class FakeState:
        obs: jax.Array
        reward: jax.Array
        done: jax.Array
        info: dict

    class FakeBraxEnv:
        action_size = 1
        observation_size = 1

        def step(self, state, action):
            del state, action
            return FakeState(
                obs=jnp.ones((1,)),
                reward=jnp.asarray(0.0),
                done=jnp.asarray(True),
                info={},
            )

    wrapper = BraxGymnaxWrapper(FakeBraxEnv())
    state = FakeState(
        obs=jnp.zeros((1,)),
        reward=jnp.asarray(0.0),
        done=jnp.asarray(False),
        info={},
    )

    with pytest.raises(ValueError, match="truncation"):
        wrapper.trace_step(
            jax.random.key(12), state, jnp.zeros((1,)), wrapper.default_params
        )
