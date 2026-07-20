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
    truncate: bool = True

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
        ended = step_count == params.horizon
        terminated = jnp.logical_and(ended, not self.truncate)
        truncated = jnp.logical_and(ended, self.truncate)
        return (
            observation,
            EndingState(step_count, observation),
            step_count.astype(jnp.float32),
            terminated,
            truncated,
            {"step_count": step_count},
        )

    def action_space(self, params):
        del params
        return spaces.Box(-1.0, 1.0, (2,), dtype=jnp.float32)

    def observation_space(self, params):
        del params
        return spaces.Box(-100.0, 100.0, (2,), dtype=jnp.float32)


@struct.dataclass
class MultiEndingState:
    step_count: jax.Array
    horizon: jax.Array
    observation: jax.Array


@dataclass(frozen=True)
class MultiEndingEnv:
    max_horizon: int = 4

    @property
    def default_params(self):
        return EndingParams(horizon=self.max_horizon)

    def reset(self, key, params):
        horizon = jax.random.randint(key, (), 1, params.horizon + 1)
        observation = jnp.array([0.0, 0.0], dtype=jnp.float32)
        return observation, MultiEndingState(jnp.asarray(0), horizon, observation)

    def step(self, key, state, action, params):
        observation, next_state, reward, terminated, truncated, info = self.trace_step(
            key, state, action, params
        )
        return observation, next_state, reward, terminated | truncated, info

    def trace_step(self, key, state, action, params):
        del key, params
        step_count = state.step_count + 1
        observation = state.observation + jnp.asarray(action, dtype=jnp.float32)
        terminated = step_count == state.horizon
        return (
            observation,
            MultiEndingState(step_count, state.horizon, observation),
            step_count.astype(jnp.float32),
            terminated,
            jnp.asarray(False),
            {"step_count": step_count},
        )

    def action_space(self, params):
        del params
        return spaces.Box(-1.0, 1.0, (2,), dtype=jnp.float32)

    def observation_space(self, params):
        del params
        return spaces.Box(-100.0, 100.0, (2,), dtype=jnp.float32)


@dataclass(frozen=True)
class StochasticEnv(TruncatingEnv):
    def trace_step(self, key, state, action, params):
        step_count = state.step_count + 1
        noise = jax.random.uniform(key, state.observation.shape, minval=-1.0, maxval=1.0)
        observation = state.observation + jnp.asarray(action, dtype=jnp.float32) + noise
        reward = jnp.sum(noise)
        ended = step_count == params.horizon
        terminated = jnp.logical_and(ended, not self.truncate)
        truncated = jnp.logical_and(ended, self.truncate)
        return (
            observation,
            EndingState(step_count, observation),
            reward,
            terminated,
            truncated,
            {"step_count": step_count},
        )


def _program(
    kind,
    rtrrl_agent_factory,
    stream_ac_agent_factory,
    env=None,
    *,
    num_envs=1,
):
    if kind == "meta":
        if num_envs != 1:
            raise ValueError("meta test factory only supports one environment")
        parts = rtrrl_agent_factory(fresh_trace=False)
        if env is not None:
            parts = replace(parts, env=env, env_params=env.default_params)
        return make_meta_program(parts, parts.cfg)
    parts = stream_ac_agent_factory(
        adaptive=False,
        continuous=True,
        env=env,
        num_envs=num_envs,
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


@pytest.mark.parametrize("kind", ["meta", "standard"])
def test_stochastic_trace_uses_the_actual_per_environment_step_key(
    kind, rtrrl_agent_factory, stream_ac_agent_factory
):
    env = StochasticEnv(truncate=False)
    program = _program(kind, rtrrl_agent_factory, stream_ac_agent_factory, env=env)
    state = program.init_fn(jax.random.key(20))
    evaluate_key = jax.random.key(21)

    evaluated, summary = program.evaluate_fn(evaluate_key, state, num_steps=1)

    reset_key, eval_key = jax.random.split(evaluate_key)
    reset_env_key = jax.random.split(reset_key, 1)[0]
    _, initial_env_state = env.reset(reset_env_key, env.default_params)
    scan_key = jax.random.split(eval_key, 1)[0]
    _, env_key = jax.random.split(scan_key)
    actual_step_key = jax.random.split(env_key, 1)[0]
    action = summary.trace.actions[0, 0]
    expected = env.trace_step(
        actual_step_key, initial_env_state, action, env.default_params
    )
    fixed_key_result = env.trace_step(
        jax.random.key(0), initial_env_state, action, env.default_params
    )

    np.testing.assert_array_equal(summary.trace.observations[1, 0], expected[0])
    np.testing.assert_array_equal(
        summary.trace.environment_states.observation[0, 0],
        expected[1].observation,
    )
    np.testing.assert_array_equal(summary.trace.rewards[0, 0], expected[2])
    np.testing.assert_array_equal(summary.trace.terminals[0, 0], expected[3])
    np.testing.assert_array_equal(summary.trace.truncations[0, 0], expected[4])
    np.testing.assert_array_equal(evaluated.timestep.obs[0], expected[0])
    np.testing.assert_array_equal(evaluated.env_state.observation[0], expected[1].observation)
    assert not np.array_equal(
        np.asarray(summary.trace.observations[1, 0]),
        np.asarray(fixed_key_result[0]),
    )


def test_multi_environment_valid_transitions_and_trace_axes(
    rtrrl_agent_factory, stream_ac_agent_factory
):
    del rtrrl_agent_factory
    num_envs = 8
    num_steps = 4
    program = _program(
        "standard",
        None,
        stream_ac_agent_factory,
        env=MultiEndingEnv(max_horizon=num_steps),
        num_envs=num_envs,
    )
    state = program.init_fn(jax.random.key(22))

    _, summary = program.evaluate_fn(jax.random.key(23), state, num_steps=num_steps)

    trace = summary.trace
    horizons = np.asarray(trace.environment_states.horizon[0])
    np.testing.assert_array_equal(trace.valid_transitions, horizons)
    assert np.unique(horizons).size > 1
    assert trace.observations.shape == (num_steps + 1, num_envs, 2)
    assert trace.actions.shape == (num_steps, num_envs, 2)
    assert trace.rewards.shape == (num_steps, num_envs)
    assert trace.terminals.shape == (num_steps, num_envs)
    assert trace.truncations.shape == (num_steps, num_envs)
    for env_index, horizon in enumerate(horizons):
        assert bool(trace.terminals[horizon - 1, env_index])


def test_valid_transitions_are_zero_when_window_has_no_complete_episode(
    rtrrl_agent_factory, stream_ac_agent_factory
):
    del rtrrl_agent_factory
    program = _program(
        "standard",
        None,
        stream_ac_agent_factory,
        env=TruncatingEnv(),
        num_envs=3,
    )
    state = program.init_fn(jax.random.key(24))

    _, summary = program.evaluate_fn(jax.random.key(25), state, num_steps=2)

    np.testing.assert_array_equal(
        summary.trace.valid_transitions,
        np.zeros((3,), dtype=np.int32),
    )


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


@pytest.mark.parametrize(
    ("truncate", "expected_terminated", "expected_truncated"),
    [(False, True, False), (True, False, True)],
    ids=["terminated", "truncated"],
)
def test_episode_statistics_ending_step_records_and_resets(
    truncate, expected_terminated, expected_truncated
):
    env = RecordEpisodeStatistics(TruncatingEnv(truncate=truncate))
    _, state = env.reset(jax.random.key(10), env.default_params)

    for step in range(3):
        observation, state, _, terminated, truncated, info = env.trace_step(
            jax.random.fold_in(jax.random.key(11), step),
            state,
            jnp.zeros((2,), dtype=jnp.float32),
            env.default_params,
        )

    np.testing.assert_array_equal(observation, state.env_state.observation)
    assert bool(terminated) is expected_terminated
    assert bool(truncated) is expected_truncated
    assert bool(info["returned_episode"])
    assert int(info["returned_episode_lengths"]) == 3
    assert float(info["returned_episode_returns"]) == 6.0
    assert float(info["returned_discounted_episode_returns"]) == pytest.approx(
        1.0 + 0.99 * 2.0 + 0.99**2 * 3.0
    )
    assert int(state.episode_lengths) == 0
    assert float(state.episode_returns) == 0.0
    assert float(state.discounted_episode_returns) == 0.0
    assert float(state.episode_discount) == 1.0


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


@pytest.mark.parametrize(
    ("natural_done", "episode_length", "expected_terminated", "expected_truncated"),
    [(True, 5, True, False), (False, 1, False, True)],
    ids=["termination", "truncation"],
)
def test_real_brax_episode_wrapper_preserves_endpoint_and_ending_semantics(
    natural_done,
    episode_length,
    expected_terminated,
    expected_truncated,
):
    from brax.envs.base import State
    from brax.envs.wrappers.training import EpisodeWrapper

    class OneStepBraxEnv:
        action_size = 1
        observation_size = 1

        def reset(self, key):
            del key
            return State(
                pipeline_state=None,
                obs=jnp.zeros((1,)),
                reward=jnp.asarray(0.0),
                done=jnp.asarray(False),
                metrics={},
                info={},
            )

        def step(self, state, action):
            del action
            return state.replace(
                obs=jnp.asarray([7.0]),
                reward=jnp.asarray(3.0),
                done=jnp.asarray(natural_done),
            )

    episode_env = EpisodeWrapper(
        OneStepBraxEnv(), episode_length=episode_length, action_repeat=1
    )
    wrapper = BraxGymnaxWrapper(episode_env, trace_env=episode_env)
    state = episode_env.reset(jax.random.key(13))

    observation, next_state, reward, terminated, truncated, _ = wrapper.trace_step(
        jax.random.key(14), state, jnp.zeros((1,)), wrapper.default_params
    )

    np.testing.assert_array_equal(observation, jnp.asarray([7.0]))
    np.testing.assert_array_equal(next_state.obs, observation)
    assert float(reward) == 3.0
    assert bool(terminated) is expected_terminated
    assert bool(truncated) is expected_truncated
