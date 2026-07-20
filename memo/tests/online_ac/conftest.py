from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax.numpy as jnp
import pytest
from flax import struct
from gymnax.environments import spaces

from memorax.algorithms import RTRRL, RTRRLConfig, StreamACConfig, StreamACRtrl
from memorax.networks import (
    RNN,
    FeatureExtractor,
    LRUCell,
    LRUConfig,
    Memoroid,
    Network,
    RTUCell,
    RTUConfig,
    heads,
)


@struct.dataclass
class TinyEnvState:
    step_count: jnp.ndarray
    observation: jnp.ndarray


@struct.dataclass
class TinyEnvParams:
    horizon: int = struct.field(pytree_node=False, default=3)


@dataclass(frozen=True)
class TinyContinuousEnv:
    @property
    def default_params(self):
        return TinyEnvParams()

    def reset(self, key, params):
        del key, params
        obs = jnp.array([0.25, -0.5], dtype=jnp.float32)
        return obs, TinyEnvState(jnp.asarray(0, jnp.int32), obs)

    def step(self, key, state, action, params):
        obs, next_state, reward, terminated, _, info = self.trace_step(
            key, state, action, params
        )
        return obs, next_state, reward, terminated, info

    def trace_step(self, key, state, action, params):
        del key
        action = jnp.asarray(action, jnp.float32)
        obs = state.observation + action + jnp.array([0.15, 0.45], jnp.float32)
        step_count = state.step_count + 1
        reward = jnp.asarray(0.4, jnp.float32) + 0.35 * step_count
        done = step_count >= params.horizon
        return (
            obs,
            TinyEnvState(step_count, obs),
            reward,
            done,
            jnp.asarray(False),
            {"step_count": step_count},
        )

    def action_space(self, params):
        del params
        return spaces.Box(-2.0, 2.0, (2,), dtype=jnp.float32)

    def observation_space(self, params):
        del params
        return spaces.Box(-10.0, 10.0, (2,), dtype=jnp.float32)


@dataclass(frozen=True)
class TinyDiscreteEnv:
    @property
    def default_params(self):
        return TinyEnvParams()

    def reset(self, key, params):
        del key, params
        obs = jnp.array([-0.25, 0.5], dtype=jnp.float32)
        return obs, TinyEnvState(jnp.asarray(0, jnp.int32), obs)

    def step(self, key, state, action, params):
        obs, next_state, reward, terminated, _, info = self.trace_step(
            key, state, action, params
        )
        return obs, next_state, reward, terminated, info

    def trace_step(self, key, state, action, params):
        del key
        direction = jnp.where(action == 0, -1.0, 1.0)
        delta = jnp.array([0.2 * direction, 0.1 + 0.05 * direction], jnp.float32)
        obs = state.observation + delta
        step_count = state.step_count + 1
        reward = jnp.asarray(0.25 * direction + 0.1 * step_count, jnp.float32)
        done = step_count >= params.horizon
        return (
            obs,
            TinyEnvState(step_count, obs),
            reward,
            done,
            jnp.asarray(False),
            {"step_count": step_count},
        )

    def action_space(self, params):
        del params
        return spaces.Discrete(2)

    def observation_space(self, params):
        del params
        return spaces.Box(-10.0, 10.0, (2,), dtype=jnp.float32)


@pytest.fixture
def tiny_continuous_env():
    env = TinyContinuousEnv()
    return env, env.default_params


@pytest.fixture
def tiny_discrete_env():
    env = TinyDiscreteEnv()
    return env, env.default_params


def build_rtrrl_agent(*, fresh_trace):
    env = TinyContinuousEnv()
    feature_extractor = FeatureExtractor(
        observation_extractor=nn.Sequential((nn.Dense(3), nn.tanh)),
        action_extractor=nn.Sequential((nn.Dense(3), nn.tanh)),
        reward_extractor=nn.Sequential((nn.Dense(3), nn.tanh)),
    )
    torso = Memoroid(
        cell=LRUCell(config=LRUConfig(features=9, hidden_dim=2, output_dim=3))
    )
    cfg = RTRRLConfig(
        num_envs=1,
        gamma=0.91,
        lambda_pi=0.73,
        lambda_v=0.67,
        lambda_rnn=0.61,
        td_lr=2e-4,
        rnn_lr=3e-5,
        eta_pi=0.4,
        eta_f=0.6,
        entropy_rate=1e-4,
        update_period=0.2,
        update_trace_before_td=fresh_trace,
    )
    return RTRRL(
        cfg,
        env,  # pyright: ignore[reportArgumentType]
        env.default_params,  # pyright: ignore[reportArgumentType]
        feature_extractor,
        torso,
        heads.Gaussian(action_dim=2),
        heads.VNetwork(),
    )


def build_stream_ac_agent(
    *,
    adaptive,
    continuous=False,
    env: Any = None,
    num_envs=1,
):
    if env is None:
        env = TinyContinuousEnv() if continuous else TinyDiscreteEnv()

    def network(head):
        return Network(
            feature_extractor=FeatureExtractor(
                observation_extractor=nn.Sequential((nn.Dense(3), nn.tanh))
            ),
            torso=RNN(cell=RTUCell(config=RTUConfig(features=3, hidden_dim=2))),
            head=head,
        )

    cfg = StreamACConfig(
        num_envs=num_envs,
        gamma=0.89,
        trace_lambda=0.71,
        actor_lr=0.15,
        critic_lr=0.12,
        actor_kappa=3.0,
        critic_kappa=2.0,
        entropy_coefficient=0.02,
        adaptive=adaptive,
        beta2=0.95,
        eps=1e-6,
    )
    return StreamACRtrl(
        cfg,
        env,
        env.default_params,
        network(
            heads.Gaussian(action_dim=2)
            if continuous
            else heads.Categorical(action_dim=2)
        ),
        network(heads.VNetwork()),
    )


@pytest.fixture
def rtrrl_agent_factory():
    return build_rtrrl_agent


@pytest.fixture
def stream_ac_agent_factory():
    return build_stream_ac_agent
