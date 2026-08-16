"""Deterministic environments small enough for algorithm contract tests."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces


class TinyEnvState(struct.PyTreeNode):
    step_count: jnp.ndarray
    observation: jnp.ndarray


class TinyEnvParams(struct.PyTreeNode):
    horizon: int = struct.field(pytree_node=False, default=3)


@dataclass(frozen=True)
class TinyContinuousEnv:
    """A two-dimensional environment that terminates during a short test."""

    @property
    def default_params(self):
        return TinyEnvParams()

    def reset(self, key, params):
        del key, params
        observation = jnp.array([0.25, -0.5], dtype=jnp.float32)
        return observation, TinyEnvState(jnp.asarray(0, jnp.int32), observation)

    def step(self, key, state, action, params):
        del key
        action = jnp.asarray(action, jnp.float32)
        observation = state.observation + action + jnp.array([0.15, 0.45], jnp.float32)
        step_count = state.step_count + 1
        reward = jnp.asarray(0.4, jnp.float32) + 0.35 * step_count
        done = step_count >= params.horizon
        return (
            observation,
            TinyEnvState(step_count, observation),
            reward,
            done,
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
    """The fixed environment used by the recorded StreamAC snapshot."""

    @property
    def default_params(self):
        return TinyEnvParams()

    def reset(self, key, params):
        del key, params
        observation = jnp.array([-0.25, 0.5], dtype=jnp.float32)
        return observation, TinyEnvState(jnp.asarray(0, jnp.int32), observation)

    def step(self, key, state, action, params):
        del key
        direction = jnp.where(action == 0, -1.0, 1.0)
        delta = jnp.array([0.2 * direction, 0.1 + 0.05 * direction], jnp.float32)
        observation = state.observation + delta
        step_count = state.step_count + 1
        reward = jnp.asarray(0.25 * direction + 0.1 * step_count, jnp.float32)
        done = step_count >= params.horizon
        return (
            observation,
            TinyEnvState(step_count, observation),
            reward,
            done,
            {"step_count": step_count},
        )

    def action_space(self, params):
        del params
        return spaces.Discrete(2)

    def observation_space(self, params):
        del params
        return spaces.Box(-10.0, 10.0, (2,), dtype=jnp.float32)
