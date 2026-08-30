"""Parallel environment interaction and the scales its values are read through."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from memorax.utils import Timestep

from .normalization import environment_owns_normalization, make_normalizer


class NormalizationState(struct.PyTreeNode):
    """The state of the observation and reward estimators."""

    observation: Any = None
    reward: Any = None


def broadcast_stream(values, leaf):
    """Broadcast one value per environment across a streamed leaf."""

    return values[(slice(None),) + (None,) * (leaf.ndim - 1)]


def select_ended(done, fresh, carried):
    """Take fresh state for ended streams and carried state for the rest."""

    return jax.tree.map(
        lambda new, old: jnp.where(broadcast_stream(done, old), new, old),
        fresh,
        carried,
    )


def terminal_of(info, done):
    """Read true termination when the environment distinguishes truncation."""

    return info.get("terminal", done)


class EnvironmentStreams:
    """Own the position and transition of every parallel environment stream."""

    def __init__(self, num_envs: int, env: Any, env_params: Any) -> None:
        self.num_envs = num_envs
        self.env = env
        self.env_params = env_params

    def blank_timestep(self, obs) -> Timestep:
        """Construct the observation before the first environment transition."""

        action_space = self.env.action_space(self.env_params)
        return Timestep(
            obs=obs,
            action=jnp.zeros(
                (self.num_envs, *action_space.shape), dtype=action_space.dtype
            ),
            reward=jnp.zeros((self.num_envs,), dtype=jnp.float32),
            done=jnp.zeros((self.num_envs,), dtype=jnp.bool_),
        )

    def init(self, key: Any):
        """Open every environment stream."""

        keys = jax.random.split(key, self.num_envs)
        return jax.vmap(self.env.reset, in_axes=(0, None))(keys, self.env_params)

    def reset(self, key, env_state, done):
        """Open a new episode only for streams whose previous one ended."""

        obs, opened = self.init(key)
        return obs, select_ended(done, opened, env_state)

    def bound(self, action):
        """The action as the environment will actually take it.

        A continuous space states what it accepts and an adapter is entitled to
        hold it to that, but a policy is not obliged to sample inside it: the
        Gaussian readouts are unsquashed, so what they draw regularly leaves the
        box. Bounding here rather than inside an adapter is what keeps the value
        the caller records equal to the value the environment received -- and
        those must agree, because a meta-RL input feeds the recorded action back
        into the network, where an unbounded copy is a loop that nothing between
        the two ever closes. It is offered rather than applied here: a caller
        that feeds the action back asks for it, and one that does not is left
        byte-identical to what it was.

        A space that declares no bounds -- a discrete one -- is returned
        untouched; there is nothing to hold it to.
        """

        space = self.env.action_space(self.env_params)
        low, high = getattr(space, "low", None), getattr(space, "high", None)
        if low is None or high is None:
            return action
        return jnp.clip(action, low, high)

    def step(self, key, env_state, action):
        """Advance every stream by one unnormalized transition."""

        keys = jax.random.split(key, self.num_envs)
        obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(keys, env_state, action, self.env_params)
        return (
            obs,
            env_state,
            jnp.asarray(reward, dtype=jnp.float32),
            done,
            terminal_of(info, done),
            info,
        )

    def persisted(self, timestep: Timestep) -> Timestep:
        """Remove feedback that must not cross an episode boundary."""

        broadcast_dims = tuple(range(timestep.done.ndim, timestep.action.ndim))
        reward = jnp.asarray(timestep.reward, dtype=jnp.float32)
        return timestep.replace(
            action=jnp.where(
                jnp.expand_dims(timestep.done, axis=broadcast_dims),
                jnp.zeros_like(timestep.action),
                timestep.action,
            ),
            reward=jnp.where(timestep.done, jnp.zeros_like(reward), reward),
        )


class InteractionNormalization:
    """Own how observations and rewards are scaled across interaction."""

    def __init__(
        self,
        num_envs: int,
        env: Any,
        *,
        observation: Any = None,
        reward: Any = None,
        reset_on_start: bool = True,
        update_during_eval: bool = True,
    ) -> None:
        self.num_envs = num_envs

        def estimator(declared):
            if declared is None:
                return None
            return make_normalizer(
                replace(
                    declared,
                    reset_on_start=reset_on_start,
                    update_during_eval=update_during_eval,
                )
            )

        self.observation = estimator(observation)
        self.reward = estimator(reward)
        self.normalizing = bool(self.observation or self.reward)
        self.resets_on_start = reset_on_start
        self.updates_during_eval = update_during_eval
        if self.normalizing and environment_owns_normalization(env):
            raise ValueError(
                "normalization owner conflict: wrapper and program normalization "
                "are both enabled"
            )

    def init(
        self,
        obs,
        scales: NormalizationState | None = None,
        *,
        update: bool = True,
    ):
        """Begin estimators from nothing or from state another run produced."""

        observation = None if scales is None else scales.observation
        reward = None if scales is None else scales.reward
        if self.observation is not None:
            obs, observation = self.observation.begin(obs, observation, update=update)
        if self.reward is not None and reward is None:
            reward = self.reward.initial(jnp.zeros((self.num_envs,), dtype=jnp.float32))
        return obs, NormalizationState(observation=observation, reward=reward)

    def reset(
        self,
        obs,
        scales: NormalizationState,
        done,
        *,
        update: bool = True,
    ):
        """Read new episode observations and reset only ended stream state."""

        observation = scales.observation
        if self.observation is not None:
            obs, observation = self.observation.begin(obs, observation, update=update)
        return obs, scales.replace(
            observation=select_ended(done, observation, scales.observation)
        )

    def apply(
        self,
        scales: NormalizationState,
        obs,
        reward,
        done,
        *,
        update: bool = True,
    ):
        """Apply and advance both estimators for one transition."""

        observation = scales.observation
        counted = scales.reward
        if self.observation is not None:
            obs, observation = self.observation.observe(
                observation, obs, done=done, update=update
            )
        if self.reward is not None:
            reward, counted = self.reward.observe(
                counted, reward, done=done, update=update
            )
        return obs, reward, NormalizationState(observation=observation, reward=counted)
