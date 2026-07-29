"""Explicit program-owned observation and reward normalization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax.numpy as jnp
from flax import struct

from memorax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
)

# Whose running statistics to keep. ``ours`` is this file; ``upstream``
# reproduces the three things streaming-drl's wrappers do differently, and is an
# arm to compare against rather than a setting to train with.
# ``normalization_upstream.py`` says what the three are.
STATISTICS = ("ours", "upstream")


@dataclass(frozen=True)
class NormalizationConfig:
    normalize_observation: bool = False
    normalize_reward: bool = False
    eps: float = 1e-8
    reward_gamma: float = 0.99
    reset_on_start: bool = True
    update_during_eval: bool = True
    statistics: str = "ours"


@struct.dataclass
class RunningStatistics:
    mean: Any
    M2: Any
    count: Any


@struct.dataclass
class RewardStatistics:
    mean: Any
    M2: Any
    count: Any
    G: Any


@struct.dataclass
class NormalizerState:
    observation: RunningStatistics | None
    reward: RewardStatistics | None
    episode_return: Any


@struct.dataclass
class NormalizedStep:
    observation: Any
    reward: Any
    state: NormalizerState
    raw_episode_return: Any


@struct.dataclass
class NormalizationMetrics:
    observation_mean: Any = None
    observation_std: Any = None
    reward_mean: Any = None
    reward_std: Any = None


def _expand_for(value, target):
    return value[(slice(None),) + (None,) * (target.ndim - value.ndim)]


class Normalizer:
    def __init__(self, config: NormalizationConfig):
        self.config = config

    def _initial_state(self, observation) -> NormalizerState:
        """Where the statistics start, before any observation is folded in.

        A method so that a normaliser reproducing another implementation can
        start where that one starts. Ours seeds a pseudo-observation of mean
        zero and second moment one, which is not what upstream does and is the
        first of the three differences ``normalization_upstream.py`` sets out.
        """

        num_envs = observation.shape[0]
        ones = jnp.ones((num_envs,), dtype=jnp.float32)
        return NormalizerState(
            observation=(
                RunningStatistics(
                    mean=jnp.zeros_like(observation),
                    M2=jnp.ones_like(observation),
                    count=ones,
                )
                if self.config.normalize_observation
                else None
            ),
            reward=(
                RewardStatistics(
                    mean=jnp.zeros((num_envs,), dtype=jnp.float32),
                    M2=ones,
                    count=ones,
                    G=jnp.zeros((num_envs,), dtype=jnp.float32),
                )
                if self.config.normalize_reward
                else None
            ),
            episode_return=jnp.zeros((num_envs,), dtype=jnp.float32),
        )

    def reset(self, observation, state=None, *, update=True):
        """Normalize reset observations, cold-starting statistics when requested."""
        observation = jnp.asarray(observation)
        if state is None:
            state = self._initial_state(observation)

        observation_stats = state.observation
        normalized = observation
        if observation_stats is not None:
            if update:
                observation_stats = self._update_observation(
                    observation_stats, observation
                )
            normalized = self._normalize_observation(observation_stats, observation)
        return normalized, replace(
            state,
            observation=observation_stats,
            episode_return=jnp.zeros_like(state.episode_return),
        )

    def step(self, state, *, observation, reward, done, update=True):
        observation = jnp.asarray(observation)
        reward = jnp.asarray(reward, dtype=jnp.float32)
        done = jnp.asarray(done)

        observation_stats = state.observation
        normalized_observation = observation
        if observation_stats is not None:
            if update:
                observation_stats = self._update_observation(
                    observation_stats, observation
                )
            normalized_observation = self._normalize_observation(
                observation_stats, observation
            )

        reward_stats = state.reward
        normalized_reward = reward
        if reward_stats is not None:
            if update:
                reward_stats = self._update_reward(reward_stats, reward, done)
            normalized_reward = self._scale_reward(reward_stats, reward)

        accumulated_return = state.episode_return + reward
        next_episode_return = jnp.where(done, 0.0, accumulated_return)
        return NormalizedStep(
            observation=normalized_observation,
            reward=normalized_reward,
            state=replace(
                state,
                observation=observation_stats,
                reward=reward_stats,
                episode_return=next_episode_return,
            ),
            raw_episode_return=accumulated_return,
        )

    def _update_observation(self, state, observation):
        count = state.count + 1
        expanded_count = _expand_for(count, observation)
        delta = observation - state.mean
        mean = state.mean + delta / expanded_count
        M2 = state.M2 + delta * (observation - mean)
        return replace(state, mean=mean, M2=M2, count=count)

    def _normalize_observation(self, state, observation):
        count = _expand_for(state.count, observation)
        return (observation - state.mean) / jnp.sqrt(state.M2 / count + self.config.eps)

    def _scale_reward(self, state, reward):
        """The spread the reward is divided by. Ours is ``M2 / count``."""

        return reward / jnp.sqrt(state.M2 / state.count + self.config.eps)

    def _update_reward(self, state, reward, done):
        G = reward + self.config.reward_gamma * state.G * (1 - done)
        count = state.count + 1
        delta = G - state.mean
        mean = state.mean + delta / count
        M2 = state.M2 + delta * (G - mean)
        return replace(
            state,
            mean=mean,
            M2=M2,
            count=count,
            G=G * (1 - done),
        )


def normalization_metrics(state, eps):
    """Return the legacy per-environment normalization log aggregates."""
    observation_mean = None
    observation_std = None
    reward_mean = None
    reward_std = None
    if state is not None and state.observation is not None:
        observation = state.observation
        feature_axes = tuple(range(1, observation.mean.ndim))
        observation_mean = observation.mean.mean(axis=feature_axes)
        observation_std = jnp.sqrt(
            observation.M2 / _expand_for(observation.count, observation.M2) + eps
        ).mean(axis=feature_axes)
    if state is not None and state.reward is not None:
        reward = state.reward
        reward_mean = reward.mean
        reward_std = jnp.sqrt(reward.M2 / reward.count + eps)
    return NormalizationMetrics(
        observation_mean=observation_mean,
        observation_std=observation_std,
        reward_mean=reward_mean,
        reward_std=reward_std,
    )


def make_normalizer(config) -> Normalizer:
    """Build an explicit normalizer from a config or config-like object."""
    if not isinstance(config, NormalizationConfig):
        config = NormalizationConfig(
            normalize_observation=bool(
                getattr(
                    config,
                    "normalize_observation",
                    getattr(config, "normalize_observations", False),
                )
            ),
            normalize_reward=bool(
                getattr(
                    config,
                    "normalize_reward",
                    getattr(config, "normalize_rewards", False),
                )
            ),
            eps=float(getattr(config, "normalization_eps", 1e-8)),
            reward_gamma=float(getattr(config, "normalization_reward_gamma", 0.99)),
            reset_on_start=bool(getattr(config, "reset_on_start", True)),
            update_during_eval=bool(getattr(config, "update_during_eval", True)),
            statistics=str(getattr(config, "normalization_statistics", "ours")),
        )
    if config.reset_on_start and not config.update_during_eval:
        raise ValueError("reset_on_start=True requires update_during_eval=True")
    if config.statistics not in STATISTICS:
        raise ValueError(
            f"unknown statistics {config.statistics!r}; use {', '.join(STATISTICS)}"
        )
    if config.statistics == "upstream":
        # Imported here because that module subclasses this one's normaliser, and
        # naming it at the top would close the circle.
        from .normalization_upstream import UpstreamNormalizer

        return UpstreamNormalizer(config)
    return Normalizer(config)


def environment_owns_normalization(env) -> bool:
    """Detect the existing normalization wrappers through wrapper chains."""
    current = env
    while current is not None:
        if isinstance(
            current,
            (NormalizeObservationWrapper, NormalizeRewardWrapper),
        ):
            return True
        current = getattr(current, "_env", None)
    return False
