"""Shared interaction components own streams and their numerical scales."""

from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp

from memorax.rl.interaction import (
    EnvironmentStreams,
    InteractionNormalization,
    NormalizationState,
)
from memorax.rl.normalization import NormalizationConfig
from memorax.utils import Timestep
from tests.support.environments import TinyContinuousEnv


def test_environment_streams_reset_only_ended_streams() -> None:
    env = TinyContinuousEnv()
    streams = EnvironmentStreams(2, env, env.default_params)
    _, state = streams.init(jax.random.key(0))
    state = state.replace(step_count=jnp.asarray([4, 7]))

    _, reset = streams.reset(jax.random.key(1), state, jnp.asarray([True, False]))

    assert reset.step_count.tolist() == [0, 7]


def test_persisted_interaction_clears_only_ended_feedback() -> None:
    env = TinyContinuousEnv()
    streams = EnvironmentStreams(2, env, env.default_params)
    timestep = Timestep(
        obs=jnp.zeros((2, 2)),
        action=jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
        reward=jnp.asarray([5.0, 6.0]),
        done=jnp.asarray([True, False]),
    )

    persisted = streams.persisted(timestep)

    assert persisted.action.tolist() == [[0.0, 0.0], [3.0, 4.0]]
    assert persisted.reward.tolist() == [0.0, 6.0]


def test_normalization_owns_one_state_for_both_estimators() -> None:
    env = TinyContinuousEnv()
    normalization = InteractionNormalization(
        2,
        env,
        observation=NormalizationConfig(center=True, cold_start="first_sample"),
        reward=NormalizationConfig(
            center=False, discount=0.9, cold_start="first_sample"
        ),
        reset_on_start=False,
        update_during_eval=True,
    )

    observation = jnp.asarray([[1.0, 3.0], [5.0, 7.0]])
    normalized, state = normalization.init(observation)

    assert isinstance(state, NormalizationState)
    assert normalized.shape == observation.shape
    assert state.observation is not None
    assert state.reward is not None
    assert normalization.resets_on_start is False
    assert normalization.updates_during_eval is True


def test_streamac_and_rtrrl_use_the_shared_components() -> None:
    stream_ac = importlib.import_module("memorax.algorithms.stream_ac")
    rtrrl = importlib.import_module("memorax.algorithms.rtrrl_aaai")

    assert stream_ac.EnvironmentStreams is EnvironmentStreams
    assert stream_ac.InteractionNormalization is InteractionNormalization
    assert rtrrl.EnvironmentStreams is EnvironmentStreams
    assert rtrrl.InteractionNormalization is InteractionNormalization
