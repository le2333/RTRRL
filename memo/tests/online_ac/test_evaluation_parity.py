from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import lox
import numpy as np
import pytest
from conftest import TinyContinuousEnv
from golden import assert_tree_allclose

from memorax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
)
from memorax.rl import (
    NormalizationConfig,
    environment_owns_normalization,
    make_normalizer,
)


def _assert_unchanged_parameters(before, after):
    for field in (
        "params",
        "slow_torso",
        "traces",
        "opt_state",
        "actor_params",
        "actor_traces",
        "actor_v",
        "critic_params",
        "critic_traces",
        "critic_v",
    ):
        if hasattr(before, field):
            assert_tree_allclose(
                getattr(after, field), getattr(before, field), rtol=0, atol=0
            )


def test_explicit_normalizer_matches_wrapper_welford_order_and_raw_return():
    normalizer = make_normalizer(
        NormalizationConfig(normalize_observation=True, normalize_reward=True)
    )
    initial_obs = jnp.array([[2.0, -1.0]], dtype=jnp.float32)
    normalized_obs, state = normalizer.reset(initial_obs)
    assert state.observation is not None

    np.testing.assert_allclose(state.observation.mean, [[1.0, -0.5]])
    np.testing.assert_allclose(state.observation.M2, [[3.0, 1.5]])
    np.testing.assert_allclose(state.observation.count, [2.0])
    np.testing.assert_allclose(
        normalized_obs,
        (initial_obs - state.observation.mean)
        / jnp.sqrt(state.observation.M2 / 2 + 1e-8),
    )

    first = normalizer.step(
        state,
        observation=jnp.array([[4.0, 2.0]], dtype=jnp.float32),
        reward=jnp.array([2.0], dtype=jnp.float32),
        done=jnp.array([False]),
    )
    second = normalizer.step(
        first.state,
        observation=jnp.array([[8.0, 3.0]], dtype=jnp.float32),
        reward=jnp.array([3.0], dtype=jnp.float32),
        done=jnp.array([True]),
    )
    third = normalizer.step(
        second.state,
        observation=jnp.array([[1.0, 5.0]], dtype=jnp.float32),
        reward=jnp.array([5.0], dtype=jnp.float32),
        done=jnp.array([False]),
    )
    assert first.state.reward is not None
    assert second.state.reward is not None
    assert third.state.observation is not None
    assert third.state.reward is not None

    np.testing.assert_allclose(first.state.reward.G, [2.0])
    np.testing.assert_allclose(second.state.reward.G, [0.0])
    np.testing.assert_allclose(third.state.reward.G, [5.0])
    np.testing.assert_allclose(second.raw_episode_return, [5.0])
    np.testing.assert_allclose(third.state.episode_return, [5.0])
    assert float(third.state.observation.count[0]) == 5.0
    assert float(third.state.reward.count[0]) == 4.0


def test_explicit_normalizer_matches_existing_wrappers_step_by_step():
    raw_env: Any = TinyContinuousEnv()
    wrapped_env: Any = NormalizeObservationWrapper(NormalizeRewardWrapper(raw_env))
    normalizer = make_normalizer(
        NormalizationConfig(normalize_observation=True, normalize_reward=True)
    )
    reset_key = jax.random.key(41)
    wrapped_obs, wrapped_state = wrapped_env.reset(reset_key, raw_env.default_params)
    raw_obs, raw_state = raw_env.reset(reset_key, raw_env.default_params)
    explicit_obs, explicit_state = normalizer.reset(raw_obs[None, ...])
    assert explicit_state.observation is not None

    np.testing.assert_allclose(explicit_obs[0], wrapped_obs, rtol=0, atol=0)
    np.testing.assert_allclose(
        explicit_state.observation.mean[0], wrapped_state.mean, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        explicit_state.observation.M2[0], wrapped_state.M2, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        explicit_state.observation.count[0], wrapped_state.count, rtol=0, atol=0
    )

    raw_return = 0.0
    explicit = None
    for index in range(3):
        key = jax.random.fold_in(jax.random.key(42), index)
        action = jnp.zeros((2,), dtype=jnp.float32)
        wrapped_obs, wrapped_state, wrapped_reward, wrapped_done, _ = wrapped_env.step(
            key, wrapped_state, action, raw_env.default_params
        )
        raw_obs, raw_state, raw_reward, raw_done, _ = raw_env.step(
            key, raw_state, action, raw_env.default_params
        )
        explicit = normalizer.step(
            explicit_state,
            observation=raw_obs[None, ...],
            reward=raw_reward[None],
            done=raw_done[None],
        )
        explicit_state = explicit.state
        assert explicit_state.observation is not None
        assert explicit_state.reward is not None
        raw_return += float(raw_reward)

        np.testing.assert_allclose(
            explicit.observation[0], wrapped_obs, rtol=1e-6, atol=1e-7
        )
        np.testing.assert_allclose(
            explicit.reward[0], wrapped_reward, rtol=1e-6, atol=1e-7
        )
        np.testing.assert_allclose(
            explicit_state.observation.mean[0],
            wrapped_state.mean,
            rtol=0,
            atol=0,
        )
        reward_wrapper_state: Any = wrapped_state.env_state
        np.testing.assert_allclose(
            explicit_state.reward.mean[0],
            reward_wrapper_state.mean,
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(
            explicit_state.reward.M2[0],
            reward_wrapper_state.M2,
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(
            explicit_state.reward.G[0],
            reward_wrapper_state.G,
            rtol=0,
            atol=0,
        )
        assert bool(raw_done) == bool(wrapped_done)

    assert explicit is not None
    np.testing.assert_allclose(explicit.raw_episode_return, [raw_return])
    np.testing.assert_allclose(explicit_state.episode_return, [0.0])


def test_frozen_normalizer_uses_copied_statistics_without_mutating_them():
    normalizer = make_normalizer(
        NormalizationConfig(normalize_observation=True, normalize_reward=True)
    )
    _, state = normalizer.reset(jnp.array([[1.0, 2.0]], dtype=jnp.float32))
    trained = normalizer.step(
        state,
        observation=jnp.array([[3.0, 6.0]], dtype=jnp.float32),
        reward=jnp.array([2.0], dtype=jnp.float32),
        done=jnp.array([False]),
    ).state
    frozen = normalizer.step(
        trained,
        observation=jnp.array([[20.0, 30.0]], dtype=jnp.float32),
        reward=jnp.array([7.0], dtype=jnp.float32),
        done=jnp.array([False]),
        update=False,
    )
    assert trained.observation is not None

    assert_tree_allclose(frozen.state.observation, trained.observation, rtol=0, atol=0)
    assert_tree_allclose(frozen.state.reward, trained.reward, rtol=0, atol=0)
    np.testing.assert_allclose(
        frozen.observation,
        (jnp.array([[20.0, 30.0]]) - trained.observation.mean)
        / jnp.sqrt(trained.observation.M2 / trained.observation.count[:, None] + 1e-8),
    )


@pytest.mark.parametrize(
    ("reset_on_start", "update_during_eval"),
    [(True, True), (False, False), (False, True)],
)
@pytest.mark.parametrize("kind", ["meta", "standard"])
def test_evaluation_modes_preserve_training_leaves_and_return_temporary_state(
    kind,
    reset_on_start,
    update_during_eval,
    rtrrl_agent_factory,
    stream_ac_agent_factory,
):
    normalization = NormalizationConfig(
        normalize_observation=True,
        normalize_reward=True,
        reset_on_start=reset_on_start,
        update_during_eval=update_during_eval,
    )
    if kind == "meta":
        from memorax.online_ac.meta import make_meta_program

        parts = rtrrl_agent_factory(fresh_trace=False)
        program = make_meta_program(
            parts, parts.cfg, normalization_config=normalization
        )
    else:
        from memorax.online_ac.standard import make_standard_program

        parts = stream_ac_agent_factory(adaptive=False)
        program = make_standard_program(
            parts, parts.cfg, normalization_config=normalization
        )

    state = program.init_fn(jax.random.key(3))
    trained, _ = program.train_epoch_fn(jax.random.key(4), state, num_steps=1)
    before = jax.tree.map(lambda x: jnp.array(x), trained)
    evaluated, summary = program.evaluate_fn(jax.random.key(5), trained, num_steps=2)

    _assert_unchanged_parameters(before, evaluated)
    assert summary.info["step_count"].shape == (2, 1)
    assert_tree_allclose(trained, before, rtol=0, atol=0)
    if reset_on_start:
        assert np.all(np.asarray(evaluated.normalizer_state.observation.count) == 4)
    elif update_during_eval:
        assert np.all(
            np.asarray(evaluated.normalizer_state.observation.count)
            == np.asarray(trained.normalizer_state.observation.count) + 3
        )
    else:
        assert_tree_allclose(
            evaluated.normalizer_state.observation,
            trained.normalizer_state.observation,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("kind", ["meta", "standard"])
def test_reset_without_eval_updates_is_a_build_error(
    kind, rtrrl_agent_factory, stream_ac_agent_factory
):
    normalization = NormalizationConfig(
        normalize_observation=True,
        reset_on_start=True,
        update_during_eval=False,
    )
    with pytest.raises(ValueError, match="reset_on_start"):
        if kind == "meta":
            from memorax.online_ac.meta import make_meta_program

            parts = rtrrl_agent_factory(fresh_trace=False)
            make_meta_program(parts, parts.cfg, normalization_config=normalization)
        else:
            from memorax.online_ac.standard import make_standard_program

            parts = stream_ac_agent_factory(adaptive=False)
            make_standard_program(parts, parts.cfg, normalization_config=normalization)


def test_meta_legacy_evaluate_matches_leaf_for_leaf(rtrrl_agent_factory):
    from memorax.online_ac.meta import make_meta_program

    legacy = rtrrl_agent_factory(fresh_trace=False)
    parts = rtrrl_agent_factory(fresh_trace=False)
    program = make_meta_program(parts, parts.cfg)
    init_key = jax.random.key(11)
    eval_key = jax.random.key(12)
    expected_initial = legacy.init(init_key)
    actual_initial = program.init_fn(init_key)

    expected, logs = lox.spool(legacy.evaluate)(eval_key, expected_initial, num_steps=4)
    actual, summary = program.evaluate_fn(eval_key, actual_initial, num_steps=4)

    assert_tree_allclose(actual, expected, rtol=0, atol=0)
    assert_tree_allclose(summary.info, logs["info"], rtol=0, atol=0)
    _assert_unchanged_parameters(actual_initial, actual)


def test_standard_legacy_evaluate_matches_leaf_for_leaf(stream_ac_agent_factory):
    from memorax.online_ac.standard import make_standard_program

    legacy = stream_ac_agent_factory(adaptive=False)
    parts = stream_ac_agent_factory(adaptive=False)
    program = make_standard_program(parts, parts.cfg)
    init_key = jax.random.key(21)
    eval_key = jax.random.key(22)
    expected_initial = legacy.init(init_key)
    actual_initial = program.init_fn(init_key)

    expected, logs = lox.spool(legacy.evaluate)(eval_key, expected_initial, num_steps=4)
    actual, summary = program.evaluate_fn(eval_key, actual_initial, num_steps=4)

    assert_tree_allclose(actual, expected, rtol=0, atol=0)
    assert_tree_allclose(summary.info, logs["info"], rtol=0, atol=0)
    _assert_unchanged_parameters(actual_initial, actual)


@pytest.mark.parametrize("kind", ["meta", "standard"])
def test_program_rejects_wrapper_and_explicit_normalization_double_owner(
    kind, rtrrl_agent_factory, stream_ac_agent_factory
):
    from memorax.environments.wrappers import NormalizeObservationWrapper

    normalization = NormalizationConfig(normalize_observation=True)
    if kind == "meta":
        from memorax.online_ac.meta import make_meta_program

        parts = rtrrl_agent_factory(fresh_trace=False)
        parts = replace(parts, env=NormalizeObservationWrapper(parts.env))
        build = make_meta_program
    else:
        from memorax.online_ac.standard import make_standard_program

        parts = stream_ac_agent_factory(adaptive=False)
        parts = replace(parts, env=NormalizeObservationWrapper(parts.env))
        build = make_standard_program
    with pytest.raises(ValueError, match="normalization owner"):
        build(parts, parts.cfg, normalization_config=normalization)


class ObservationWrapperSubclass(NormalizeObservationWrapper):
    pass


class RewardWrapperSubclass(NormalizeRewardWrapper):
    pass


@pytest.mark.parametrize(
    "wrapped",
    [
        NormalizeObservationWrapper(TinyContinuousEnv()),
        NormalizeRewardWrapper(TinyContinuousEnv()),
        ObservationWrapperSubclass(TinyContinuousEnv()),
        RewardWrapperSubclass(TinyContinuousEnv()),
    ],
    ids=[
        "observation",
        "reward",
        "observation-subclass",
        "reward-subclass",
    ],
)
def test_owner_detection_accepts_real_normalization_wrappers_and_subclasses(wrapped):
    assert environment_owns_normalization(wrapped)


@pytest.mark.parametrize(
    "class_name",
    ["NormalizeObservationWrapper", "NormalizeRewardWrapper"],
)
def test_owner_detection_rejects_unrelated_same_named_classes(class_name):
    fake_type = type(class_name, (), {})
    assert not environment_owns_normalization(fake_type())


def test_config_like_algorithm_gamma_does_not_override_reward_normalizer_gamma():
    normalizer = make_normalizer(SimpleNamespace(normalize_reward=True, gamma=0.73))

    assert normalizer.config.reward_gamma == 0.99


def test_config_like_explicit_reward_normalizer_gamma_remains_configurable():
    normalizer = make_normalizer(
        SimpleNamespace(
            normalize_reward=True,
            gamma=0.73,
            normalization_reward_gamma=0.87,
        )
    )

    assert normalizer.config.reward_gamma == 0.87
