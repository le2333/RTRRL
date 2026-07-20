# pyright: reportMissingImports=false
# ruff: noqa: E402

import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import lox
import pytest

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "base"))

from conftest import TinyContinuousEnv, TinyDiscreteEnv
from golden import assert_tree_allclose

from memorax.algorithms import IndependentRTRRL
from memorax.online_ac import (
    EvalSummary,
    ExactRTRLConfig,
    LegacyProgram,
    MetaProgramConfig,
    NormalizationConfig,
    SlowSubtreeTargetConfig,
    StandardProgramConfig,
    WholeTreeOBGDConfig,
    build_meta_program,
    build_standard_program,
    legacy_env_adapter,
)

_NORMALIZATION_LOG_KEYS = {
    "normalize_observation/mean",
    "normalize_observation/std",
    "normalize_reward/mean",
    "normalize_reward/std",
}


def _rtrrl_config(**overrides):
    values = dict(
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
        b1=0.9,
        b2=0.95,
        eps=1e-6,
        rnn_grad_clip=0.7,
        act_clip=0.05,
        freeze_gamma=True,
        pred_obs=True,
        pred_coeff=0.7,
        update_trace_before_td=False,
        logprob_reduction="mean",
        use_encoder=False,
        encoder_dim=3,
        hidden_dim=2,
        lru_output_dim=3,
        meta_rl=True,
        backbone="lru",
        bound_actor=True,
        normalize_obs=False,
        normalize_reward=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _stream_config(**overrides):
    values = dict(
        agent_type="rtu_rtrl",
        num_envs=1,
        gamma=0.89,
        trace_lambda=0.71,
        actor_lr=0.15,
        critic_lr=0.12,
        actor_kappa=3.0,
        critic_kappa=2.0,
        entropy_coefficient=0.02,
        adaptive=True,
        beta2=0.95,
        eps=1e-6,
        hidden_dim=1,
        encoder_dim=1,
        meta_rl=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _normalized_record_env(env):
    from memorax.environments.wrappers import (
        NormalizeObservationWrapper,
        NormalizeRewardWrapper,
        RecordEpisodeStatistics,
    )

    return NormalizeRewardWrapper(
        NormalizeObservationWrapper(RecordEpisodeStatistics(env))
    )


def _assert_normalized_builder_state_mapping(actual, legacy, *, kind):
    common_fields = (
        (
            "step",
            "update_step",
            "params",
            "slow_torso",
            "traces",
            "opt_state",
            "carry",
            "sensitivity",
            "I",
        )
        if kind == "rtrrl"
        else (
            "step",
            "update_step",
            "actor_params",
            "actor_traces",
            "actor_v",
            "actor_carry",
            "actor_sensitivity",
            "critic_params",
            "critic_traces",
            "critic_v",
            "critic_carry",
            "critic_sensitivity",
        )
    )
    for field in common_fields:
        assert_tree_allclose(
            getattr(actual, field), getattr(legacy, field), rtol=1e-6, atol=1e-7
        )
    assert_tree_allclose(actual.timestep, legacy.timestep, rtol=1e-6, atol=1e-7)

    legacy_reward = legacy.env_state
    legacy_observation = legacy_reward.env_state
    legacy_episode_stats = legacy_observation.env_state
    assert actual.normalizer_state.observation is not None
    assert actual.normalizer_state.reward is not None
    for field in ("mean", "M2", "count"):
        assert_tree_allclose(
            getattr(actual.normalizer_state.observation, field),
            getattr(legacy_observation, field),
            rtol=1e-6,
            atol=1e-7,
        )
    for field in ("mean", "M2", "count", "G"):
        assert_tree_allclose(
            getattr(actual.normalizer_state.reward, field),
            getattr(legacy_reward, field),
            rtol=1e-6,
            atol=1e-7,
        )
    assert_tree_allclose(actual.env_state, legacy_episode_stats, rtol=1e-6, atol=1e-7)


def _assert_normalized_builder_logs(actual, legacy):
    common_legacy_keys = set(legacy) - _NORMALIZATION_LOG_KEYS
    assert common_legacy_keys <= set(actual)
    for key in common_legacy_keys:
        assert_tree_allclose(actual[key], legacy[key], rtol=1e-6, atol=1e-7)
    assert _NORMALIZATION_LOG_KEYS <= set(legacy)
    assert _NORMALIZATION_LOG_KEYS <= set(actual)
    for key in _NORMALIZATION_LOG_KEYS:
        assert_tree_allclose(actual[key], legacy[key], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("kind", ["rtrrl", "stream-ac-rtrl"])
def test_legacy_builder_combined_normalization_train_and_eval_parity(kind):
    from experiment import build_rtrrl_agent, build_stream_ac_agent

    from memorax.algorithms import RTRRL, StreamACRtrl

    raw_env = TinyContinuousEnv() if kind == "rtrrl" else TinyDiscreteEnv()
    env = _normalized_record_env(raw_env)
    cfg = (
        _rtrrl_config(normalize_obs=True, normalize_reward=True)
        if kind == "rtrrl"
        else _stream_config(normalize_obs=True, normalize_reward=True)
    )
    actual_agent = (
        build_rtrrl_agent(cfg, env, env.default_params)
        if kind == "rtrrl"
        else build_stream_ac_agent(cfg, env, env.default_params)
    )
    translated = actual_agent.program_config
    legacy_agent = (
        RTRRL(
            translated.static_config,
            env,  # pyright: ignore[reportArgumentType]
            env.default_params,
            translated.feature_extractor,
            translated.torso,
            translated.actor_head,
            translated.critic_head,
            pred_head=translated.pred_head,
        )
        if kind == "rtrrl"
        else StreamACRtrl(
            translated.static_config,
            env,  # pyright: ignore[reportArgumentType]
            env.default_params,
            translated.actor_network,
            translated.critic_network,
        )
    )

    init_key, train_key, eval_key = jax.random.split(jax.random.key(73), 3)
    legacy_initial = legacy_agent.init(init_key)
    actual_initial = actual_agent.init(init_key)
    _assert_normalized_builder_state_mapping(actual_initial, legacy_initial, kind=kind)

    legacy_trained, legacy_train_logs = lox.spool(legacy_agent.train)(
        train_key, legacy_initial, 3
    )
    actual_trained, actual_train_logs = lox.spool(actual_agent.train)(
        train_key, actual_initial, 3
    )
    _assert_normalized_builder_state_mapping(actual_trained, legacy_trained, kind=kind)
    _assert_normalized_builder_logs(actual_train_logs, legacy_train_logs)

    legacy_evaluated, legacy_eval_logs = lox.spool(legacy_agent.evaluate)(
        eval_key, legacy_trained, 3
    )
    actual_evaluated, actual_eval_logs = lox.spool(actual_agent.evaluate)(
        eval_key, actual_trained, 3
    )
    _assert_normalized_builder_state_mapping(
        actual_evaluated, legacy_evaluated, kind=kind
    )
    _assert_normalized_builder_logs(actual_eval_logs, legacy_eval_logs)


def test_meta_builder_rejects_invalid_composition(rtrrl_agent_factory):
    parts = rtrrl_agent_factory(fresh_trace=False)
    env = legacy_env_adapter(parts.env, parts.env_params)
    base = MetaProgramConfig.from_legacy_parts(parts)

    with pytest.raises(ValueError, match="core-credit"):
        build_meta_program(
            base.replace(credit=object()),
            env,
        )
    with pytest.raises(ValueError, match="target subtree"):
        build_meta_program(
            base.replace(target=SlowSubtreeTargetConfig(subtree=None)),
            env,
        )
    with pytest.raises(ValueError, match="target domain"):
        build_meta_program(
            base.replace(
                target=SlowSubtreeTargetConfig(subtree="torso", gradient_domain="slow")
            ),
            env,
        )


def test_standard_builder_rejects_invalid_update_domain(stream_ac_agent_factory):
    parts = stream_ac_agent_factory(adaptive=False)
    env = legacy_env_adapter(parts.env, parts.env_params)
    base = StandardProgramConfig.from_legacy_parts(parts)

    with pytest.raises(ValueError, match="update domain"):
        build_standard_program(
            base.replace(update=WholeTreeOBGDConfig(domain="actor_only")),
            env,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reset_on_start", "T"),
        ("update_during_eval", "F"),
    ],
)
def test_builder_rejects_non_boolean_evaluation_flags(
    rtrrl_agent_factory, field, value
):
    parts = rtrrl_agent_factory(fresh_trace=False)
    env = legacy_env_adapter(parts.env, parts.env_params)
    config = MetaProgramConfig.from_legacy_parts(parts)
    evaluation = config.evaluation.replace(**{field: value})

    with pytest.raises(ValueError, match="evaluation.*bool"):
        build_meta_program(config.replace(evaluation=evaluation), env)


def test_builder_rejects_double_normalization(rtrrl_agent_factory):
    from memorax.environments.wrappers import NormalizeObservationWrapper

    parts = rtrrl_agent_factory(fresh_trace=False)
    wrapped = NormalizeObservationWrapper(parts.env)
    env = legacy_env_adapter(wrapped, parts.env_params, strip_normalization=False)
    config = MetaProgramConfig.from_legacy_parts(parts).replace(
        normalization=NormalizationConfig(normalize_observation=True)
    )

    with pytest.raises(ValueError, match="normalization owner"):
        build_meta_program(config, env)


def test_rtrrl_legacy_builder_translates_every_field_and_runs_one_step():
    from experiment import build_rtrrl_agent

    from memorax.algorithms import RTRRL, RTRRLConfig
    from memorax.networks import FeatureExtractor, LRUCell, Memoroid, heads

    env = TinyContinuousEnv()
    cfg = _rtrrl_config()
    agent = build_rtrrl_agent(cfg, env, env.default_params)

    assert isinstance(agent, LegacyProgram)
    translated = agent.program_config
    assert isinstance(translated, MetaProgramConfig)
    assert translated.static_config == RTRRLConfig(
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
        b1=0.9,
        b2=0.95,
        eps=1e-6,
        rnn_grad_clip=0.7,
        act_clip=0.05,
        freeze_gamma=True,
        pred_obs=True,
        pred_coeff=0.7,
        update_trace_before_td=False,
        logprob_scale=0.5,
    )
    assert isinstance(translated.feature_extractor, FeatureExtractor)
    assert isinstance(translated.torso, Memoroid)
    assert isinstance(translated.torso.cell, LRUCell)
    assert translated.torso.cell.config.output_dim == 3
    assert isinstance(translated.actor_head, heads.Gaussian)
    assert translated.actor_head.bound is True
    assert isinstance(translated.pred_head, heads.Regressor)
    assert translated.normalization.reward_gamma == 0.99
    assert translated.evaluation.reset_on_start is True
    assert translated.evaluation.update_during_eval is True

    init_key, train_key = jax.random.split(jax.random.key(9))
    oracle = RTRRL(
        translated.static_config,
        env,  # pyright: ignore[reportArgumentType]
        env.default_params,  # pyright: ignore[reportArgumentType]
        translated.feature_extractor,
        translated.torso,
        translated.actor_head,
        translated.critic_head,
        pred_head=translated.pred_head,
    )
    expected_initial = oracle.init(init_key)
    actual_initial = agent.init(init_key)
    assert_tree_allclose(actual_initial, expected_initial, rtol=0, atol=0)
    expected, expected_logs = lox.spool(oracle.train)(train_key, expected_initial, 1)
    actual, actual_logs = lox.spool(agent.train)(train_key, actual_initial, 1)
    assert_tree_allclose(actual, expected)
    assert_tree_allclose(actual_logs, expected_logs)
    assert agent.warmup(train_key, actual, 1) is actual
    eval_key = jax.random.key(10)
    expected_eval, expected_eval_logs = lox.spool(oracle.evaluate)(
        eval_key, expected, 2
    )
    actual_eval, actual_eval_logs = lox.spool(agent.evaluate)(eval_key, actual, 2)
    assert_tree_allclose(actual_eval, expected_eval)
    assert_tree_allclose(actual_eval_logs, expected_eval_logs)


def test_stream_ac_legacy_builder_translates_exact_rtrl_and_runs_one_step():
    from experiment import build_stream_ac_agent

    from memorax.algorithms import StreamACConfig, StreamACRtrl
    from memorax.networks import RNN, FeatureExtractor, RTUCell, heads

    env = TinyDiscreteEnv()
    cfg = _stream_config()
    agent = build_stream_ac_agent(cfg, env, env.default_params)

    assert isinstance(agent, LegacyProgram)
    translated = agent.program_config
    assert isinstance(translated, StandardProgramConfig)
    assert isinstance(translated.credit, ExactRTRLConfig)
    assert translated.update == WholeTreeOBGDConfig(domain="whole_tree")
    assert translated.static_config == StreamACConfig(
        num_envs=1,
        gamma=0.89,
        trace_lambda=0.71,
        actor_lr=0.15,
        critic_lr=0.12,
        actor_kappa=3.0,
        critic_kappa=2.0,
        entropy_coefficient=0.02,
        adaptive=True,
        beta2=0.95,
        eps=1e-6,
    )
    assert translated.actor_network is not translated.critic_network
    assert isinstance(translated.actor_network.feature_extractor, FeatureExtractor)
    assert isinstance(translated.critic_network.feature_extractor, FeatureExtractor)
    assert (
        translated.actor_network.feature_extractor
        is translated.critic_network.feature_extractor
    )
    assert isinstance(translated.actor_network.torso, RNN)
    assert isinstance(translated.actor_network.torso.cell, RTUCell)
    assert translated.actor_network.torso.cell.config.features == 1
    assert translated.actor_network.torso.cell.config.hidden_dim == 1
    assert isinstance(translated.actor_network.head, heads.Categorical)
    assert isinstance(translated.critic_network.head, heads.VNetwork)
    assert translated.normalization == NormalizationConfig(reward_gamma=0.99)
    assert translated.evaluation.reset_on_start is True
    assert translated.evaluation.update_during_eval is True

    init_key, train_key = jax.random.split(jax.random.key(11))
    oracle = StreamACRtrl(
        translated.static_config,
        env,  # pyright: ignore[reportArgumentType]
        env.default_params,  # pyright: ignore[reportArgumentType]
        translated.actor_network,
        translated.critic_network,
    )
    expected_initial = oracle.init(init_key)
    actual_initial = agent.init(init_key)
    assert_tree_allclose(actual_initial, expected_initial, rtol=0, atol=0)
    expected, expected_logs = lox.spool(oracle.train)(train_key, expected_initial, 1)
    actual, actual_logs = lox.spool(agent.train)(train_key, actual_initial, 1)
    assert_tree_allclose(actual, expected)
    assert_tree_allclose(actual_logs, expected_logs)
    eval_key = jax.random.key(12)
    expected_eval, expected_eval_logs = lox.spool(oracle.evaluate)(
        eval_key, expected, 2
    )
    actual_eval, actual_eval_logs = lox.spool(agent.evaluate)(eval_key, actual, 2)
    assert_tree_allclose(actual_eval, expected_eval)
    assert_tree_allclose(actual_eval_logs, expected_eval_logs)


def test_legacy_builder_rejects_legacy_new_config_conflict():
    from experiment import build_rtrrl_agent

    env = TinyContinuousEnv()
    cfg = _rtrrl_config(online_ac_config=object())
    with pytest.raises(ValueError, match="legacy/new"):
        build_rtrrl_agent(cfg, env, env.default_params)


def test_rtrrl_legacy_builder_rejects_removed_recurrent_branch():
    from experiment import build_rtrrl_agent

    from memorax.algorithms.rtrrl.compatibility import UnsupportedRTRRLBranch

    env = TinyContinuousEnv()
    with pytest.raises(UnsupportedRTRRLBranch, match="ctrnn"):
        build_rtrrl_agent(
            _rtrrl_config(rnn_model="ctrnn"),
            env,
            env.default_params,
        )


def test_rtrrl_legacy_builder_preserves_fresh_trace_default_when_omitted():
    from experiment import build_rtrrl_agent

    env = TinyContinuousEnv()
    cfg = _rtrrl_config()
    del cfg.update_trace_before_td

    agent = build_rtrrl_agent(cfg, env, env.default_params)

    assert agent.program_config.static_config.update_trace_before_td is True


def test_independent_rtrrl_builder_and_export_remain_constructible():
    from experiment import build_independent_rtrrl_agent

    env = TinyContinuousEnv()
    agent = build_independent_rtrrl_agent(
        _rtrrl_config(pred_obs=False),
        env,
        env.default_params,
    )
    assert isinstance(agent, IndependentRTRRL)


def test_meta_facade_restores_fixed_training_and_evaluation_logs():
    from experiment import _episode_stats

    from memorax.online_ac import AgentProgram

    info = {
        "returned_episode": jnp.array([[False], [True]]),
        "returned_episode_returns": jnp.array([[0.0], [3.0]]),
        "returned_episode_lengths": jnp.array([[0], [2]]),
    }
    metrics = SimpleNamespace(
        info=info,
        td_error=jnp.array([[1.0], [2.0]]),
        entropy=jnp.array([0.2, 0.3]),
        value=jnp.array([[4.0], [5.0]]),
        emphasis=jnp.array([1.0, 0.9]),
        diag_lambda_max=jnp.array([0.8, 0.81]),
        diag_gamma_max=jnp.array([1.1, 1.2]),
        diag_sens_norm=jnp.array([2.0, 2.1]),
        diag_carry_norm=jnp.array([3.0, 3.1]),
        diag_z_rnn=jnp.array([4.0, 4.1]),
        diag_z_actor=jnp.array([5.0, 5.1]),
        diag_z_critic=jnp.array([6.0, 6.1]),
        diag_grad_rnn=jnp.array([7.0, 7.1]),
        diag_grad_actor=jnp.array([8.0, 8.1]),
        diag_grad_critic=jnp.array([9.0, 9.1]),
        diag_upd_rnn=jnp.array([10.0, 10.1]),
        diag_p_torso=jnp.array([11.0, 11.1]),
        diag_p_actor=jnp.array([12.0, 12.1]),
        diag_p_critic=jnp.array([13.0, 13.1]),
        diag_value_abs=jnp.array([14.0, 14.1]),
        diag_td_abs=jnp.array([15.0, 15.1]),
        diag_actor_loc_abs=jnp.array([16.0, 16.1]),
        diag_actor_scale=jnp.array([17.0, 17.1]),
        diag_act_abs=jnp.array([18.0, 18.1]),
    )
    program = AgentProgram(
        init_fn=lambda key: jnp.asarray(0),
        train_epoch_fn=lambda key, state, steps: (state + steps, metrics),
        evaluate_fn=lambda key, state, steps: (
            state + steps,
            EvalSummary(info=info),
        ),
        state_schema=None,
        metric_schema=None,
    )
    agent = LegacyProgram(program, MetaProgramConfig())

    trained, train_logs = lox.spool(agent.train)(jax.random.key(0), jnp.asarray(2), 2)
    evaluated, eval_logs = lox.spool(agent.evaluate)(jax.random.key(1), trained, 2)

    assert int(trained) == 4
    assert int(evaluated) == 6
    assert set(train_logs) == {
        "info",
        "critic/td_error",
        "actor/entropy",
        "critic/value",
        "emphasis/I",
        "diag/lambda_max",
        "diag/gamma_max",
        "diag/sens_norm",
        "diag/carry_norm",
        "diag/z_rnn",
        "diag/z_actor",
        "diag/z_critic",
        "diag/grad_rnn",
        "diag/grad_actor",
        "diag/grad_critic",
        "diag/upd_rnn",
        "diag/p_torso",
        "diag/p_actor",
        "diag/p_critic",
        "diag/value_abs",
        "diag/td_abs",
        "diag/actor_loc_abs",
        "diag/actor_scale",
        "diag/act_abs",
    }
    assert_tree_allclose(train_logs["info"], info, rtol=0, atol=0)
    assert set(eval_logs) == {"info"}
    assert_tree_allclose(eval_logs["info"], info, rtol=0, atol=0)
    assert _episode_stats(eval_logs["info"]) == (3.0, 2.0)


def test_standard_facade_restores_training_and_evaluation_logs():
    from memorax.online_ac import AgentProgram

    info = {"returned_episode": jnp.array([[True]])}
    metrics = SimpleNamespace(
        info=info,
        td_error=jnp.array([[2.0, 4.0]]),
        entropy=jnp.array([0.25]),
        value=jnp.array([[3.0, 5.0]]),
    )
    program = AgentProgram(
        init_fn=lambda key: jnp.asarray(0),
        train_epoch_fn=lambda key, state, steps: (state, metrics),
        evaluate_fn=lambda key, state, steps: (state, EvalSummary(info=info)),
        state_schema=None,
        metric_schema=None,
    )
    agent = LegacyProgram(program, StandardProgramConfig())

    _, train_logs = lox.spool(agent.train)(jax.random.key(0), jnp.asarray(0), 1)
    _, eval_logs = lox.spool(agent.evaluate)(jax.random.key(1), jnp.asarray(0), 1)

    assert set(train_logs) == {
        "info",
        "critic/td_error",
        "actor/entropy",
        "critic/value",
    }
    assert set(eval_logs) == {"info"}
    assert_tree_allclose(eval_logs["info"], info, rtol=0, atol=0)


def test_stream_ac_builder_preserves_rtu_tbptt_legacy_path():
    from experiment import build_stream_ac_agent

    from memorax.algorithms import StreamAC

    env = TinyDiscreteEnv()
    agent = build_stream_ac_agent(
        _stream_config(agent_type="rtu_tbptt"),
        env,
        env.default_params,
    )

    assert isinstance(agent, StreamAC)
    assert not isinstance(agent, LegacyProgram)


def test_legacy_adapter_rejects_unstrippable_inner_normalization():
    from memorax.environments.wrappers import (
        NormalizeObservationWrapper,
        RecordEpisodeStatistics,
    )

    env = RecordEpisodeStatistics(NormalizeObservationWrapper(TinyContinuousEnv()))
    with pytest.raises(ValueError, match="inner normalization"):
        legacy_env_adapter(env, env.default_params)
