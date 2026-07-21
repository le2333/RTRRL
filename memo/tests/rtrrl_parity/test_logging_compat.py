"""Historical RTRRL metric aggregation and logger-step compatibility."""

# pyright: reportMissingImports=false
# ruff: noqa: E402

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

WORKTREE_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(WORKTREE_ROOT / "memo"))
sys.path.insert(0, str(WORKTREE_ROOT / "memo" / "experiments" / "base"))

from experiment import (  # noqa: E402
    _historical_rtrrl_metrics,
    _log_historical_rtrrl_epoch,
    train_loop,
)
from memorax.algorithms.rtrrl import RTRRL
from memorax.algorithms.rtrrl.program import (
    aggregate_epoch_summary,
    build_rtrrl_program,
)
from memorax.algorithms.rtrrl.types import TrainStepMetrics

from .test_init_parity import _strict_setup
from .test_step_parity import _ThreeStepEnvironment


def test_logging_contract_uses_memo_logging_utility():
    import logging_util

    assert Path(logging_util.__file__).resolve() == (
        WORKTREE_ROOT / "memo" / "logging_util.py"
    ).resolve()


class _RecordingLogger(dict):
    def __init__(self):
        super().__init__(best_eval_reward=-jnp.inf)
        self.calls = []
        self.videos = []

    def log(self, metrics, step):
        self.calls.append((metrics, step))

    def log_video(self, name, frames, *, fps, caption):
        self.videos.append((name, frames, fps, caption))

    def log_params(self, params):
        self.params = params

    def finalize(self):
        self.finalized = True


@dataclass
class _LoopConfig:
    seed: int | None = 7
    total_timesteps: int = 3
    num_epochs: int = 1
    num_envs: int = 1
    eval_every: int = 1
    eval_steps: int = 3
    patience: int = 0
    log_every: int = 1
    log_norms: bool = True
    render: bool = True

def _synthetic_summary():
    step_metrics = TrainStepMetrics(
        reward=jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
        done=jnp.array([0.0, 0.5, 1.0], dtype=jnp.float32),
        td_error_mean=jnp.array([0.5, -0.25, 1.0], dtype=jnp.float32),
        value_mean=jnp.array([10.0, 20.0, 30.0], dtype=jnp.float32),
        value_target_mean=jnp.array([11.0, 19.0, 32.0], dtype=jnp.float32),
        entropy_mean=jnp.array([0.2, 0.4, 0.6], dtype=jnp.float32),
        actor_loss_mean=jnp.array([-0.3, -0.6, -0.9], dtype=jnp.float32),
    )
    final_state = SimpleNamespace(
        average_reward=jnp.array([0.3], dtype=jnp.float32),
        traces={"trace": jnp.array([3.0, 4.0], dtype=jnp.float32)},
        parameters={"weight": jnp.array([6.0, 8.0], dtype=jnp.float32)},
        slow_parameters={"weight": jnp.array([5.0, 12.0], dtype=jnp.float32)},
        optimizer_state=None,
        step_count=jnp.array(15, dtype=jnp.int32),
    )
    return aggregate_epoch_summary(
        step_metrics,
        final_state,
        num_steps=3,
        num_envs=2,
        learning_rate_td=jnp.array(1e-3, dtype=jnp.float32),
        learning_rate_rnn=jnp.array(2e-4, dtype=jnp.float32),
    )


def test_historical_aggregation_names_values_and_dtypes():
    summary = _synthetic_summary()
    metrics = _historical_rtrrl_metrics(
        summary,
        log_td_lr=True,
        log_rnn_lr=True,
        log_norms=True,
    )

    expected_core = {
        "steps",
        "mean_reward",
        "num_episodes",
        "mean_delta",
        "mean_r_bar",
        "mean_v",
        "total_td_loss",
        "actor_loss",
        "critic_loss",
        "entropy",
        "v_targ",
        "lr/td",
        "lr/rnn",
    }
    assert expected_core <= metrics.keys()
    assert {
        "norms/['z']['trace']",
        "norms/['params']['weight']",
        "norms/['slow_params']['weight']",
    } <= metrics.keys()
    assert "magnitude_loss" not in metrics

    assert metrics["steps"].dtype == jnp.int32
    assert metrics["num_episodes"].dtype == jnp.int32
    for key in metrics.keys() - {"steps", "num_episodes"}:
        assert metrics[key].dtype == jnp.float32, key

    np.testing.assert_allclose(metrics["mean_reward"], 4.0)
    np.testing.assert_allclose(metrics["mean_delta"], 2.5 / 3.0)
    np.testing.assert_allclose(metrics["mean_r_bar"], 0.1)
    np.testing.assert_allclose(metrics["mean_v"], 20.0)
    np.testing.assert_allclose(metrics["actor_loss"], -0.6)
    np.testing.assert_allclose(metrics["critic_loss"], 20.0)
    np.testing.assert_allclose(metrics["total_td_loss"], 19.4)
    np.testing.assert_allclose(metrics["entropy"], 0.4)
    np.testing.assert_allclose(metrics["v_targ"], 62.0 / 3.0)
    np.testing.assert_allclose(metrics["norms/['z']['trace']"], 5.0)
    np.testing.assert_allclose(metrics["norms/['params']['weight']"], 10.0)
    np.testing.assert_allclose(
        metrics["norms/['slow_params']['weight']"], 13.0
    )
    assert int(metrics["steps"]) == 30


def test_optional_historical_keys_follow_configuration():
    summary = _synthetic_summary().replace(  # pyright: ignore[reportAttributeAccessIssue]
        magnitude_loss=jnp.array(0.75, dtype=jnp.float32)
    )
    metrics = _historical_rtrrl_metrics(
        summary,
        log_td_lr=False,
        log_rnn_lr=False,
        log_norms=False,
    )

    assert metrics["magnitude_loss"].dtype == jnp.float32
    assert float(metrics["magnitude_loss"]) == 0.75
    assert "lr/td" not in metrics
    assert "lr/rnn" not in metrics
    assert not any(key.startswith("norms/") for key in metrics)


def test_logger_uses_epoch_scan_steps_not_batched_metric_steps():
    logger = _RecordingLogger()
    summary = _synthetic_summary()
    components, config, _ = _strict_setup()
    program = build_rtrrl_program(config, components, _ThreeStepEnvironment())
    state = program.init_fn(jax.random.key(41))
    _, evaluation = program.evaluate_fn(jax.random.key(43), state, 3)
    rendered_states = []

    def render_evaluation(environment_state):
        rendered_states.append(environment_state)
        return np.zeros((2, 3, 4, 3), dtype=np.uint8)

    _log_historical_rtrrl_epoch(
        logger,
        summary,
        epoch_index=4,
        steps_per_epoch=3,
        log_every=1,
        log_td_lr=True,
        log_rnn_lr=True,
        log_norms=True,
        evaluation_summary=evaluation,
        render_evaluation=render_evaluation,
        render_start=1,
        render_steps=5,
    )

    assert len(logger.calls) == 1
    metrics, logger_step = logger.calls[0]
    assert logger_step == 15
    assert int(metrics["steps"]) == 30
    np.testing.assert_allclose(metrics["eval/rewards"], 1.125)
    np.testing.assert_allclose(metrics["eval/best_eval_reward"], 1.125)
    np.testing.assert_allclose(logger["best_eval_reward"], 1.125)
    assert len(logger.videos) == 1
    assert logger.videos[0][0] == "env/video"
    assert len(rendered_states) == 1
    assert jax.tree.leaves(rendered_states[0])[0].shape[0] == 2

    _log_historical_rtrrl_epoch(
        logger,
        summary,
        epoch_index=5,
        steps_per_epoch=3,
        log_every=2,
        log_td_lr=False,
        log_rnn_lr=False,
        log_norms=False,
        evaluation_summary=evaluation,
    )
    second_metrics, second_step = logger.calls[1]
    assert second_step == 18
    assert set(second_metrics) == {"eval/rewards"}
    assert "eval/best_eval_reward" not in second_metrics


def test_video_render_slice_honors_exact_configured_window():
    logger = _RecordingLogger()
    components, config, _ = _strict_setup()
    program = build_rtrrl_program(config, components, _ThreeStepEnvironment())
    state = program.init_fn(jax.random.key(59))
    _, evaluation = program.evaluate_fn(jax.random.key(61), state, 3)
    rendered_states = []

    _log_historical_rtrrl_epoch(
        logger,
        _synthetic_summary(),
        epoch_index=0,
        steps_per_epoch=3,
        log_every=1,
        log_td_lr=False,
        log_rnn_lr=False,
        log_norms=False,
        evaluation_summary=evaluation,
        render_evaluation=lambda states: (
            rendered_states.append(states)
            or np.zeros((1, 2, 2, 3), dtype=np.uint8)
        ),
        render_start=1,
        render_steps=1,
    )

    leaves = jax.tree.leaves(rendered_states[0])
    assert leaves
    assert all(leaf.shape[0] == 1 for leaf in leaves)


def test_normalized_evaluation_logs_original_episode_return():
    import experiment
    from conftest import TinyContinuousEnv
    from memorax.environments.wrappers import (
        NormalizeRewardWrapper,
        RecordEpisodeStatistics,
    )

    env = NormalizeRewardWrapper(
        RecordEpisodeStatistics(TinyContinuousEnv())
    )
    cfg = SimpleNamespace(
        profile="memo_experimental",
        num_envs=1,
        hidden_dim=3,
        normalize_obs=False,
        normalize_reward=True,
    )
    agent = experiment.build_rtrrl_agent(cfg, env, env.default_params)
    state = agent.init(jax.random.key(67))
    _, evaluation = agent.evaluate_summary(jax.random.key(71), state, 3)
    logger = _RecordingLogger()

    _log_historical_rtrrl_epoch(
        logger,
        _synthetic_summary(),
        epoch_index=0,
        steps_per_epoch=3,
        log_every=1,
        log_td_lr=False,
        log_rnn_lr=False,
        log_norms=False,
        evaluation_summary=evaluation,
    )

    normalized_total = float(jnp.sum(evaluation.info["reward"]))
    original_return = float(
        jnp.max(evaluation.info["returned_episode_returns"])
    )
    assert "environment_info" in evaluation.info
    assert normalized_total != original_return
    np.testing.assert_allclose(original_return, 3.3, rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        logger.calls[0][0]["eval/rewards"],
        original_return,
        rtol=0,
        atol=1e-6,
    )


def test_enabled_action_magnitude_has_real_step_source():
    components, config, _ = _strict_setup()
    config = replace(config, action_magnitude_factor=0.25)
    program = build_rtrrl_program(config, components, _ThreeStepEnvironment())
    state = program.init_fn(jax.random.key(47))

    _, summary = program.train_epoch_fn(jax.random.key(53), state, 1)

    assert summary.magnitude_loss is not None
    assert summary.magnitude_loss.dtype == jnp.float32
    assert bool(jnp.isfinite(summary.magnitude_loss))


def test_real_strict_train_loop_emits_historical_program_metrics(monkeypatch):
    components, config, _ = _strict_setup()
    program = build_rtrrl_program(config, components, _ThreeStepEnvironment())
    renderer_calls = []
    agent = RTRRL.from_program(
        program,
        profile="aaai25_strict_lru",
        num_envs=1,
        runtime_config=SimpleNamespace(
            optimizer_params_td=SimpleNamespace(decay_type="exponential"),
            optimizer_params_rnn=SimpleNamespace(decay_type="exponential"),
            log_norms=True,
            env_params=SimpleNamespace(render=True),
            render_every_evals=10,
            render_start=1,
            render_steps=1,
        ),
        render_evaluation=lambda states: (
            renderer_calls.append(states)
            or np.zeros((1, 2, 2, 3), dtype=np.uint8)
        ),
    )
    logger = _RecordingLogger()
    monkeypatch.setattr(
        "experiment.trange", lambda count, mininterval: range(count)
    )

    result = train_loop(agent, _LoopConfig(), logger)

    assert float(result) == float(logger["best_eval_reward"])
    assert logger.finalized is True
    assert len(logger.calls) == 1
    metrics, step = logger.calls[0]
    assert step == 3
    assert {
        "steps",
        "mean_reward",
        "num_episodes",
        "mean_delta",
        "mean_r_bar",
        "mean_v",
        "total_td_loss",
        "actor_loss",
        "critic_loss",
        "entropy",
        "v_targ",
        "lr/td",
        "lr/rnn",
        "eval/rewards",
        "eval/best_eval_reward",
    } <= metrics.keys()
    assert any(key.startswith("norms/") for key in metrics)
    assert int(metrics["steps"]) == 3
    assert len(renderer_calls) == 1
    assert jax.tree.leaves(renderer_calls[0])[0].shape[0] == 1
    assert logger.videos[0][0] == "env/video"
