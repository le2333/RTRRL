"""Historical RTRRL metric aggregation and logger-step compatibility."""

# pyright: reportMissingImports=false
# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

WORKTREE_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(WORKTREE_ROOT / "rtrrl"))
sys.path.insert(0, str(WORKTREE_ROOT / "memo" / "experiments" / "base"))

from experiment import (  # noqa: E402
    _historical_rtrrl_metrics,
    _log_historical_rtrrl_epoch,
)
from memorax.algorithms.rtrrl.program import aggregate_epoch_summary
from memorax.algorithms.rtrrl.types import TrainStepMetrics


class _RecordingLogger(dict):
    def __init__(self):
        super().__init__(best_eval_reward=-jnp.inf)
        self.calls = []
        self.videos = []

    def log(self, metrics, step):
        self.calls.append((metrics, step))

    def log_video(self, name, frames, *, fps, caption):
        self.videos.append((name, frames, fps, caption))


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
    frames = np.zeros((2, 3, 4, 3), dtype=np.uint8)

    _log_historical_rtrrl_epoch(
        logger,
        summary,
        epoch_index=4,
        steps_per_epoch=3,
        log_every=1,
        log_td_lr=True,
        log_rnn_lr=True,
        log_norms=True,
        eval_reward=7.5,
        video_frames=frames,
    )

    assert len(logger.calls) == 1
    metrics, logger_step = logger.calls[0]
    assert logger_step == 15
    assert int(metrics["steps"]) == 6
    assert metrics["eval/rewards"] == 7.5
    assert metrics["eval/best_eval_reward"] == 7.5
    assert logger["best_eval_reward"] == 7.5
    assert len(logger.videos) == 1
    assert logger.videos[0][0] == "env/video"

    _log_historical_rtrrl_epoch(
        logger,
        summary,
        epoch_index=5,
        steps_per_epoch=3,
        log_every=2,
        log_td_lr=False,
        log_rnn_lr=False,
        log_norms=False,
        eval_reward=6.0,
    )
    second_metrics, second_step = logger.calls[1]
    assert second_step == 18
    assert second_metrics == {"eval/rewards": 6.0}
    assert "eval/best_eval_reward" not in second_metrics
