from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

MEMO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(MEMO_ROOT))
sys.path.insert(0, str(MEMO_ROOT / "experiments"))

from base.experiment import (  # noqa: E402
    ExperimentConfig,
    _epoch_env_steps,
    emit_evaluation_episodes,
    emit_training_summaries,
    episode_from_trace,
    train_loop,
)
from memorax.online_ac.build import LegacyProgram  # noqa: E402
from memorax.online_ac.types import EvalSummary, EvalTrace  # noqa: E402


class RecordingRun:
    def __init__(self):
        self.summaries = []
        self.episodes = []

    def log_episode_summary(self, **summary):
        self.summaries.append(summary)

    def log_episode(self, episode):
        self.episodes.append(episode)


def _trace(*, count=2, terminal=True, truncated=False):
    return EvalTrace(
        observations=jnp.asarray([[[0.0]], [[1.0]], [[2.0]], [[9.0]]]),
        actions=jnp.asarray([[[0.1]], [[0.2]], [[0.3]]]),
        rewards=jnp.asarray([[1.0], [2.0], [3.0]]),
        terminals=jnp.asarray([[False], [terminal], [False]]),
        truncations=jnp.asarray([[False], [truncated], [False]]),
        valid_transitions=jnp.asarray([count]),
        environment_states=jnp.asarray([[[4.0]], [[5.0]], [[6.0]]]),
    )


def test_each_completed_training_episode_is_emitted_at_real_state_step():
    run = RecordingRun()
    info = {
        "returned_episode": np.array(
            [[False, True], [True, False], [True, True]]
        ),
        "returned_episode_returns": np.array(
            [[0.0, 2.0], [3.0, 0.0], [4.0, 5.0]]
        ),
        "returned_episode_lengths": np.array(
            [[0, 7], [9, 0], [2, 3]]
        ),
    }

    emit_training_summaries(
        run,
        info,
        epoch_start_env_steps=100,
        num_envs=2,
    )

    assert run.summaries == [
        {"env_steps": 102, "episode_return": 2.0, "episode_length": 7},
        {"env_steps": 104, "episode_return": 3.0, "episode_length": 9},
        {"env_steps": 106, "episode_return": 4.0, "episode_length": 2},
        {"env_steps": 106, "episode_return": 5.0, "episode_length": 3},
    ]


def test_budget_is_partitioned_as_total_environment_interactions():
    epochs = _epoch_env_steps(total=24, epochs=4, num_envs=3)
    assert epochs == (6, 6, 6, 6)
    assert sum(epochs) == 24
    assert all(item % 3 == 0 for item in epochs)


def test_budget_rejects_unrepresentable_vector_interactions():
    with pytest.raises(ValueError, match="divisible"):
        _epoch_env_steps(total=10, epochs=2, num_envs=3)
    with pytest.raises(ValueError, match="epochs"):
        _epoch_env_steps(total=30, epochs=4, num_envs=3)


@pytest.mark.parametrize(
    ("terminal", "truncated"), [(True, False), (False, True)]
)
def test_trace_conversion_preserves_n_plus_one_and_ending(terminal, truncated):
    episode = episode_from_trace(
        _trace(terminal=terminal, truncated=truncated),
        environment_index=0,
        episode_number=4,
        env_steps=100,
    )
    assert len(episode.observations) == 3
    assert len(episode.actions) == len(episode.rewards) == 2
    assert episode.terminals[-1] is terminal
    assert episode.truncations[-1] is truncated
    assert episode.start_env_steps == 100
    assert episode.end_env_steps == 100
    assert len(episode.environment_states) == 2


def test_trace_conversion_rejects_nonpositive_or_incomplete_trace():
    with pytest.raises(ValueError, match="valid_transitions"):
        episode_from_trace(
            _trace(count=0),
            environment_index=0,
            episode_number=1,
            env_steps=1,
        )
    with pytest.raises(ValueError, match="complete"):
        episode_from_trace(
            _trace(terminal=False, truncated=False),
            environment_index=0,
            episode_number=1,
            env_steps=2,
        )


def test_all_complete_eval_environments_are_logged_each_cadence():
    run = RecordingRun()
    trace = _trace()
    two = trace.replace(
        observations=jnp.repeat(trace.observations, 2, axis=1),
        actions=jnp.repeat(trace.actions, 2, axis=1),
        rewards=jnp.repeat(trace.rewards, 2, axis=1),
        terminals=jnp.repeat(trace.terminals, 2, axis=1),
        truncations=jnp.repeat(trace.truncations, 2, axis=1),
        valid_transitions=jnp.asarray([2, 0]),
        environment_states=jnp.repeat(trace.environment_states, 2, axis=1),
    )

    next_number = emit_evaluation_episodes(
        run, two, first_episode_number=7, env_steps=20
    )

    assert [episode.number for episode in run.episodes] == [7]
    assert next_number == 8


@dataclass
class FakeProgram:
    trace: EvalTrace

    def evaluate_fn(self, key, state, num_steps):
        del key, num_steps
        return state + 1, EvalSummary(info={"x": jnp.ones((1,))}, trace=self.trace)


def test_legacy_program_evaluate_exposes_trace(monkeypatch):
    monkeypatch.setattr(
        "memorax.online_ac.build._emit_step_logs", lambda logs: None
    )
    legacy = LegacyProgram(FakeProgram(_trace()), object())

    state, trace = legacy.evaluate(None, 4, 3)

    assert state == 5
    assert trace is not None


def test_train_loop_bridges_fake_state_step_and_completion_offsets(monkeypatch):
    import base.experiment as experiment_module

    @dataclass
    class State:
        step: int

    class Agent:
        def init(self, key):
            del key
            return State(0)

        def train(self, key, state, num_steps):
            del key
            assert num_steps == 4
            info = {
                "returned_episode": np.array(
                    [[False, True], [True, False]]
                ),
                "returned_episode_returns": np.array(
                    [[0.0, state.step + 2.0], [state.step + 4.0, 0.0]]
                ),
                "returned_episode_lengths": np.array([[0, 2], [4, 0]]),
            }
            return State(state.step + num_steps), {"info": info}

        def evaluate(self, key, state, num_steps):
            raise AssertionError("evaluation is disabled")

    class Logger(dict):
        def __init__(self):
            super().__init__()
            self.summaries = []
            self.finalized = 0
            self.failed = []

        def log_params(self, params):
            del params

        def log_episode_summary(self, **summary):
            self.summaries.append(summary)

        def log(self, metrics, step):
            assert metrics["train/env_steps"] == step

        def finalize(self):
            self.finalized += 1

        def fail(self, error):
            self.failed.append(error)

    class Progress:
        def __init__(self, count):
            self._values = range(count)

        def __iter__(self):
            return iter(self._values)

        def set_description(self, *args, **kwargs):
            pass

        def write(self, *args, **kwargs):
            pass

    monkeypatch.setattr(experiment_module.jax, "jit", lambda fn, **kwargs: fn)
    monkeypatch.setattr(experiment_module.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(experiment_module.lox, "spool", lambda fn: fn)
    monkeypatch.setattr(
        experiment_module, "trange", lambda count, **kwargs: Progress(count)
    )
    logger = Logger()
    config = ExperimentConfig(
        seed=1,
        total_timesteps=8,
        num_epochs=2,
        num_envs=2,
        eval_every=0,
    )

    train_loop(Agent(), config, logger)

    assert [item["env_steps"] for item in logger.summaries] == [2, 4, 6, 8]
    assert logger.finalized == 1
    assert logger.failed == []
