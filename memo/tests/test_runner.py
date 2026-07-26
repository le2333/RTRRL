"""The runner reaches the sinks without knowing which algorithm it drove."""

from __future__ import annotations

import numpy as np
import pytest
from training_sdk.contract import Catalog

from runner.catalog import build_catalog, source_hash
from runner.episodes import complete_episodes
from runner.main import run
from runner.metrics import scalar_metrics
from runner.registry import EPISODE_METRICS, TOPOLOGIES, topology


class Recorder:
    """A reporter that keeps what it was told instead of shipping it."""

    def __init__(self) -> None:
        self.reports: list[tuple[int, dict[str, float]]] = []
        self.episodes: list = []

    def report(self, step, metrics):
        self.reports.append((step, dict(metrics)))

    def log_episode(self, episode):
        self.episodes.append(episode)


def parameters(**overrides):
    params = {
        "environment": "gymnax::Pendulum-v1",
        "num_envs": 2,
        "total_steps": 16,
        "epoch_steps": 8,
        "eval_steps": 4,
        "seed": 0,
        "hidden_dim": 4,
        "feature_dim": 3,
    }
    params.update(overrides)
    return params


@pytest.mark.parametrize(
    "entry, extra",
    [
        ("rtrrl", {"backbone": "lru"}),
        ("rtrrl", {"backbone": "rtu", "update_rule": "obgd"}),
        (
            "stream_ac_rtrl",
            {
                "backbone": "rtu",
                "gamma": 0.9,
                "trace_lambda": 0.8,
                "actor_lr": 1e-3,
                "critic_lr": 1e-3,
            },
        ),
    ],
)
def test_an_epoch_reaches_the_reporter(entry, extra):
    recorder = Recorder()
    run(recorder, entry, parameters(**extra))

    assert [step for step, _ in recorder.reports] == [8, 8, 16, 16]
    for _, metrics in recorder.reports:
        assert metrics, "an epoch reported nothing at all"
        assert all(np.isfinite(value) for value in metrics.values())
    assert any(name.startswith("train/") for _, m in recorder.reports for name in m)


def test_evaluation_reports_the_metric_a_score_may_name():
    recorder = Recorder()
    run(recorder, "rtrrl", parameters(eval_steps=400, total_steps=8, epoch_steps=8))

    reported = {name for _, metrics in recorder.reports for name in metrics}
    assert set(EPISODE_METRICS) <= reported
    assert recorder.episodes, "a 400-step rollout finished no episode"


def test_every_episode_carries_one_more_observation_than_action():
    recorder = Recorder()
    run(recorder, "rtrrl", parameters(eval_steps=400, total_steps=8, epoch_steps=8))

    for episode in recorder.episodes:
        assert len(episode.observations) == len(episode.actions) + 1
        assert len(episode.rewards) == len(episode.actions)
        assert episode.terminals[-1] is True
        assert not any(episode.terminals[:-1])


def test_a_misspelled_parameter_is_refused_rather_than_ignored():
    with pytest.raises(ValueError, match="no parameter named gama"):
        topology("rtrrl").build(parameters(gama=0.9))


def test_the_catalog_only_promises_what_a_topology_accepts():
    catalog = Catalog.model_validate(build_catalog().model_dump(mode="json"))

    assert set(catalog.entries) == set(TOPOLOGIES)
    for name, entry in catalog.entries.items():
        assert entry.command == ("python", "-m", "runner.main")
        assert set(entry.space) == set(TOPOLOGIES[name].space)
        assert "total_steps" in entry.space
        assert entry.metrics == EPISODE_METRICS


def test_the_source_hash_ignores_bytecode(tmp_path):
    root = tmp_path / "pkg"
    (root / "__pycache__").mkdir(parents=True)
    (root / "kernel.py").write_text("x = 1\n")
    before = source_hash((root,))
    (root / "__pycache__" / "kernel.pyc").write_bytes(b"\x00compiled")
    assert source_hash((root,)) == before
    (root / "kernel.py").write_text("x = 2\n")
    assert source_hash((root,)) != before


class Summary:
    def __init__(self, before, after, action, reward, done):
        self.observation = before
        self.next_observation = after
        self.action = action
        self.reward = reward
        self.done = done


def test_a_partial_episode_at_either_end_is_left_out():
    #                   env 0: done at 1 and 4     env 1: never done
    done = np.array([[0, 0], [1, 0], [0, 0], [0, 0], [1, 0]], dtype=bool)
    steps, envs = done.shape
    summary = Summary(
        before=np.arange(steps * envs, dtype=float).reshape(steps, envs, 1),
        after=np.arange(steps * envs, dtype=float).reshape(steps, envs, 1) + 100,
        action=np.zeros((steps, envs, 1)),
        reward=np.ones((steps, envs)),
        done=done,
    )
    episodes = list(
        complete_episodes(summary, phase="eval", start_env_steps=0, num_envs=envs)
    )

    # Steps 2 and 3 of env 0 run past the end without terminating, and env 1
    # never terminates at all; neither is a return anyone can report.
    assert [len(episode.actions) for episode in episodes] == [2, 3]
    assert [episode.number for episode in episodes] == [1, 2]
    assert episodes[0].observations[-1] == [102.0]


def test_a_metric_is_a_number_per_step_and_nothing_wider():
    metrics = {
        "loss": np.arange(4.0),
        "trajectory": np.zeros((4, 2, 6)),
        "once": np.float32(3.0),
        "label": "adam",
    }
    assert scalar_metrics(metrics, steps=4) == {"loss": 1.5}
