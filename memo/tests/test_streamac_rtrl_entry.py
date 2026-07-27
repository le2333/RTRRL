"""The entry's declared space and the entry's code say the same thing.

An experiment file is written against ``SPACE`` and a sampler fills exactly the
names in it, so the two failures worth catching are a parameter the code reads
without declaring, which reaches the kernel as a ``KeyError`` after the job has
started and the money is spent, and a parameter declared without being read,
which is a knob an experiment can turn to no effect. Running the entry with a
mapping that records every lookup catches both at once.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from training_sdk.episode import Episode

from entries.streamac_rtrl import (
    METRICS,
    SPACE,
    TRAINING_METRICS,
    evaluation_report,
    run,
)

TINY: dict[str, Any] = {
    "environment": "brax::hopper",
    "env_mode": "P",
    "env_backend": "generalized",
    "backbone": "rtu",
    "hidden_dim": 2,
    "feature_dim": 3,
    "meta_rl": True,
    "normalize_observation": True,
    "normalize_reward": True,
    "num_envs": 2,
    "total_steps": 8,
    "epoch_steps": 4,
    "eval_steps": 4,
    "seed": 0,
    "gamma": 0.99,
    "trace_lambda": 0.8,
    "actor_lr": 1.0,
    "critic_lr": 1.0,
    "actor_kappa": 3.0,
    "critic_kappa": 2.0,
    "entropy_coefficient": 0.01,
    "adaptive": False,
    "beta2": 0.999,
    "eps": 1e-8,
}


class Watched(Mapping):
    """A parameter mapping that remembers which names were looked up."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        self._values = dict(values)
        self.read: set[str] = set()

    def __getitem__(self, name: str) -> Any:
        self.read.add(name)
        return self._values[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


class Collector:
    """Stands in for the reporter and keeps what the entry sent it."""

    def __init__(self) -> None:
        self.reports: list[tuple[int, dict[str, float]]] = []
        self.episodes: list[Episode] = []

    def report(self, step: int, metrics: Mapping[str, float]) -> None:
        self.reports.append((step, dict(metrics)))

    def log_episode(self, episode: Episode) -> None:
        self.episodes.append(episode)


@pytest.fixture(scope="module")
def trained() -> tuple[Collector, Watched]:
    collector, params = Collector(), Watched(TINY)
    run(collector, params)
    return collector, params


def test_the_entry_reads_nothing_it_has_not_declared(trained):
    _, params = trained
    assert not params.read - set(SPACE)


def test_the_entry_declares_nothing_it_does_not_read(trained):
    _, params = trained
    assert not set(SPACE) - params.read


def test_it_reports_once_per_epoch_at_the_step_it_has_reached(trained):
    collector, _ = trained
    steps = sorted({step for step, _ in collector.reports})
    assert steps == [4, 8]


def test_every_training_scalar_it_names_arrives(trained):
    collector, _ = trained
    sent = {name for _, report in collector.reports for name in report}
    assert {f"train/{name}" for name in TRAINING_METRICS} <= sent


def test_a_budget_that_leaves_a_ragged_epoch_is_refused():
    for name, value in (("total_steps", 9), ("epoch_steps", 3)):
        with pytest.raises(ValueError, match=name):
            run(Collector(), {**TINY, name: value})


def rollout(dones: list[bool]):
    """An evaluation summary of one stream, terminating where told to."""

    steps = len(dones)
    column = np.arange(steps, dtype=np.float32).reshape(steps, 1)
    return SimpleNamespace(
        observation=column.reshape(steps, 1, 1),
        next_observation=column.reshape(steps, 1, 1) + 1,
        action=column.reshape(steps, 1, 1),
        reward=column,
        done=np.array(dones).reshape(steps, 1),
    )


def test_a_whole_episode_becomes_a_score_and_a_trajectory():
    collector = Collector()
    number = evaluation_report(
        collector,
        rollout([False, True, False]),
        done=10,
        num_envs=1,
        number=1,
    )

    assert [episode.number for episode in collector.episodes] == [1]
    assert number == 2
    ((step, report),) = collector.reports
    assert step == 10
    # Rewards 0 and 1 are inside the episode; the trailing step is not.
    assert report["eval/episode_return"] == 1.0
    assert report["eval/episode_length"] == 2.0
    assert not set(METRICS) - set(report)


def test_a_rollout_without_a_terminal_reports_no_score():
    """A mean over no episode is zero, and zero is a plausible score. The
    entry must leave the metric out rather than send one that reads as a
    result the run never produced.
    """

    collector = Collector()
    evaluation_report(collector, rollout([False] * 3), done=10, num_envs=1, number=1)

    ((_, report),) = collector.reports
    assert not collector.episodes
    assert set(METRICS) - set(report) == set(METRICS)
