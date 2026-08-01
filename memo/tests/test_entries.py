"""Every entry's declared space and its code say the same thing.

An experiment is written against ``SPACE`` and the sampler fills exactly the
names in it, so two failures are worth catching. A parameter the code reads
without declaring arrives as a ``KeyError`` once the job has started and the
money is spent. A parameter declared without being read is a knob an experiment
can turn to no effect. Running each entry with a mapping that records every
lookup catches both, for whatever entries exist, without this file naming them.

The settings come from the space itself rather than from a table here, so an
entry that grows a parameter is covered the moment it declares one.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import SimpleNamespace
from typing import Any

import pytest
from training_sdk.episode import Episode
from training_sdk.parameters import expand, flatten

from entries import rtrrl
from runner.catalog import discover

PARAM_OVERRIDES: dict[str, Any] = {
    # The cell the recorded runs used. Taking whichever one happens to sort
    # first would test a pairing nobody runs.
    "backbone": "rtu",
    "hidden_dim": 2,
    "feature_dim": 3,
}
ENVIRONMENT = SimpleNamespace(
    id="brax::hopper",
    backend="generalized",
    seed=0,
    observed=(0, 1, 2, 3, 4),
)
TRAINING = SimpleNamespace(num_envs=2, total_steps=8, epoch_steps=4)
EVALUATION = SimpleNamespace(steps=4, num_envs=2)


def manifest(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """What a launch would send, taken through the loader rather than by hand."""

    declared = flatten(parameters)
    pinned = {
        name: value for name, value in PARAM_OVERRIDES.items() if name in declared
    }
    return expand(parameters, pinned)


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


ENTRIES = discover()


@pytest.fixture(scope="module", params=sorted(ENTRIES), ids=sorted(ENTRIES))
def trained(request) -> tuple[Any, Collector, Watched]:
    entry = ENTRIES[request.param]
    collector, params = Collector(), Watched(manifest(entry.PARAMETERS))
    config = SimpleNamespace(
        params=params,
        environment=ENVIRONMENT,
        training=TRAINING,
        evaluation=EVALUATION,
    )
    entry.run(collector, config)
    return entry, collector, params


def test_the_entry_reads_nothing_it_has_not_declared(trained):
    entry, _, params = trained
    assert not params.read - set(flatten(entry.PARAMETERS))


def test_the_entry_declares_nothing_it_does_not_read(trained):
    entry, _, params = trained
    assert not set(flatten(entry.PARAMETERS)) - params.read


def test_it_reports_once_per_epoch_at_the_step_it_has_reached(trained):
    _, collector, _ = trained
    assert sorted({step for step, _ in collector.reports}) == [4, 8]


def test_the_scalars_it_names_are_scalars_it_sends(trained):
    entry, collector, _ = trained
    sent = {name for _, report in collector.reports for name in report}
    named = {f"train/{name}" for name in entry.TRAINING_METRICS}
    # Only what this configuration produces: a metric belonging to an update
    # rule the run did not use is declared but absent, which is not a fault.
    assert named & sent, "not one declared training scalar arrived"
    assert not sent - named - {"eval/reward", *entry.METRICS}


def test_rtrrl_entry_can_reproduce_the_papers_bounded_actor():
    """The paper bounds its actor before clipping its environment action."""

    params = manifest(rtrrl.PARAMETERS) | {"bound_actor": True, "act_clip": 1.0}
    agent = rtrrl.build(params, ENVIRONMENT, TRAINING)
    assert agent.actor_head.bound is True
    assert agent.cfg.act_clip == 1.0


RESERVED = frozenset(
    {
        "environment",
        "env_mode",
        "env_backend",
        "observed",
        "seed",
        "num_envs",
        "total_steps",
        "epoch_steps",
        "eval_steps",
        "chunk_steps",
        "early_stop_patience",
        "eval_envs",
    }
)


def test_no_entry_declares_the_environment_or_the_budget():
    for name, module in ENTRIES.items():
        taken = RESERVED & set(flatten(module.PARAMETERS))
        assert not taken, f"{name} still declares {sorted(taken)}"
