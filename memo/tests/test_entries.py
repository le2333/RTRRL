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
from typing import Any

import pytest
from training_sdk.episode import Episode

from entries import rtrrl
from runner.catalog import discover

# The few settings a run cannot be given arbitrarily: it has to end quickly,
# on a task that exists, in whole epochs.
BUDGET: dict[str, Any] = {
    "environment": "brax::hopper",
    "env_mode": "P",
    "env_backend": "generalized",
    # The cell the recorded runs used. Taking whichever one happens to sort
    # first would test a pairing nobody runs.
    "backbone": "rtu",
    "hidden_dim": 2,
    "feature_dim": 3,
    "num_envs": 2,
    "total_steps": 8,
    "epoch_steps": 4,
    "eval_steps": 4,
    "seed": 0,
}


def smallest(space: Mapping[str, Any]) -> dict[str, Any]:
    """One value per declared parameter, chosen for cheapness not for sense."""

    values = {
        name: spec[0] if isinstance(spec, list) else spec["low"]
        for name, spec in space.items()
    }
    return values | {name: BUDGET[name] for name in BUDGET if name in space}


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
    collector, params = Collector(), Watched(smallest(entry.SPACE))
    entry.run(collector, params)
    return entry, collector, params


def test_the_entry_reads_nothing_it_has_not_declared(trained):
    entry, _, params = trained
    assert not params.read - set(entry.SPACE)


def test_the_entry_declares_nothing_it_does_not_read(trained):
    entry, _, params = trained
    assert not set(entry.SPACE) - params.read


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


def test_the_reserved_budget_parameter_is_declared(trained):
    entry, _, _ = trained
    # The control plane refuses an entry without it, since a run with no
    # budget has nothing to stop it.
    assert "total_steps" in entry.SPACE


def test_rtrrl_entry_can_reproduce_the_papers_bounded_actor():
    """The paper bounds its actor before clipping its environment action."""

    params = smallest(rtrrl.SPACE) | BUDGET | {"bound_actor": True, "act_clip": 1.0}
    agent = rtrrl.build(params)
    assert agent.actor_head.bound is True
    assert agent.cfg.act_clip == 1.0
