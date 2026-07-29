"""Every experiment names exactly the parameters its entry declares.

An omitted parameter is not an error anywhere. The control plane resolves a run's
space as ``dict(entry.space) | dict(overrides)``, so a key the experiment file
leaves out keeps the whole domain the entry declared and the sampler is free to
pick from it. Nothing fails, nothing warns, and the run is not the run the file
describes: a grid of one point becomes a grid of two with one of them drawn, five
seeds stop being five seeds, and a search gains a dimension the other trials in
the study never varied.

That makes adding a parameter to an entry a change to every experiment written
against it, which is not what adding a parameter looks like. This turns the
silence into a failure here, before an image is built.

The declared side is checked against the code by ``test_entries.py``, so between
the two an entry cannot declare a knob it does not read, read one it has not
declared, or grow one without every experiment saying what to hold it at.

Naming a parameter is not pinning it: a search names ``actor_lr`` with a range
and that is the point of a search. Only the reproduction is held to single
values, by ``test_hopper_reproduction.py``, because only it exists to land on one
recorded number.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from runner.catalog import discover

EXPERIMENTS = sorted(
    (Path(__file__).resolve().parents[2] / "experiments").glob("*.yaml")
)
ENTRIES = discover()


@pytest.fixture(
    params=EXPERIMENTS,
    ids=[path.stem for path in EXPERIMENTS],
)
def experiment(request) -> dict[str, Any]:
    return yaml.safe_load(request.param.read_text(encoding="utf-8"))


@pytest.fixture
def declared(experiment) -> set[str]:
    return set(ENTRIES[experiment["entry"]].SPACE)


@pytest.fixture
def named(experiment) -> set[str]:
    return set(experiment.get("space") or {})


def test_there_are_experiments_to_check():
    """The glob finding nothing would pass every other test in this file."""

    assert EXPERIMENTS


def test_the_entry_it_names_is_one_the_image_carries(experiment):
    assert experiment["entry"] in ENTRIES, sorted(ENTRIES)


def test_it_says_what_to_hold_every_declared_parameter_at(declared, named):
    assert not declared - named


def test_it_asks_for_nothing_its_entry_does_not_declare(declared, named):
    """The control plane refuses this one, but it refuses it after submission."""

    assert not named - declared
