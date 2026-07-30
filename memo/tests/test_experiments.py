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

The experiments directory is not memo's. It holds every experiment this
repository can launch, and since the Hopper comparison gained an arm built from
the authors' forked implementation, some of those name entries that live in
another image -- one this project cannot import, because it pins a jax two major
versions behind ours and installing both at once is what having two images
avoids. Skipping those files would leave the newest arm unchecked by the test
written to catch exactly the mistake it is most exposed to. So a foreign image's
entries are read from the catalog it publishes, which is the same artefact the
control plane validates an experiment against, and everything below is asserted
of every file either way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from runner.catalog import discover

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = sorted((ROOT / "experiments").glob("*.yaml"))
# Images built elsewhere in this repository whose entries an experiment file may
# name. The acceptance trainer under rtrrl/infra is deliberately absent: it has a
# catalog too, but its experiment lives beside it rather than here.
FOREIGN_CATALOGS = (ROOT / "rtrrl" / "catalog.json",)


def declared_spaces() -> dict[str, set[str]]:
    """What each entry accepts, whichever image it is carried by."""

    spaces = {name: set(module.SPACE) for name, module in discover().items()}
    for path in FOREIGN_CATALOGS:
        catalog = json.loads(path.read_text(encoding="utf-8"))
        for name, entry in catalog["entries"].items():
            if name in spaces:
                raise ValueError(
                    f"two images both carry an entry called {name!r}; an "
                    "experiment naming it would be ambiguous about what runs it"
                )
            spaces[name] = set(entry["space"])
    return spaces


SPACES = declared_spaces()


@pytest.fixture(
    params=EXPERIMENTS,
    ids=[path.stem for path in EXPERIMENTS],
)
def experiment(request) -> dict[str, Any]:
    return yaml.safe_load(request.param.read_text(encoding="utf-8"))


@pytest.fixture
def declared(experiment) -> set[str]:
    return SPACES[experiment["entry"]]


@pytest.fixture
def named(experiment) -> set[str]:
    return set(experiment.get("space") or {})


def test_there_are_experiments_to_check():
    """The glob finding nothing would pass every other test in this file."""

    assert EXPERIMENTS


def test_the_entry_it_names_is_one_some_image_carries(experiment):
    assert experiment["entry"] in SPACES, sorted(SPACES)


def test_it_says_what_to_hold_every_declared_parameter_at(declared, named):
    assert not declared - named


def test_it_asks_for_nothing_its_entry_does_not_declare(declared, named):
    """The control plane refuses this one, but it refuses it after submission."""

    assert not named - declared
