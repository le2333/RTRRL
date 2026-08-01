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
# What the control plane injects rather than samples. An experiment naming one
# of these under `space` is asking the sampler for a value the manifest already
# carries, and the two would not have to agree.
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


def test_it_asks_for_nothing_its_entry_does_not_declare(declared, named):
    assert not named - declared


def test_it_has_training_and_evaluation_sections(experiment):
    assert "budget" not in experiment
    assert experiment["environment"]["seed"] >= 0
    assert experiment["training"]["total_steps"] > 0
    assert experiment["training"]["num_envs"] > 0
    assert experiment["evaluation"]["num_envs"] > 0


def test_it_leaves_the_injected_fields_out_of_its_space(named):
    taken = RESERVED & named
    assert not taken, f"space still names {sorted(taken)}"


def test_its_environment_names_no_training_stream_count(experiment):
    """`num_envs` describes the run, not the task, and now lives beside the
    budget it has to divide into."""

    assert "num_envs" not in experiment["environment"]
