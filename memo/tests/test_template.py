"""The target experiment file names what the entry declares, and it runs.

Two failures this catches. A name in ``space`` the entry does not declare is a
knob that turns nothing, and the sampler never fills it. A name the entry reads
but the file spells differently arrives as a ``KeyError`` once a job has
started.

Resolution itself -- how a pin becomes a distribution, and what the sampler does
with it -- belongs to the control plane and is tested there. What is here is the
half that needs the entry: that the two agree on the names, and that the entry
builds and steps on a manifest honouring the file's pins.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax
import pytest
import yaml
from training_sdk.parameters import expand, flatten

from entries import stream_ac

TEMPLATE = (
    Path(__file__).resolve().parents[2] / "experiments" / "streamac template.yaml"
)


@pytest.fixture(scope="module")
def experiment() -> dict:
    return yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def declared() -> dict:
    return flatten(expand(stream_ac.PARAMETERS))


def test_every_name_the_file_pins_is_one_the_entry_declares(experiment, declared):
    unknown = sorted(set(experiment["space"]) - set(declared))

    assert not unknown, f"the entry declares no {unknown}"


def test_every_structure_is_pinned_to_exactly_one_branch(experiment):
    structures = {
        name: node.branches
        for name, node in stream_ac.PARAMETERS.items()
        if hasattr(node, "branches")
    }
    for name, pinned in experiment["space"].items():
        if name not in structures:
            continue
        assert (
            isinstance(pinned, list) and len(pinned) == 1
        ), f"{name} is a structure and is not searched; pin it to one branch"
        assert pinned[0] in structures[name], (
            f"{name} names {pinned[0]!r}, not one of "
            f"{', '.join(sorted(structures[name]))}"
        )


def test_the_score_reads_a_metric_this_entry_reports(experiment):
    assert experiment["score"]["metric"] in stream_ac.METRICS


def manifest(experiment, declared) -> dict:
    """What the sampler would hand the entry, with the file's pins applied."""

    return {**declared, **{name: one[0] for name, one in experiment["space"].items()}}


def test_the_entry_builds_and_steps_on_that_manifest(experiment, declared):
    section = experiment["environment"]
    agent = stream_ac.build(
        manifest(experiment, declared),
        SimpleNamespace(**section),
        SimpleNamespace(num_envs=2),
    )
    state = agent.init(jax.random.key(int(section["seed"])))
    _, metrics = agent.train(jax.random.key(1), state, 4)

    assert metrics.interaction.reward.shape == (2, 2)
