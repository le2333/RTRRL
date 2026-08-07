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

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import jax
import pytest
import yaml

from entries import stream_ac
from memorax.parameters import KIND, expand, flatten

TEMPLATE = (
    Path(__file__).resolve().parents[2] / "experiments" / "streamac template.yaml"
)


def dotted(section: Mapping, prefix: str = "") -> dict[str, object]:
    """A pinned space as leaf paths. The file nests; a name is the path to it."""

    found: dict[str, object] = {}
    for name, node in section.items():
        key = f"{prefix}{name}"
        if isinstance(node, Mapping):
            found |= dotted(node, f"{key}.")
        else:
            found[key] = node
    return found


@pytest.fixture(scope="module")
def experiment() -> dict:
    return yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pins(experiment) -> dict:
    return dotted(experiment["space"])


@pytest.fixture(scope="module")
def declared() -> dict:
    return flatten(stream_ac.PARAMETERS)


def test_every_name_the_file_pins_is_one_the_entry_declares(pins, declared):
    unknown = sorted(set(pins) - set(declared))

    assert not unknown, f"the entry declares no {unknown}"


def test_every_choice_of_component_is_pinned_to_exactly_one_branch(pins, declared):
    for name, pinned in pins.items():
        if not name.endswith(f".{KIND}"):
            continue
        assert (
            isinstance(pinned, list) and len(pinned) == 1
        ), f"{name} chooses a component and is not searched; pin it to one"
        assert pinned[0] in declared[name].valid.values, (
            f"{name} names {pinned[0]!r}, not one of "
            f"{', '.join(sorted(map(str, declared[name].valid.values)))}"
        )


def test_the_score_reads_a_metric_this_entry_reports(experiment):
    assert experiment["score"]["metric"] in stream_ac.METRICS


def manifest(pins) -> dict:
    """What the sampler would hand the entry, with the file's pins applied."""

    return expand(stream_ac.PARAMETERS, {name: one[0] for name, one in pins.items()})


def test_the_manifest_carries_only_the_branches_the_file_chose(pins):
    """Not every branch filled in with something, which is the older mistake."""

    values = manifest(pins)

    assert values[f"backbone.{KIND}"] == "rtu"
    assert not [name for name in values if name.startswith("backbone.mlp.")]
    assert len(values) < len(flatten(stream_ac.PARAMETERS))


def test_the_entry_builds_and_steps_on_that_manifest(experiment, pins):
    section = experiment["environment"]
    agent = stream_ac.build(
        manifest(pins),
        SimpleNamespace(**section),
        SimpleNamespace(num_envs=2),
    )
    state = agent.init(jax.random.key(int(section["seed"])))
    _, metrics = agent.train(jax.random.key(1), state, 4)

    assert metrics.interaction.reward.shape == (2, 2)
