"""The four R3.1 conditions reach the graph from the documents that name them.

The algebra of each condition is driven in ``tests/unit/algorithms/rtrrl``. What
is here is the half that needs the whole chain: that a catalog built from this
image declares the choice, that the control plane resolves each file's pin
through it, and that what comes out the far side is the readout that file named
and a graph that steps. A condition nobody can select from a run document is not
an experiment.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import pytest
import yaml
from trainer_infra.experiment import ExperimentRunner

from deployment.catalog import build_catalog
from entries._contract import RunSpec
from memorax.algorithms.rtrrl_aaai import (
    ELIGIBILITY_SOURCES,
    METRICS,
    PARAMETERS,
    RTRRL,
)
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble
from memorax.parameters import KIND, flatten
from tests.support.builders import graph_of

pytestmark = pytest.mark.integration

EXPERIMENTS = Path(__file__).resolve().parents[4] / "experiments"
CELLS = {
    "trace": "baseline",
    "gradient": "no-trace",
    "direction": "direction-only",
    "gain": "gain-only",
}
# The files carry `image: TBD` until the rebuild that declares this component is
# published, and the control plane refuses a reference with no digest before it
# reads anything else. Nothing is pulled here -- the graph is assembled in
# process -- so a pinned placeholder is what lets the rest of the file be read.
PINNED = f"rtrrl@sha256:{'0' * 64}"


def dotted(section: Mapping, prefix: str = "") -> dict[str, Any]:
    found: dict[str, Any] = {}
    for name, node in section.items():
        key = f"{prefix}{name}"
        if isinstance(node, Mapping):
            found |= dotted(node, f"{key}.")
        else:
            found[key] = node
    return found


def document(cell: str) -> dict:
    text = (EXPERIMENTS / f"rtrrl issue41 {CELLS[cell]}.yaml").read_text(
        encoding="utf-8"
    )
    return {**yaml.safe_load(text), "image": PINNED}


@pytest.mark.parametrize("cell", tuple(ELIGIBILITY_SOURCES))
def test_each_condition_is_selected_by_the_document_that_names_it(cell, tmp_path):
    experiment = document(cell)
    pins = dotted(experiment["space"])
    declared = flatten(PARAMETERS)

    assert not sorted(set(pins) - set(declared)), "the entry declares no such name"
    for name, values in pins.items():
        if name.endswith(f".{KIND}"):
            assert len(values) == 1, f"{name} is a choice and is not searched"
    assert pins[f"eligibility.{KIND}"] == [cell]
    assert experiment["score"]["metric"] in METRICS

    configuration = ExperimentRunner(
        experiment=experiment,
        catalog=build_catalog().model_dump(mode="json"),
        database=tmp_path / "study.db",
        launch_id="20260817-000000",
    ).next_round()[0]
    config = RunSpec.model_validate(configuration)
    environment = config.algorithm.environment
    built = assemble(
        RTRRL,
        BuildRequest(
            parameters=config.algorithm.parameters,
            environment=EnvironmentSpec(
                id=environment.id,
                backend=environment.backend,
                observed=environment.observed,
                episode_length=environment.episode_length,
            ),
            num_envs=config.algorithm.num_envs,
        ),
    )

    readout = graph_of(built).core.eligibility
    assert (readout.direction, readout.magnitude) == ELIGIBILITY_SOURCES[cell]

    state = built.program.init(jax.random.key(config.training.seed))
    _, observations = built.program.train(jax.random.key(1), state, 2)

    assert observations.interaction.reward.shape == (2, 1)


def test_the_four_documents_differ_in_the_condition_and_nothing_else():
    """Otherwise the factorial measures whatever else moved with it."""

    documents = {cell: document(cell) for cell in ELIGIBILITY_SOURCES}
    for cell, experiment in documents.items():
        space = dict(experiment["space"])
        del space["eligibility"]
        baseline = dict(documents["trace"]["space"])
        del baseline["eligibility"]

        assert space == baseline, f"{cell} differs from the baseline elsewhere"
        assert experiment["environment"] == documents["trace"]["environment"]
        assert experiment["training"] == documents["trace"]["training"]
        assert experiment["evaluation"] == documents["trace"]["evaluation"]
