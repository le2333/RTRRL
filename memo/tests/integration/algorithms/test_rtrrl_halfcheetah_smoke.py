"""The shipped RTRRL acceptance configuration closes over HalfCheetah."""

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
from memorax.algorithms.rtrrl_aaai import METRICS, PARAMETERS, RTRRL
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble
from memorax.parameters import KIND, flatten

pytestmark = pytest.mark.integration

TEMPLATE = (
    Path(__file__).resolve().parents[4] / "experiments" / "rtrrl halfcheetah smoke.yaml"
)


def dotted(section: Mapping, prefix: str = "") -> dict[str, Any]:
    found: dict[str, Any] = {}
    for name, node in section.items():
        key = f"{prefix}{name}"
        if isinstance(node, Mapping):
            found |= dotted(node, f"{key}.")
        else:
            found[key] = node
    return found


def test_rtrrl_template_resolves_and_runs(tmp_path: Path) -> None:
    experiment = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    pins = dotted(experiment["space"])
    declared = flatten(PARAMETERS)
    assert not sorted(set(pins) - set(declared))
    for name, values in pins.items():
        if name.endswith(f".{KIND}"):
            assert len(values) == 1
    assert experiment["score"]["metric"] in METRICS

    configuration = ExperimentRunner(
        experiment=experiment,
        catalog=build_catalog().model_dump(mode="json"),
        database=tmp_path / "study.db",
        launch_id="20260813-000000",
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
    state = built.program.init(jax.random.key(config.training.seed))
    _, observations = built.program.train(jax.random.key(1), state, 2)

    assert observations.interaction.reward.shape == (2, 1)
