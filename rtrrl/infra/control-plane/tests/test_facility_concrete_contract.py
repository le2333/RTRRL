from __future__ import annotations

from dataclasses import replace
import json
import os
import sys
from pathlib import Path
import subprocess
from types import MappingProxyType

import pytest
import yaml

from trainer_infra.controller import _estimated_jobs
from trainer_infra.execution import build_run_context
from trainer_infra.image_catalog import load_catalog_index
from trainer_infra.loaders import load_experiment
from trainer_infra.materialize import materialize_run
from trainer_infra.models import (
    ExecutionSpec,
    ExperimentDefaults,
    ExperimentIdentity,
    ExperimentSpec,
    GroupSpec,
    HpoSpec,
    ResourcesSpec,
)
from trainer_infra.resolve import resolve_experiment
from test_materialize import FakeTrial, make_group

REPOSITORY_ROOT = Path(__file__).parents[4]
MOCK_TRAINER_ROOT = REPOSITORY_ROOT / "rtrrl" / "infra" / "mock-trainer"
SMOKE_EXPERIMENT = (
    REPOSITORY_ROOT
    / "rtrrl"
    / "infra"
    / "control-plane"
    / "examples"
    / "experiment-smoke.yaml"
)


def _load_acceptance(path: Path) -> dict[str, object]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(MOCK_TRAINER_ROOT / "src")
    result = subprocess.run(
        [
            str(MOCK_TRAINER_ROOT / ".venv" / "bin" / "python"),
            "-c",
            (
                "import json, sys;"
                "from brax_ppo_acceptance.config import AcceptanceConfig;"
                "config = AcceptanceConfig.load(sys.argv[1], environ={});"
                "print(json.dumps({"
                "'environment_name': config.environment_name,"
                "'learning_rate': config.learning_rate,"
                "'num_timesteps': config.num_timesteps,"
                "'failure_mode': config.failure_mode"
                "}, sort_keys=True))"
            ),
            str(path),
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _spec(image: str) -> ExperimentSpec:
    return ExperimentSpec(
        experiment=ExperimentIdentity(
            name="facility-contract",
            metadata={"purpose": "acceptance-contract"},
        ),
        defaults=ExperimentDefaults(
            image=image,
            environment={
                "name": "inverted_pendulum",
                "options": {"backend": "generalized"},
            },
            training_budget={"env_steps": 128},
            logging={
                "aim_every_env_steps": 1,
                "rerun_every_episodes": 1,
            },
            resources=ResourcesSpec(profile="c7am"),
            hpo=HpoSpec(total_trials=1, configs_per_batch=1),
            execution=ExecutionSpec(runs_per_job=1),
        ),
        groups={
            "cpu": GroupSpec(
                script="brax_ppo_acceptance",
                parameters={"learning_rate": {"values": [0.0004]}},
            )
        },
    )


def test_descriptor_resolve_materialize_loads_exact_nested_concrete_yaml(tmp_path):
    image = "repo/acceptance@sha256:" + "a" * 64
    resolved = resolve_experiment(
        _spec(image),
        {image: load_catalog_index(MOCK_TRAINER_ROOT / "scripts" / "index.yaml")},
    ).groups[0]
    resolved = replace(resolved, study_key="exp-1:cpu")

    concrete = materialize_run(
        resolved,
        FakeTrial(3),
        {"learning_rate": 0.0004},
        run_number=1,
    )
    path = tmp_path / "concrete.yaml"
    path.write_text(concrete.config_yaml)
    payload = yaml.safe_load(concrete.config_yaml)
    acceptance = _load_acceptance(path)

    assert payload["protocol_version"] == "1"
    assert payload["environment"] == {
        "name": "inverted_pendulum",
        "options": {"backend": "generalized"},
    }
    assert payload["parameters"] == {
        "algorithm": {
            "episode_length": 32,
            "failure_mode": "none",
            "learning_rate": 0.0004,
            "num_envs": 4,
        },
        "runtime": {"seed": 0},
    }
    assert acceptance["learning_rate"] == 0.0004
    assert acceptance["num_timesteps"] == 128
    assert acceptance["failure_mode"] == "none"
    assert concrete.final_parameters == {
        "episode_length": 32,
        "failure_mode": "none",
        "learning_rate": 0.0004,
        "num_envs": 4,
        "seed": 0,
    }
    context = build_run_context(
        "facility",
        "exp-1",
        resolved.name,
        concrete,
        tmp_path / "artifacts",
    )
    assert context.seed == 0
    assert context.script == "brax_ppo_acceptance"
    assert context.metadata == {"purpose": "acceptance-contract"}
    assert context.final_parameters["learning_rate"] == 0.0004


def test_committed_smoke_resolves_real_acceptance_contract_and_materializes(
    tmp_path: Path,
) -> None:
    spec = load_experiment(SMOKE_EXPERIMENT)
    catalog = load_catalog_index(MOCK_TRAINER_ROOT / "scripts" / "index.yaml")
    resolved = resolve_experiment(
        spec,
        {
            spec.defaults.image: catalog,
            str(spec.groups["gpu"].image): catalog,
        },
    )

    assert [group.name for group in resolved.groups] == ["cpu", "gpu"]
    assert all(group.hpo.total_trials == 5 for group in resolved.groups)
    assert all(group.hpo.configs_per_batch == 2 for group in resolved.groups)
    assert all(group.execution.runs_per_job == 2 for group in resolved.groups)
    assert [_estimated_jobs(group) for group in resolved.groups] == [3, 3]
    assert all(group.script == "brax_ppo_acceptance" for group in resolved.groups)
    assert [group.resources.profile for group in resolved.groups] == ["c7am", "g6x"]
    assert [group.image for group in resolved.groups] == [
        (
            "007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl:"
            "infra-acceptance-brax-ppo-cpu-20260723"
        ),
        (
            "007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl:"
            "infra-acceptance-brax-ppo-gpu-20260723"
        ),
    ]
    assert all(group.environment.name == "inverted_pendulum" for group in resolved.groups)
    assert all(group.environment.options == {"backend": "generalized"} for group in resolved.groups)
    expected_values = (0.0002, 0.0003, 0.0004, 0.0005, 0.0006)
    assert all(
        group.searchable_parameters()["learning_rate"].values == expected_values
        for group in resolved.groups
    )

    group = replace(resolved.groups[0], study_key="smoke-1:cpu")
    concrete = materialize_run(
        group,
        FakeTrial(0),
        {"learning_rate": 0.0002},
        run_number=1,
    )
    path = tmp_path / "smoke-concrete.yaml"
    path.write_text(concrete.config_yaml, encoding="utf-8")
    acceptance = _load_acceptance(path)
    assert acceptance["environment_name"] == "inverted_pendulum"
    assert acceptance["learning_rate"] == 0.0002
    assert acceptance["num_timesteps"] == 128


def test_acceptance_contract_import_does_not_pollute_interpreter() -> None:
    assert str(MOCK_TRAINER_ROOT / "src") not in sys.path
    assert not any(
        name == "brax_ppo_acceptance" or name.startswith("brax_ppo_acceptance.")
        for name in sys.modules
    )


def test_materialize_rejects_parameter_path_prefix_conflicts():
    group = make_group()
    object.__setattr__(
        group,
        "parameter_paths",
        MappingProxyType({"seed": "agent", "topology": "agent.type"}),
    )

    with pytest.raises(ValueError, match="conflict"):
        materialize_run(
            group,
            FakeTrial(1),
            {"topology": "shared"},
            run_number=1,
        )
