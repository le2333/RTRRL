from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path
from types import MappingProxyType

import pytest
import yaml

from trainer_infra.execution import build_run_context
from trainer_infra.image_catalog import load_catalog_index
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
sys.path.insert(0, str(MOCK_TRAINER_ROOT / "src"))

from brax_ppo_acceptance.config import AcceptanceConfig  # noqa: E402


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
    acceptance = AcceptanceConfig.load(path, environ={})

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
    assert acceptance.learning_rate == 0.0004
    assert acceptance.num_timesteps == 128
    assert acceptance.failure_mode == "none"
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
