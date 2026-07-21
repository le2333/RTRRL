from __future__ import annotations

import sys
from pathlib import Path
from types import MappingProxyType

import pytest
import yaml

from trainer_infra.materialize import materialize_run
from trainer_infra.execution import build_run_context
from trainer_infra.models import (
    DescriptorDefaults,
    EnvironmentSpec,
    ExecutionSpec,
    ExperimentDefaults,
    ExperimentIdentity,
    ExperimentSpec,
    FieldDescriptor,
    GroupSpec,
    HpoSpec,
    LoggingSpec,
    ObjectiveSpec,
    ResourcesSpec,
    ScriptCatalog,
    ScriptDescriptor,
    TrainingBudgetSpec,
)
from trainer_infra.resolve import resolve_experiment
from test_materialize import FakeTrial, make_group

REPOSITORY_ROOT = Path(__file__).parents[4]
MEMO_ROOT = REPOSITORY_ROOT / "memo"
sys.path.insert(0, str(MEMO_ROOT))
sys.path.insert(0, str(MEMO_ROOT / "experiments"))

from base.facility import FacilityInput  # noqa: E402


def _descriptor() -> ScriptDescriptor:
    return ScriptDescriptor(
        name="memo_stream_ac",
        argv=("python", "experiments/memo_stream_ac/run.py", "--config", "{config_path}"),
        sdk_protocol_version="1",
        defaults=DescriptorDefaults(
            environment=EnvironmentSpec(
                name="memory_chain",
                options={
                    "length": 8,
                    "max_episode_steps": 8,
                    "nested": {"observe": ["query"]},
                },
            ),
            training_budget=TrainingBudgetSpec(env_steps=24),
            logging=LoggingSpec(
                aim_every_env_steps=6,
                rerun_every_episodes=2,
            ),
        ),
        objective=ObjectiveSpec(
            metric="eval/rewards",
            direction="maximize",
            reduction="last",
        ),
        environments=("memory_chain",),
        fields={
            "agent_type": FieldDescriptor(
                path="algorithm.agent_type",
                type="str",
                default="rtu_rtrl",
                choices=("rtu_rtrl",),
            ),
            "hidden_dim": FieldDescriptor(
                path="network.hidden_dim",
                type="int",
                default=32,
                searchable=True,
                default_search={"values": [32, 64]},
            ),
            "seed": FieldDescriptor(
                path="runtime.seed",
                type="int",
                default=7,
            ),
        },
    )


def _spec(image: str) -> ExperimentSpec:
    return ExperimentSpec(
        experiment=ExperimentIdentity(name="facility-contract"),
        defaults=ExperimentDefaults(
            image=image,
            resources=ResourcesSpec(profile="c7am"),
            hpo=HpoSpec(total_trials=1, configs_per_batch=1),
            execution=ExecutionSpec(runs_per_job=1),
        ),
        groups={
            "stream": GroupSpec(
                script="memo_stream_ac",
                parameters={"hidden_dim": {"values": [64]}},
            )
        },
    )


def test_descriptor_resolve_materialize_loads_exact_nested_concrete_yaml(tmp_path):
    image = "repo/memo@sha256:" + "a" * 64
    resolved = resolve_experiment(
        _spec(image),
        {
            image: ScriptCatalog(
                protocol_version="1",
                scripts={"memo_stream_ac": _descriptor()},
            )
        },
    ).groups[0]

    concrete = materialize_run(
        resolved,
        FakeTrial(3),
        {},
        run_number=1,
    )
    path = tmp_path / "concrete.yaml"
    path.write_text(concrete.config_yaml)
    payload = yaml.safe_load(concrete.config_yaml)
    facility = FacilityInput.load(path)

    assert payload["protocol_version"] == "1"
    assert payload["environment"]["options"]["nested"] == {
        "observe": ["query"]
    }
    assert payload["parameters"] == {
        "algorithm": {"agent_type": "rtu_rtrl"},
        "network": {"hidden_dim": 64},
        "runtime": {"seed": 7},
    }
    assert facility.parameters["network"]["hidden_dim"] == 64
    assert concrete.final_parameters == {
        "agent_type": "rtu_rtrl",
        "hidden_dim": 64,
        "seed": 7,
    }
    context = build_run_context(
        "exp-1",
        resolved.name,
        concrete,
        tmp_path / "artifacts",
    )
    assert context.seed == 7
    assert context.final_parameters["seed"] == 7


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
