from __future__ import annotations

import hashlib
import json
from types import MappingProxyType

import pytest
import yaml

from trainer_infra.identities import canonical_json, canonical_yaml, study_name
from trainer_infra.materialize import materialize_run
from trainer_infra.models import (
    DiscreteSearch,
    EnvironmentSpec,
    ExecutionSpec,
    HpoSpec,
    LoggingSpec,
    ObjectiveSpec,
    ResolvedGroup,
    ResolvedParameter,
    ResourcesSpec,
    TrainingBudgetSpec,
)


class FakeTrial:
    def __init__(self, number: int) -> None:
        self.number = number


def make_group(name: str = "shared") -> ResolvedGroup:
    return ResolvedGroup(
        name=name,
        study_key=f"experiment-123:{name}",
        image="repo/image@sha256:" + "a" * 64,
        script="rtrrl",
        argv=("python", "-m", "train"),
        sdk_protocol_version="1",
        objective=ObjectiveSpec(metric="reward", direction="maximize", reduction="last"),
        environment=EnvironmentSpec(
            env_name="hopper",
            backend="spring",
            observation_mode="P",
            max_episode_steps=1000,
        ),
        training_budget=TrainingBudgetSpec(env_steps=100),
        logging=LoggingSpec(aim_every_env_steps=10, rerun_every_episodes=2),
        resources=ResourcesSpec(profile="gpu"),
        hpo=HpoSpec(total_trials=10, configs_per_batch=2),
        execution=ExecutionSpec(runs_per_job=2),
        metadata=MappingProxyType({"labels": ["baseline"], "owner": {"name": "research"}}),
        parameters=MappingProxyType(
            {
                "seed": ResolvedParameter(fixed_value=7, search_domain=None),
                "topology": ResolvedParameter(
                    fixed_value=None,
                    search_domain=DiscreteSearch(("shared", "dual")),
                ),
            }
        ),
    )


def test_study_name_is_group_scoped() -> None:
    assert study_name("experiment-123", "shared") == "experiment-123:shared"
    assert study_name("experiment-123", "shared") != study_name("experiment-123", "dual")


def test_run_identity_defaults_to_trial_sequence_but_accepts_controller_sequence() -> None:
    group = make_group()

    default = materialize_run(group, FakeTrial(7), {"topology": "shared"})
    accepted = materialize_run(
        group,
        FakeTrial(19),
        {"topology": "dual"},
        run_number=3,
    )

    assert default.run_number == 8
    assert default.trial_number == 7
    assert default.run_name == "shared-rtrrl-0008"
    assert default.run_id == "experiment-123:shared:0008"
    assert accepted.run_number == 3
    assert accepted.trial_number == 19
    assert accepted.run_name == "shared-rtrrl-0003"
    assert accepted.run_id == "experiment-123:shared:0003"
    assert accepted.context["run_number"] == 3
    assert accepted.context["trial_number"] == 19
    assert accepted.context["run_id"] == accepted.run_id


def test_run_identity_is_independent_per_group() -> None:
    first = materialize_run(make_group("shared"), FakeTrial(0), {"topology": "shared"})
    second = materialize_run(make_group("dual"), FakeTrial(0), {"topology": "dual"})

    assert first.run_name == "shared-rtrrl-0001"
    assert second.run_name == "dual-rtrrl-0001"
    assert first.study_key != second.study_key
    assert first.run_id != second.run_id


def test_materialized_run_is_an_immutable_complete_snapshot() -> None:
    run = materialize_run(
        make_group(),
        FakeTrial(0),
        {"seed": 7, "topology": "dual"},
    )

    assert run.fixed_parameters == {"seed": 7}
    assert run.sampled_parameters == {"topology": "dual"}
    assert run.final_parameters == {"seed": 7, "topology": "dual"}
    assert run.argv == ("python", "-m", "train")
    assert run.image == "repo/image@sha256:" + "a" * 64
    assert yaml.safe_load(run.config_yaml)["parameters"] == {
        "seed": 7,
        "topology": "dual",
    }
    assert json.loads(run.run_json)["context"]["run_number"] == 1
    assert run.config_sha256 == hashlib.sha256(run.config_yaml.encode()).hexdigest()
    assert run.run_sha256 == hashlib.sha256(run.run_json.encode()).hexdigest()

    with pytest.raises(TypeError):
        run.final_parameters["seed"] = 9
    with pytest.raises(TypeError):
        run.metadata["labels"].append("changed")
    with pytest.raises(TypeError):
        run.context["run_number"] = 2
    with pytest.raises(Exception):
        run.run_number = 2


def test_materialize_rejects_fixed_overrides_and_missing_searchable_values() -> None:
    with pytest.raises(ValueError, match="fixed parameter 'seed'"):
        materialize_run(
            make_group(),
            FakeTrial(0),
            {"seed": 9, "topology": "dual"},
        )
    with pytest.raises(ValueError, match="missing searchable parameter 'topology'"):
        materialize_run(make_group(), FakeTrial(0), {"seed": 7})


def test_canonical_serialization_and_hashes_ignore_mapping_input_order() -> None:
    left = {"b": {"z": 1, "a": 2}, "a": [3, 4]}
    right = {"a": [3, 4], "b": {"a": 2, "z": 1}}

    assert canonical_json(left) == canonical_json(right)
    assert canonical_yaml(left) == canonical_yaml(right)

    first = materialize_run(
        make_group(),
        FakeTrial(0),
        {"topology": "dual"},
    )
    reordered_group = make_group()
    object.__setattr__(
        reordered_group,
        "metadata",
        MappingProxyType({"owner": {"name": "research"}, "labels": ["baseline"]}),
    )
    second = materialize_run(
        reordered_group,
        FakeTrial(0),
        {"topology": "dual"},
    )

    assert first.config_yaml == second.config_yaml
    assert first.config_sha256 == second.config_sha256
    assert first.run_json == second.run_json
    assert first.run_sha256 == second.run_sha256
