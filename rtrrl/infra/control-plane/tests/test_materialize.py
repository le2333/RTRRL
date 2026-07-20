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


def test_run_identity_uses_explicit_sequence_independent_of_trial_number() -> None:
    group = make_group()

    first = materialize_run(
        group,
        FakeTrial(7),
        {"topology": "shared"},
        run_number=1,
    )
    second = materialize_run(
        group,
        FakeTrial(19),
        {"topology": "dual"},
        run_number=2,
    )

    assert first.run_number == 1
    assert first.trial_number == 7
    assert first.run_name == "shared-0001"
    assert first.run_id == "experiment-123:shared:0001"
    assert second.run_number == 2
    assert second.trial_number == 19
    assert second.run_name == "shared-0002"
    assert second.run_id == "experiment-123:shared:0002"
    assert second.context["run_number"] == 2
    assert second.context["trial_number"] == 19
    assert second.context["run_name"] == second.run_name
    assert json.loads(second.run_json)["context"] == dict(second.context)


def test_materialize_requires_keyword_only_run_number() -> None:
    group = make_group()
    trial = FakeTrial(0)
    sampled = {"topology": "shared"}

    with pytest.raises(TypeError):
        materialize_run(group, trial, sampled)
    with pytest.raises(TypeError):
        materialize_run(group, trial, sampled, 1)


@pytest.mark.parametrize("run_number", [True, False, 1.5, "1", None])
def test_materialize_rejects_non_integer_run_number(run_number: object) -> None:
    with pytest.raises(TypeError, match="run_number must be an integer"):
        materialize_run(
            make_group(),
            FakeTrial(0),
            {"topology": "shared"},
            run_number=run_number,
        )


@pytest.mark.parametrize("run_number", [0, -1, 10_000])
def test_materialize_rejects_out_of_range_run_number(run_number: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 9999"):
        materialize_run(
            make_group(),
            FakeTrial(0),
            {"topology": "shared"},
            run_number=run_number,
        )


def test_run_identity_is_independent_per_group() -> None:
    first = materialize_run(
        make_group("shared"),
        FakeTrial(0),
        {"topology": "shared"},
        run_number=1,
    )
    second = materialize_run(
        make_group("dual"),
        FakeTrial(0),
        {"topology": "dual"},
        run_number=1,
    )

    assert first.run_name == "shared-0001"
    assert second.run_name == "dual-0001"
    assert first.study_key != second.study_key
    assert first.run_id != second.run_id


def test_materialized_run_is_an_immutable_complete_snapshot() -> None:
    run = materialize_run(
        make_group(),
        FakeTrial(0),
        {"seed": 7, "topology": "dual"},
        run_number=1,
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
            run_number=1,
        )
    with pytest.raises(ValueError, match="missing searchable parameter 'topology'"):
        materialize_run(make_group(), FakeTrial(0), {"seed": 7}, run_number=1)


def test_canonical_serialization_and_hashes_ignore_mapping_input_order() -> None:
    left = {"b": {"z": 1, "a": 2}, "a": [3, 4]}
    right = {"a": [3, 4], "b": {"a": 2, "z": 1}}

    assert canonical_json(left) == canonical_json(right)
    assert canonical_yaml(left) == canonical_yaml(right)

    first = materialize_run(
        make_group(),
        FakeTrial(0),
        {"topology": "dual"},
        run_number=1,
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
        run_number=1,
    )

    assert first.config_yaml == second.config_yaml
    assert first.config_sha256 == second.config_sha256
    assert first.run_json == second.run_json
    assert first.run_sha256 == second.run_sha256
