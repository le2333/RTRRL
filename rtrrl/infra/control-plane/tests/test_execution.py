from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import hashlib
import json

import pytest
from pydantic import ValidationError
from training_sdk import RunContext

from trainer_infra.aws_profiles import PROFILES, profile
from trainer_infra.execution import (
    CompletionMarker,
    JobBundle,
    JobQuery,
    RunBundle,
    build_run_context,
)
from trainer_infra.identities import canonical_json
from trainer_infra.materialize import materialize_run
from test_materialize import FakeTrial, make_group


def test_four_formal_aws_profiles_are_exact_and_immutable() -> None:
    assert set(PROFILES) == {"c7am", "c7al", "c7ax", "g6x"}
    assert {
        name: (
            item.run_queue,
            item.compute_environment,
            item.vcpus,
            item.memory_mib,
            item.gpus,
        )
        for name, item in PROFILES.items()
    } == {
        "c7am": ("run-cpu-c7am-queue", "rtrrl-cpu-c7am-ce", 1, 1600, 0),
        "c7al": ("run-cpu-c7al-queue", "rtrrl-cpu-c7al-ce", 2, 3200, 0),
        "c7ax": ("run-cpu-c7ax-queue", "rtrrl-cpu-c7ax-ce", 4, 7168, 0),
        "g6x": ("run-gpu-queue", "rtrrl-gpu-g6x-ce", 4, 12000, 1),
    }
    assert profile("c7al") is PROFILES["c7al"]

    with pytest.raises(TypeError):
        PROFILES["other"] = PROFILES["c7am"]
    with pytest.raises(FrozenInstanceError):
        PROFILES["c7am"].vcpus = 2
    with pytest.raises(ValueError, match="unknown resource profile"):
        profile("gpu")


def make_run_bundle() -> RunBundle:
    context = {"run_id": "experiment-123:shared:0001", "nested": {"items": [1, 2]}}
    context_json = json.dumps(context, separators=(",", ":"), sort_keys=True)
    config_yaml = "parameters:\n  seed: 7\n"
    return RunBundle(
        run_id="experiment-123:shared:0001",
        attempt=0,
        argv=("python", "-m", "train", "--config", "/input/config.yaml"),
        config_yaml=config_yaml,
        config_sha256=hashlib.sha256(config_yaml.encode()).hexdigest(),
        run_context=context,
        run_context_sha256=hashlib.sha256(context_json.encode()).hexdigest(),
        artifact_prefix="experiments/e/groups/shared/runs/r/input/",
    )


def test_run_and_job_bundles_round_trip_with_canonical_hashes_and_deep_freeze() -> None:
    run = make_run_bundle()
    reordered = RunBundle.model_validate(
        {
            "artifact_prefix": run.artifact_prefix,
            "run_context_sha256": run.run_context_sha256,
            "run_context": {"nested": {"items": [1, 2]}, "run_id": run.run_id},
            "config_sha256": run.config_sha256,
            "config_yaml": run.config_yaml,
            "argv": list(run.argv),
            "attempt": 0,
            "run_id": run.run_id,
        }
    )
    assert run.to_json() == reordered.to_json()
    assert run.sha256 == hashlib.sha256(run.to_json().encode()).hexdigest()
    assert RunBundle.from_json(run.to_json()) == run
    with pytest.raises(TypeError):
        run.run_context["nested"]["items"].append(3)

    job = JobBundle(
        job_id="job-0001",
        image_digest="repo/image@sha256:" + "a" * 64,
        resource_profile="g6x",
        runs=(run,),
    )
    assert JobBundle.from_json(job.to_json()) == job
    assert job.sha256 == hashlib.sha256(job.to_json().encode()).hexdigest()


@pytest.mark.parametrize("attempt", [-1, 1, 2, False, 0.0, "0"])
def test_execution_attempt_is_exact_integer_zero(attempt: object) -> None:
    payload = make_run_bundle().model_dump(mode="json")
    payload["attempt"] = attempt
    with pytest.raises(ValidationError):
        RunBundle.model_validate(payload)


def test_completion_marker_and_job_query_are_strict_round_trip_records() -> None:
    marker = CompletionMarker(
        run_id="experiment-123:shared:0001",
        attempt=0,
        exit_code=0,
        started_at=datetime(2026, 7, 21, 8, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 7, 21, 8, 1, tzinfo=timezone.utc),
        artifacts=("checkpoints/latest", "rerun/eval.rrd"),
    )
    assert CompletionMarker.from_json(marker.to_json()) == marker
    assert marker.sha256 == hashlib.sha256(marker.to_json().encode()).hexdigest()
    query = JobQuery(job_id="aws-job", status="FAILED", status_reason="container failed")
    assert JobQuery.from_json(query.to_json()) == query

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        JobQuery(job_id="aws-job", status="RUNNING", retry=True)


def test_build_run_context_populates_every_sdk_field() -> None:
    concrete = materialize_run(
        make_group(),
        FakeTrial(11),
        {"topology": "shared"},
        run_number=3,
    )

    context = build_run_context(
        "facility-456",
        "shared",
        concrete,
        "/tmp/artifacts/facility-456/shared/0003",
    )

    assert isinstance(context, RunContext)
    assert context.experiment_name == "experiment-123"
    assert context.experiment_id == "facility-456"
    assert context.group == "shared"
    assert context.script == concrete.script
    assert context.run_id == concrete.run_id
    assert context.run_number == 3
    assert context.trial_number == 11
    assert context.seed == 7
    assert canonical_json(context.metadata) == canonical_json(concrete.metadata)
    assert canonical_json(context.environment) == canonical_json(
        concrete.environment.model_dump(mode="json")
    )
    assert canonical_json(context.training_budget) == canonical_json(
        concrete.training_budget.model_dump(mode="json")
    )
    assert canonical_json(context.fixed_parameters) == canonical_json(
        concrete.fixed_parameters
    )
    assert canonical_json(context.sampled_parameters) == canonical_json(
        concrete.sampled_parameters
    )
    assert canonical_json(context.final_parameters) == canonical_json(
        concrete.final_parameters
    )
    assert context.image_digest == concrete.image
    assert context.resource_profile == "g6x"
    assert str(context.artifact_directory) == "/tmp/artifacts/facility-456/shared/0003"
    assert context.logging == concrete.logging.model_dump(mode="json")
    assert context.objective == concrete.objective.model_dump(mode="json")
