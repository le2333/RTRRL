from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError
from training_sdk import RunContext
from training_sdk.execution import RunBundle as SdkRunBundle

from trainer_infra.aws_profiles import PROFILES, profile
from trainer_infra.execution import (
    CompletionMarker,
    JobBundle,
    JobQuery,
    RunBundle,
    build_run_context,
)
from trainer_infra.identities import canonical_json, canonical_yaml
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


IMAGE = "registry.example/repository/image@sha256:" + "a" * 64


def make_run_context_payload() -> dict[str, object]:
    return {
        "experiment_name": "experiment-123",
        "experiment_id": "facility-456",
        "group": "shared",
        "script": "rtrrl",
        "run_id": "experiment-123:shared:0001",
        "run_number": 1,
        "trial_number": 11,
        "seed": 7,
        "metadata": {"nested": {"items": [1, 2]}},
        "environment": {"name": "brax-hopper", "options": {"backend": "spring"}},
        "training_budget": {"env_steps": 100},
        "fixed_parameters": {"seed": 7},
        "sampled_parameters": {"topology": "shared"},
        "final_parameters": {"seed": 7, "topology": "shared"},
        "image_digest": IMAGE,
        "resource_profile": "g6x",
        "artifact_directory": "/tmp/artifacts/facility-456/shared/0001",
        "logging": {"aim_every_env_steps": 10, "rerun_every_episodes": 2},
        "objective": {"metric": "reward", "direction": "maximize", "reduction": "last"},
    }


def make_run_bundle(
    *,
    image_digest: str = IMAGE,
    resource_profile: str = "g6x",
    run_context: object | None = None,
) -> RunBundle:
    if run_context is None:
        context = make_run_context_payload()
        context["image_digest"] = image_digest
        context["resource_profile"] = resource_profile
    else:
        context = run_context
    context_json = canonical_json(context)
    config = {"parameters": {"seed": 7}}
    config_yaml = canonical_yaml(config)
    return RunBundle(
        run_id="experiment-123:shared:0001",
        attempt=0,
        argv=("python", "-m", "train", "--config", "{config_path}"),
        image_digest=image_digest,
        resource_profile=resource_profile,
        config_yaml=config_yaml,
        config_sha256=hashlib.sha256(config_yaml.encode()).hexdigest(),
        run_context=context,
        run_context_sha256=hashlib.sha256(context_json.encode()).hexdigest(),
        artifact_prefix="experiments/e/groups/shared/runs/r/input/",
    )


def test_control_plane_reexports_sdk_execution_protocol() -> None:
    assert RunBundle is SdkRunBundle


def test_run_and_job_bundles_round_trip_with_canonical_hashes_and_deep_freeze() -> None:
    run = make_run_bundle()
    reordered_context = make_run_context_payload()
    reordered_context["metadata"] = {"nested": {"items": [1, 2]}}
    reordered = RunBundle.model_validate(
        {
            "artifact_prefix": run.artifact_prefix,
            "run_context_sha256": run.run_context_sha256,
            "run_context": reordered_context,
            "config_sha256": run.config_sha256,
            "config_yaml": run.config_yaml,
            "argv": list(run.argv),
            "attempt": 0,
            "run_id": run.run_id,
            "image_digest": IMAGE,
            "resource_profile": "g6x",
        }
    )
    assert run.to_json() == reordered.to_json()
    assert run.sha256 == hashlib.sha256(run.to_json().encode()).hexdigest()
    assert RunBundle.from_json(run.to_json()) == run
    with pytest.raises(TypeError):
        run.run_context["metadata"]["nested"]["items"].append(3)

    job = JobBundle(
        job_id="job-0001",
        image_digest=IMAGE,
        resource_profile="g6x",
        runs=(run,),
    )
    assert JobBundle.from_json(job.to_json()) == job
    assert job.sha256 == hashlib.sha256(job.to_json().encode()).hexdigest()


@pytest.mark.parametrize(
    "image",
    [
        "repository/image:latest",
        "repository/image:tag@sha256:" + "a" * 64,
        "repository/image@sha256:" + "a" * 63,
        "repository/image@sha256:" + "A" * 64,
        "https://registry.example/repository/image@sha256:" + "a" * 64,
    ],
)
def test_bundle_image_digest_requires_exact_immutable_reference(image: str) -> None:
    with pytest.raises(ValidationError, match="image_digest"):
        make_run_bundle(image_digest=image)
    with pytest.raises(ValidationError, match="image_digest"):
        JobBundle(
            job_id="job-0001",
            image_digest=image,
            resource_profile="g6x",
            runs=(make_run_bundle(),),
        )


@pytest.mark.parametrize(
    ("child_overrides", "expected"),
    [
        ({"image_digest": "registry.example/other@sha256:" + "b" * 64}, "image"),
        ({"resource_profile": "c7ax"}, "profile"),
    ],
)
def test_job_rejects_child_identity_mismatch(
    child_overrides: dict[str, str], expected: str
) -> None:
    child = make_run_bundle(**child_overrides)
    with pytest.raises(ValidationError, match=expected):
        JobBundle(
            job_id="job-0001",
            image_digest=IMAGE,
            resource_profile="g6x",
            runs=(child,),
        )


class CustomMapping(Mapping[str, object]):
    def __init__(self, value: dict[str, object]) -> None:
        self._value = value

    def __getitem__(self, key: str) -> object:
        return self._value[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)


@pytest.mark.parametrize(
    "invalid_context",
    [
        CustomMapping(make_run_context_payload()),
        {**make_run_context_payload(), "metadata": {"tuple": (1, 2)}},
        {**make_run_context_payload(), "metadata": {"nan": float("nan")}},
        {**make_run_context_payload(), "metadata": {"inf": float("inf")}},
        {**make_run_context_payload(), "metadata": {1: "not-a-string-key"}},
    ],
)
def test_run_context_rejects_non_strict_json(invalid_context: object) -> None:
    with pytest.raises((TypeError, ValueError, ValidationError)):
        make_run_bundle(run_context=invalid_context)


def test_run_context_is_complete_identity_checked_and_detached_from_input() -> None:
    source = make_run_context_payload()
    run = make_run_bundle(run_context=source)
    source_metadata = source["metadata"]
    assert isinstance(source_metadata, dict)
    nested = source_metadata["nested"]
    assert isinstance(nested, dict)
    items = nested["items"]
    assert isinstance(items, list)
    items.append(3)

    assert run.run_context["metadata"]["nested"]["items"] == [1, 2]
    with pytest.raises(TypeError):
        run.run_context["metadata"]["nested"]["items"].append(3)

    incomplete = make_run_context_payload()
    del incomplete["seed"]
    with pytest.raises(ValidationError, match="run_context"):
        make_run_bundle(run_context=incomplete)

    wrong_image = make_run_context_payload()
    wrong_image["image_digest"] = "registry.example/other@sha256:" + "b" * 64
    with pytest.raises(ValidationError, match="image"):
        make_run_bundle(run_context=wrong_image)

    wrong_profile = make_run_context_payload()
    wrong_profile["resource_profile"] = "c7ax"
    with pytest.raises(ValidationError, match="profile"):
        make_run_bundle(run_context=wrong_profile)


def test_run_bundle_rejects_hash_mismatches_and_noncanonical_yaml() -> None:
    payload = make_run_bundle().model_dump(mode="json")
    payload["config_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="config_sha256"):
        RunBundle.model_validate(payload)

    payload = make_run_bundle().model_dump(mode="json")
    payload["run_context_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="run_context_sha256"):
        RunBundle.model_validate(payload)

    for invalid_yaml in ("parameters: {seed: 7}\n", "parameters: [\n"):
        payload = make_run_bundle().model_dump(mode="json")
        payload["config_yaml"] = invalid_yaml
        payload["config_sha256"] = hashlib.sha256(invalid_yaml.encode()).hexdigest()
        with pytest.raises(ValidationError, match="config_yaml"):
            RunBundle.model_validate(payload)


@pytest.mark.parametrize("attempt", [-1, 1, 2, False, 0.0, "0"])
def test_execution_attempt_is_exact_integer_zero(attempt: object) -> None:
    payload = make_run_bundle().model_dump(mode="json")
    payload["attempt"] = attempt
    with pytest.raises(ValidationError):
        RunBundle.model_validate(payload)


@pytest.mark.parametrize("attempt", [-1, 1, 2, False, 0.0, "0"])
def test_completion_marker_attempt_is_exact_integer_zero(attempt: object) -> None:
    with pytest.raises(ValidationError):
        CompletionMarker(
            run_id="experiment-123:shared:0001",
            attempt=attempt,
            exit_code=0,
            started_at=datetime(2026, 7, 21, 8, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 7, 21, 8, 1, tzinfo=timezone.utc),
        )


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
        replace(make_group(), study_key="facility-456:shared"),
        FakeTrial(11),
        {"topology": "shared"},
        run_number=3,
    )

    context = build_run_context(
        "user-facing-experiment",
        "facility-456",
        "shared",
        concrete,
        "/tmp/artifacts/facility-456/shared/0003",
    )

    assert isinstance(context, RunContext)
    assert context.experiment_name == "user-facing-experiment"
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


def test_build_run_context_rejects_group_and_identity_mismatches() -> None:
    concrete = materialize_run(
        replace(make_group(), study_key="facility-456:shared"),
        FakeTrial(11),
        {"topology": "shared"},
        run_number=3,
    )
    with pytest.raises(ValueError, match="group"):
        build_run_context(
            "user-facing-experiment",
            "facility-456",
            "other",
            concrete,
            Path("/tmp/artifacts"),
        )

    wrong_study = materialize_run(
        replace(make_group(), study_key="other-experiment:shared"),
        FakeTrial(11),
        {"topology": "shared"},
        run_number=3,
    )
    with pytest.raises(ValueError, match="study_key.*experiment_id"):
        build_run_context(
            "user-facing-experiment",
            "facility-456",
            "shared",
            wrong_study,
            Path("/tmp/artifacts"),
        )

    object.__setattr__(concrete, "run_id", "experiment-123:other:0003")
    with pytest.raises(ValueError, match="run_id"):
        build_run_context(
            "user-facing-experiment",
            "facility-456",
            "shared",
            concrete,
            Path("/tmp/artifacts"),
        )


def test_build_run_context_rejects_tagged_image_reference() -> None:
    concrete = materialize_run(
        replace(make_group(), study_key="facility-456:shared"),
        FakeTrial(11),
        {"topology": "shared"},
        run_number=3,
    )
    object.__setattr__(
        concrete,
        "image",
        "registry.example/repository/image:latest@sha256:" + "a" * 64,
    )

    with pytest.raises(ValueError, match="canonical immutable"):
        build_run_context(
            "user-facing-experiment",
            "facility-456",
            "shared",
            concrete,
            Path("/tmp/artifacts"),
        )
