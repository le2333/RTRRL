from __future__ import annotations

from typing import Any

import pytest

from trainer_infra.adapters.aws_batch import (
    AwsBatchAdapter,
    AwsBatchPreflight,
    SubmittedJob,
    ValidatedJobDefinition,
)
from trainer_infra.aws_profiles import PROFILES
from trainer_infra.execution import JobBundle
from test_execution import IMAGE, make_run_bundle


class FakeBatch:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def submit_job(self, **kwargs: Any) -> dict[str, str]:
        self.calls.append(("submit_job", kwargs))
        return {"jobId": "aws-job-1", "jobName": kwargs["jobName"]}

    def describe_jobs(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("describe_jobs", kwargs))
        return {
            "jobs": [
                {
                    "jobId": job_id,
                    "status": "FAILED" if job_id == "job-100" else "RUNNING",
                    "statusReason": "raw FAILED reason" if job_id == "job-100" else None,
                }
                for job_id in kwargs["jobs"]
            ]
        }


def make_job(profile_name: str = "g6x") -> JobBundle:
    return JobBundle(
        job_id="bundle-1",
        image_digest=IMAGE,
        resource_profile=profile_name,
        runs=(make_run_bundle(resource_profile=profile_name),),
    )


def definition(profile_name: str = "g6x") -> ValidatedJobDefinition:
    return ValidatedJobDefinition(
        arn=(
            "arn:aws:batch:eu-north-1:123456789012:job-definition/"
            f"trainer-{profile_name}-{'a' * 64}:7"
        ),
        image_digest=IMAGE,
        resource_profile=profile_name,
    )


def test_submit_uses_exact_run_queue_definition_worker_command_and_resources() -> None:
    client = FakeBatch()
    adapter = AwsBatchAdapter(client, "s3://bucket/experiments/exp-1/")

    submitted = adapter.submit(make_job(), PROFILES["g6x"], definition())

    assert submitted == SubmittedJob(job_id="aws-job-1", bundle_id="bundle-1")
    assert client.calls == [
        (
            "submit_job",
            {
                "jobName": "trainer-bundle-1",
                "jobQueue": "run-gpu-queue",
                "jobDefinition": definition().arn,
                "containerOverrides": {
                    "command": [
                        "python",
                        "/opt/trainer/worker.py",
                        "--bundle-s3-uri",
                        "s3://bucket/experiments/exp-1/jobs/bundle-1/bundle.json",
                    ],
                    "resourceRequirements": [
                        {"type": "VCPU", "value": "4"},
                        {"type": "MEMORY", "value": "12000"},
                        {"type": "GPU", "value": "1"},
                    ],
                },
                "retryStrategy": {"attempts": 1},
            },
        )
    ]


@pytest.mark.parametrize("profile_name", list(PROFILES))
def test_submit_uses_every_task1_profile_exactly(profile_name: str) -> None:
    client = FakeBatch()
    adapter = AwsBatchAdapter(client, "s3://bucket/experiments/e/")
    profile = PROFILES[profile_name]

    adapter.submit(make_job(profile_name), profile, definition(profile_name))

    request = client.calls[0][1]
    assert request["jobQueue"] == profile.run_queue
    resources = request["containerOverrides"]["resourceRequirements"]
    expected = [
        {"type": "VCPU", "value": str(profile.vcpus)},
        {"type": "MEMORY", "value": str(profile.memory_mib)},
    ]
    if profile.gpus:
        expected.append({"type": "GPU", "value": str(profile.gpus)})
    assert resources == expected


def test_submit_rejects_bundle_profile_image_or_unvalidated_definition_drift() -> None:
    adapter = AwsBatchAdapter(FakeBatch(), "s3://bucket/experiments/e/")
    with pytest.raises(ValueError, match="profile"):
        adapter.submit(make_job("g6x"), PROFILES["c7ax"], definition("g6x"))
    wrong_image = definition().model_copy(update={"image_digest": "repo/image@sha256:" + "b" * 64})
    with pytest.raises(ValueError, match="image"):
        adapter.submit(make_job(), PROFILES["g6x"], wrong_image)

    with pytest.raises(ValueError, match="digest-bound"):
        ValidatedJobDefinition(
            arn="arn:aws:batch:eu-north-1:123456789012:job-definition/trainer-g6x:7",
            image_digest=IMAGE,
            resource_profile="g6x",
        )


def test_query_chunks_at_100_and_preserves_raw_failed_state() -> None:
    client = FakeBatch()
    adapter = AwsBatchAdapter(client, "s3://bucket/experiments/e/")
    ids = [f"job-{index}" for index in range(205)]

    result = adapter.query(ids)

    calls = [call for name, call in client.calls if name == "describe_jobs"]
    assert [len(call["jobs"]) for call in calls] == [100, 100, 5]
    assert [item.job_id for item in result] == ids
    assert result[100].status == "FAILED"
    assert result[100].status_reason == "raw FAILED reason"


class FakePreflightBatch:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def describe_compute_environments(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("describe_compute_environments", kwargs))
        name = kwargs["computeEnvironments"][0]
        return {
            "computeEnvironments": [
                {
                    "computeEnvironmentName": name,
                    "computeEnvironmentArn": f"arn:ce:{name}",
                    "state": "ENABLED",
                    "status": "VALID",
                }
            ]
        }

    def describe_job_queues(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("describe_job_queues", kwargs))
        name = kwargs["jobQueues"][0]
        profile = next(item for item in PROFILES.values() if item.run_queue == name)
        return {
            "jobQueues": [
                {
                    "jobQueueName": name,
                    "state": "ENABLED",
                    "status": "VALID",
                    "computeEnvironmentOrder": [
                        {"order": 1, "computeEnvironment": f"arn:ce:{profile.compute_environment}"}
                    ],
                }
            ]
        }

    def describe_job_definitions(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("describe_job_definitions", kwargs))
        definition_name = kwargs["jobDefinitionName"]
        profile_name = definition_name.removeprefix("trainer-").split("-", 1)[0]
        profile = PROFILES[profile_name]
        resources = [
            {"type": "VCPU", "value": str(profile.vcpus)},
            {"type": "MEMORY", "value": str(profile.memory_mib)},
        ]
        if profile.gpus:
            resources.append({"type": "GPU", "value": str(profile.gpus)})
        arn = (
            "arn:aws:batch:eu-north-1:123456789012:"
            f"job-definition/{definition_name}:7"
        )
        return {
            "jobDefinitions": [
                {
                    "jobDefinitionArn": arn,
                    "status": "ACTIVE",
                    "containerProperties": {
                        "image": IMAGE,
                        "resourceRequirements": resources,
                    },
                }
            ]
        }


def test_read_only_preflight_validates_all_four_profiles_and_definitions() -> None:
    client = FakePreflightBatch()
    definitions = {name: definition(name).arn for name in PROFILES}

    validated = AwsBatchPreflight(client).validate(definitions)

    assert {item.resource_profile for item in validated} == set(PROFILES)
    assert {name for name, _ in client.calls} == {
        "describe_compute_environments",
        "describe_job_queues",
        "describe_job_definitions",
    }
    for forbidden in ("register", "update", "delete", "cancel", "retry", "cleanup"):
        assert not hasattr(AwsBatchAdapter, forbidden)
        assert not hasattr(AwsBatchPreflight, forbidden)
