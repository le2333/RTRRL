from __future__ import annotations

from typing import Any

import pytest

from trainer_infra.adapters.aws_batch import (
    AwsBatchAdapter,
    AwsInfrastructurePreflightContract,
    AwsBatchPreflight,
    AwsBatchPreflightContract,
    JobDefinitionExpectation,
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


@pytest.mark.parametrize(
    "prefix",
    [
        "s3://bucket/other/experiments/e/",
        "s3://bucket/experiments/e/../x/",
        "s3://bucket/experiments/e/?version=x",
        "s3://bucket/experiments/e/#fragment",
    ],
)
def test_adapter_rejects_noncanonical_experiment_prefix(prefix: str) -> None:
    with pytest.raises(ValueError, match="experiment|S3 URI"):
        AwsBatchAdapter(FakeBatch(), prefix)


@pytest.mark.parametrize("job_id", ["../escape", "nested/id", "id?query", "id#fragment"])
def test_submit_rejects_job_ids_that_cannot_form_one_s3_key(job_id: str) -> None:
    payload = make_job().model_dump(mode="json")
    payload["job_id"] = job_id
    bundle = JobBundle.model_validate(payload)
    with pytest.raises(ValueError, match="S3|key|job"):
        AwsBatchAdapter(FakeBatch(), "s3://bucket/experiments/e/").submit(
            bundle, PROFILES["g6x"], definition()
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
        profile_name = next(
            key for key, item in PROFILES.items() if item.compute_environment == name
        )
        instance_type, max_vcpus, image_type = {
            "c7am": ("c7a.medium", 16, "ECS_AL2023"),
            "c7al": ("c7a.large", 32, "ECS_AL2023"),
            "c7ax": ("c7a.xlarge", 16, "ECS_AL2023"),
            "g6x": ("g6.xlarge", 32, "ECS_AL2023_NVIDIA"),
        }[profile_name]
        return {
            "computeEnvironments": [
                {
                    "computeEnvironmentName": name,
                    "computeEnvironmentArn": f"arn:ce:{name}",
                    "type": "MANAGED",
                    "state": "ENABLED",
                    "status": "VALID",
                    "computeResources": {
                        "type": "EC2",
                        "instanceTypes": [instance_type],
                        "minvCpus": 0,
                        "maxvCpus": max_vcpus,
                        "subnets": ["subnet-a", "subnet-b"],
                        "securityGroupIds": ["sg-a"],
                        "instanceRole": "arn:aws:iam::123456789012:instance-profile/ecs",
                        "ec2Configuration": [{"imageType": image_type}],
                    },
                }
            ]
        }

    def describe_job_queues(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("describe_job_queues", kwargs))
        name = kwargs["jobQueues"][0]
        profile = next(
            item
            for item in PROFILES.values()
            if name in {item.dev_queue, item.run_queue}
        )
        priority = 10 if name == profile.dev_queue else 100
        return {
            "jobQueues": [
                {
                    "jobQueueName": name,
                    "state": "ENABLED",
                    "status": "VALID",
                    "priority": priority,
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
                    "type": "container",
                    "platformCapabilities": ["EC2"],
                    "containerProperties": {
                        "image": IMAGE,
                        "resourceRequirements": resources,
                        "command": ["python", "/opt/trainer/worker.py"],
                        "environment": [
                            {"name": "TRAINER_WORKER_PROTOCOL_VERSION", "value": "1"}
                        ],
                        "jobRoleArn": "arn:aws:iam::123456789012:role/job",
                        "executionRoleArn": "arn:aws:iam::123456789012:role/execution",
                        "logConfiguration": {"logDriver": "awslogs"},
                    },
                }
            ]
        }


def test_read_only_preflight_validates_all_four_profiles_and_definitions() -> None:
    client = FakePreflightBatch()
    definitions = {
        name: JobDefinitionExpectation(
            arn=definition(name).arn,
            job_role_arn="arn:aws:iam::123456789012:role/job",
            execution_role_arn="arn:aws:iam::123456789012:role/execution",
            worker_protocol_version="1",
            log_configuration={"logDriver": "awslogs"},
        )
        for name in PROFILES
    }
    contract = AwsBatchPreflightContract(
        subnets=("subnet-a", "subnet-b"),
        security_group_ids=("sg-a",),
        instance_role="arn:aws:iam::123456789012:instance-profile/ecs",
        job_definitions=definitions,
    )

    validated = AwsBatchPreflight(client).validate(contract)

    assert {item.resource_profile for item in validated} == set(PROFILES)
    assert {name for name, _ in client.calls} == {
        "describe_compute_environments",
        "describe_job_queues",
        "describe_job_definitions",
    }
    for forbidden in ("register", "update", "delete", "cancel", "retry", "cleanup"):
        assert not hasattr(AwsBatchAdapter, forbidden)
        assert not hasattr(AwsBatchPreflight, forbidden)


def test_authoritative_infrastructure_preflight_can_validate_without_definitions() -> None:
    client = FakePreflightBatch()
    contract = AwsInfrastructurePreflightContract(
        subnets=("subnet-a", "subnet-b"),
        security_group_ids=("sg-a",),
        instance_role="arn:aws:iam::123456789012:instance-profile/ecs",
    )

    validated = AwsBatchPreflight(client).validate_profiles(contract)

    assert [item.profile.name for item in validated] == list(PROFILES)
    assert [
        (item.profile.vcpus, item.profile.memory_mib, item.profile.gpus)
        for item in validated
    ] == [(1, 1600, 0), (2, 3200, 0), (4, 7168, 0), (4, 12000, 1)]
    assert [item.profile.dev_queue for item in validated] == [
        "dev-cpu-c7am-queue",
        "dev-cpu-c7al-queue",
        "dev-cpu-c7ax-queue",
        "dev-gpu-queue",
    ]
    queue_calls = [
        arguments["jobQueues"][0]
        for name, arguments in client.calls
        if name == "describe_job_queues"
    ]
    assert queue_calls == [
        "dev-cpu-c7am-queue",
        "run-cpu-c7am-queue",
        "dev-cpu-c7al-queue",
        "run-cpu-c7al-queue",
        "dev-cpu-c7ax-queue",
        "run-cpu-c7ax-queue",
        "dev-gpu-queue",
        "run-gpu-queue",
    ]
    assert {name for name, _ in client.calls} == {
        "describe_compute_environments",
        "describe_job_queues",
    }


def make_preflight_contract() -> AwsBatchPreflightContract:
    return AwsBatchPreflightContract(
        subnets=("subnet-a", "subnet-b"),
        security_group_ids=("sg-a",),
        instance_role="arn:aws:iam::123456789012:instance-profile/ecs",
        job_definitions={
            name: JobDefinitionExpectation(
                arn=definition(name).arn,
                job_role_arn="arn:aws:iam::123456789012:role/job",
                execution_role_arn="arn:aws:iam::123456789012:role/execution",
                worker_protocol_version="1",
                log_configuration={"logDriver": "awslogs"},
            )
            for name in PROFILES
        },
    )


def test_preflight_accepts_aws_image_status_but_rejects_ami_override() -> None:
    client = FakePreflightBatch()
    original = client.describe_compute_environments

    def with_status(**kwargs: Any) -> dict[str, Any]:
        response = original(**kwargs)
        response["computeEnvironments"][0]["computeResources"]["ec2Configuration"][0][
            "batchImageStatus"
        ] = "LATEST"
        return response

    client.describe_compute_environments = with_status  # type: ignore[method-assign]
    assert len(AwsBatchPreflight(client).validate(make_preflight_contract())) == 4

    def with_override(**kwargs: Any) -> dict[str, Any]:
        response = original(**kwargs)
        response["computeEnvironments"][0]["computeResources"]["ec2Configuration"][0][
            "imageIdOverride"
        ] = "ami-unapproved"
        return response

    client.describe_compute_environments = with_override  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="ec2Configuration"):
        AwsBatchPreflight(client).validate(make_preflight_contract())


@pytest.mark.parametrize(
    ("target", "path", "value"),
    [
        ("ce", ("type",), "UNMANAGED"),
        ("ce", ("state",), "DISABLED"),
        ("ce", ("status",), "INVALID"),
        ("ce", ("computeResources", "instanceTypes"), ["c7a.48xlarge"]),
        ("ce", ("computeResources", "minvCpus"), 1),
        ("ce", ("computeResources", "maxvCpus"), 999),
        ("ce", ("computeResources", "subnets"), ["subnet-other"]),
        ("ce", ("computeResources", "securityGroupIds"), ["sg-other"]),
        ("ce", ("computeResources", "instanceRole"), "wrong"),
        ("queue", ("priority",), 10),
        ("definition", ("status",), "INACTIVE"),
        ("definition", ("type",), "multinode"),
        ("definition", ("platformCapabilities",), ["FARGATE"]),
        ("definition", ("containerProperties", "image"), "repo/image@sha256:" + "b" * 64),
        ("definition", ("containerProperties", "resourceRequirements"), []),
        ("definition", ("containerProperties", "environment"), []),
        ("definition", ("containerProperties", "jobRoleArn"), "wrong"),
        ("definition", ("containerProperties", "executionRoleArn"), "wrong"),
        ("definition", ("containerProperties", "logConfiguration"), {}),
    ],
)
def test_preflight_fails_closed_on_contract_drift(
    target: str, path: tuple[str, ...], value: Any
) -> None:
    client = FakePreflightBatch()
    method_name = {
        "ce": "describe_compute_environments",
        "queue": "describe_job_queues",
        "definition": "describe_job_definitions",
    }[target]
    collection = {
        "ce": "computeEnvironments",
        "queue": "jobQueues",
        "definition": "jobDefinitions",
    }[target]
    original = getattr(client, method_name)

    def drift(**kwargs: Any) -> dict[str, Any]:
        response = original(**kwargs)
        current = response[collection][0]
        for key in path[:-1]:
            current = current[key]
        current[path[-1]] = value
        return response

    setattr(client, method_name, drift)
    with pytest.raises(ValueError):
        AwsBatchPreflight(client).validate(make_preflight_contract())
