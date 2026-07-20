from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


class ProfileDriftError(RuntimeError):
    """Raised when an AWS Batch resource does not match its fixed profile."""


@dataclass(frozen=True)
class HeavyTestProfile:
    queue: str
    compute_environment: str
    instance_type: str
    vcpus: int
    memory_mib: int
    gpus: int
    gpu_model: str | None = None


@dataclass(frozen=True)
class ValidatedTestProfile:
    profile: HeavyTestProfile
    queue_arn: str
    compute_environment_arn: str


@dataclass(frozen=True)
class AwsNetworkSettings:
    subnets: tuple[str, ...]
    security_group_ids: tuple[str, ...]
    instance_role: str


DEFAULT_AWS_NETWORK_SETTINGS = AwsNetworkSettings(
    subnets=(
        "subnet-08127d1c5d4de6ac2",
        "subnet-0b8c68ea0a9784758",
        "subnet-01a2aa195678f8411",
    ),
    security_group_ids=("sg-0c0ed6b927c5113dc",),
    instance_role=(
        "arn:aws:iam::007122174918:instance-profile/rtrrl-ecs-instance-role"
    ),
)


TEST_PROFILES: Mapping[str, HeavyTestProfile] = MappingProxyType(
    {
        "c7am": HeavyTestProfile(
            queue="rtrrl-cpu-c7am-queue",
            compute_environment="rtrrl-cpu-c7am-ce",
            instance_type="c7a.medium",
            vcpus=1,
            memory_mib=1600,
            gpus=0,
        ),
        "c7ax": HeavyTestProfile(
            queue="rtrrl-cpu-c7ax-queue",
            compute_environment="rtrrl-cpu-c7ax-ce",
            instance_type="c7a.xlarge",
            vcpus=4,
            memory_mib=7168,
            gpus=0,
        ),
        "g6x": HeavyTestProfile(
            queue="rtrrl-gpu-g6x-queue",
            compute_environment="rtrrl-gpu-g6x-ce",
            instance_type="g6.xlarge",
            vcpus=4,
            memory_mib=12000,
            gpus=1,
            gpu_model="NVIDIA L4",
        ),
    }
)

_COMPUTE_RESOURCE_FIELDS: Mapping[str, object] = MappingProxyType(
    {
        "type": "EC2",
        "minvCpus": 0,
        "maxvCpus": 16,
    }
)
_QUEUE_FIELDS: Mapping[str, object] = MappingProxyType(
    {
        "state": "ENABLED",
        "status": "VALID",
        "priority": 1,
    }
)


def _require_field(resource: Mapping[str, Any], field: str, expected: object) -> None:
    actual = resource.get(field)
    if actual != expected:
        raise ProfileDriftError(f"{field}: expected {expected!r}, got {actual!r}")


def _require_nonempty_string(resource: Mapping[str, Any], field: str) -> str:
    value = resource.get(field)
    if not isinstance(value, str) or not value:
        raise ProfileDriftError(
            f"{field}: expected non-empty string, got {value!r}"
        )
    return value


def _require_string_set(
    resource: Mapping[str, Any], field: str, expected: tuple[str, ...]
) -> None:
    actual = resource.get(field)
    if not isinstance(actual, list) or any(type(value) is not str for value in actual):
        raise ProfileDriftError(
            f"{field}: expected a list of strings, got {actual!r}"
        )
    if len(actual) != len(set(actual)):
        raise ProfileDriftError(
            f"{field}: duplicate values are not allowed: {actual!r}"
        )
    if len(expected) != len(set(expected)):
        raise ProfileDriftError(
            f"{field}: duplicate expected values are not allowed: {expected!r}"
        )
    if set(actual) != set(expected):
        raise ProfileDriftError(
            f"{field}: expected elements {expected!r}, got {actual!r}"
        )


def _describe_compute_environment(
    batch: Any, profile: HeavyTestProfile
) -> Mapping[str, Any] | None:
    response = batch.describe_compute_environments(
        computeEnvironments=[profile.compute_environment]
    )
    environments = response.get("computeEnvironments", [])
    return environments[0] if environments else None


def _describe_job_queue(batch: Any, profile: HeavyTestProfile) -> Mapping[str, Any] | None:
    response = batch.describe_job_queues(jobQueues=[profile.queue])
    queues = response.get("jobQueues", [])
    return queues[0] if queues else None


def _validate_compute_environment(
    environment: Mapping[str, Any] | None,
    profile: HeavyTestProfile,
    settings: AwsNetworkSettings,
) -> str:
    if environment is None:
        raise ProfileDriftError(
            f"missing compute environment {profile.compute_environment!r}"
        )

    _require_field(
        environment, "computeEnvironmentName", profile.compute_environment
    )
    _require_field(environment, "type", "MANAGED")
    _require_field(environment, "state", "ENABLED")
    _require_field(environment, "status", "VALID")
    resources = environment.get("computeResources")
    if not isinstance(resources, Mapping):
        raise ProfileDriftError(
            f"computeResources: expected mapping, got {resources!r}"
        )
    for field, expected in _COMPUTE_RESOURCE_FIELDS.items():
        _require_field(resources, field, expected)
    _require_field(resources, "instanceTypes", [profile.instance_type])
    _require_string_set(resources, "subnets", settings.subnets)
    _require_string_set(
        resources, "securityGroupIds", settings.security_group_ids
    )
    _require_field(resources, "instanceRole", settings.instance_role)

    return _require_nonempty_string(environment, "computeEnvironmentArn")


def _validate_job_queue(
    queue: Mapping[str, Any] | None,
    profile: HeavyTestProfile,
    compute_environment_arn: str,
) -> str:
    if queue is None:
        raise ProfileDriftError(f"missing job queue {profile.queue!r}")

    _require_field(queue, "jobQueueName", profile.queue)
    for field, expected in _QUEUE_FIELDS.items():
        _require_field(queue, field, expected)
    _require_field(
        queue,
        "computeEnvironmentOrder",
        [{"order": 1, "computeEnvironment": compute_environment_arn}],
    )

    return _require_nonempty_string(queue, "jobQueueArn")


def _get_profile(name: str) -> HeavyTestProfile:
    try:
        return TEST_PROFILES[name]
    except KeyError as error:
        expected = ", ".join(TEST_PROFILES)
        raise ValueError(
            f"unknown test profile {name!r}; expected one of: {expected}"
        ) from error


def validate_test_profile(
    batch: Any,
    name: str,
    *,
    settings: AwsNetworkSettings = DEFAULT_AWS_NETWORK_SETTINGS,
) -> ValidatedTestProfile:
    profile = _get_profile(name)
    compute_environment_arn = _validate_compute_environment(
        _describe_compute_environment(batch, profile), profile, settings
    )
    queue_arn = _validate_job_queue(
        _describe_job_queue(batch, profile),
        profile,
        compute_environment_arn,
    )
    return ValidatedTestProfile(
        profile=profile,
        queue_arn=queue_arn,
        compute_environment_arn=compute_environment_arn,
    )


def create_c7ax_if_missing(batch: Any, settings: AwsNetworkSettings) -> None:
    profile = TEST_PROFILES["c7ax"]
    environment = _describe_compute_environment(batch, profile)
    if environment is None:
        response = batch.create_compute_environment(
            computeEnvironmentName=profile.compute_environment,
            type="MANAGED",
            state="ENABLED",
            computeResources={
                "type": "EC2",
                "minvCpus": 0,
                "maxvCpus": 16,
                "desiredvCpus": 0,
                "instanceTypes": [profile.instance_type],
                "subnets": list(settings.subnets),
                "securityGroupIds": list(settings.security_group_ids),
                "instanceRole": settings.instance_role,
            },
        )
        compute_environment_arn = _require_nonempty_string(
            response, "computeEnvironmentArn"
        )
    else:
        compute_environment_arn = _validate_compute_environment(
            environment, profile, settings
        )

    queue = _describe_job_queue(batch, profile)
    if queue is None:
        batch.create_job_queue(
            jobQueueName=profile.queue,
            state="ENABLED",
            priority=1,
            computeEnvironmentOrder=[
                {
                    "order": 1,
                    "computeEnvironment": compute_environment_arn,
                }
            ],
        )
    else:
        _validate_job_queue(queue, profile, compute_environment_arn)
