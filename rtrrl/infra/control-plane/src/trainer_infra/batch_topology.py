from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

ACCOUNT_ID = "007122174918"
REGION = "eu-north-1"


class ExecutionPurpose(StrEnum):
    DEV = "dev"
    RUN = "run"


@dataclass(frozen=True)
class ComputeEnvironmentSpec:
    name: str
    instance_type: str
    max_vcpus: int
    ami_family: str


@dataclass(frozen=True)
class ResourceProfile:
    name: str
    compute_environment: str
    vcpus: int
    memory_mib: int
    gpus: int
    gpu_model: str | None = None

    @property
    def resource_requirements(self) -> tuple[tuple[str, str], ...]:
        values = [("VCPU", str(self.vcpus)), ("MEMORY", str(self.memory_mib))]
        if self.gpus:
            values.append(("GPU", str(self.gpus)))
        return tuple(values)


@dataclass(frozen=True)
class QueueSpec:
    name: str
    priority: int
    compute_environments: tuple[str, ...]
    purpose: ExecutionPurpose


@dataclass(frozen=True)
class BatchTopology:
    compute_environments: Mapping[str, ComputeEnvironmentSpec]
    profiles: Mapping[str, ResourceProfile]
    queues: Mapping[str, QueueSpec]


class ProfileDriftError(RuntimeError):
    """Raised when deployed Batch topology differs from the exact contract."""


@dataclass(frozen=True)
class AwsNetworkSettings:
    subnets: tuple[str, ...]
    security_group_ids: tuple[str, ...]
    instance_role: str


@dataclass(frozen=True)
class ValidatedTopology:
    compute_environment_arns: Mapping[str, str]
    queue_arns: Mapping[str, str]


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

_COMPUTE_ENVIRONMENTS = MappingProxyType(
    {
        "c7am": ComputeEnvironmentSpec(
            "rtrrl-cpu-c7am-ce", "c7a.medium", 16, "ECS_AL2023"
        ),
        "c7al": ComputeEnvironmentSpec(
            "rtrrl-cpu-c7al-ce", "c7a.large", 32, "ECS_AL2023"
        ),
        "c7ax": ComputeEnvironmentSpec(
            "rtrrl-cpu-c7ax-ce", "c7a.xlarge", 16, "ECS_AL2023"
        ),
        "g6x": ComputeEnvironmentSpec(
            "rtrrl-gpu-g6x-ce", "g6.xlarge", 32, "ECS_AL2023_NVIDIA"
        ),
    }
)
_PROFILES = MappingProxyType(
    {
        "c7am": ResourceProfile("c7am", "c7am", 1, 1600, 0),
        "c7al": ResourceProfile("c7al", "c7al", 2, 3200, 0),
        "c7ax": ResourceProfile("c7ax", "c7ax", 4, 7168, 0),
        "g6x": ResourceProfile("g6x", "g6x", 4, 12000, 1, "NVIDIA L4"),
    }
)
_QUEUES = MappingProxyType(
    {
        "dev-c7am": QueueSpec(
            "dev-cpu-c7am-queue", 10, ("c7am",), ExecutionPurpose.DEV
        ),
        "dev-c7al": QueueSpec(
            "dev-cpu-c7al-queue", 10, ("c7al",), ExecutionPurpose.DEV
        ),
        "dev-c7ax": QueueSpec(
            "dev-cpu-c7ax-queue", 10, ("c7ax",), ExecutionPurpose.DEV
        ),
        "run-c7am": QueueSpec(
            "run-cpu-c7am-queue", 100, ("c7am",), ExecutionPurpose.RUN
        ),
        "run-c7al": QueueSpec(
            "run-cpu-c7al-queue", 100, ("c7al",), ExecutionPurpose.RUN
        ),
        "run-c7ax": QueueSpec(
            "run-cpu-c7ax-queue", 100, ("c7ax",), ExecutionPurpose.RUN
        ),
        "dev-g6x": QueueSpec(
            "dev-gpu-queue", 10, ("g6x",), ExecutionPurpose.DEV
        ),
        "run-g6x": QueueSpec(
            "run-gpu-queue", 100, ("g6x",), ExecutionPurpose.RUN
        ),
    }
)
_TOPOLOGY = BatchTopology(
    compute_environments=_COMPUTE_ENVIRONMENTS,
    profiles=_PROFILES,
    queues=_QUEUES,
)


def expected_topology() -> BatchTopology:
    return _TOPOLOGY


def queue_for(purpose: ExecutionPurpose, profile: str) -> QueueSpec:
    if profile not in _PROFILES:
        raise ValueError(f"unknown Batch resource profile: {profile!r}")
    return _QUEUES[f"{purpose.value}-{profile}"]


def _drift(field: str, expected: object, actual: object) -> ProfileDriftError:
    return ProfileDriftError(f"{field}: expected {expected!r}, got {actual!r}")


def _require_field(
    resource: Mapping[str, Any],
    field: str,
    expected: object,
    *,
    path: str = "",
) -> None:
    actual = resource.get(field)
    if type(actual) is not type(expected) or actual != expected:
        raise _drift(f"{path}{field}", expected, actual)


def _require_string_set(
    resource: Mapping[str, Any],
    field: str,
    expected: tuple[str, ...],
    *,
    path: str = "",
) -> None:
    actual = resource.get(field)
    field_path = f"{path}{field}"
    if not isinstance(actual, list) or any(type(value) is not str for value in actual):
        raise ProfileDriftError(
            f"{field_path}: expected a list of strings, got {actual!r}"
        )
    if len(actual) != len(set(actual)):
        raise ProfileDriftError(
            f"{field_path}: duplicate values are not allowed: {actual!r}"
        )
    if len(expected) != len(set(expected)):
        raise ProfileDriftError(
            f"{field_path}: duplicate expected values are not allowed: {expected!r}"
        )
    if set(actual) != set(expected):
        raise ProfileDriftError(
            f"{field_path}: expected elements {expected!r}, got {actual!r}"
        )


def _require_mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfileDriftError(f"{path}: expected mapping, got {value!r}")
    return value


def _require_list(value: object, path: str) -> list[Any]:
    if type(value) is not list:
        raise ProfileDriftError(f"{path}: expected list, got {value!r}")
    return value


def require_one(
    resources: object,
    *,
    kind: str,
    name: str,
    path: str | None = None,
) -> Mapping[str, Any]:
    field_path = path or kind
    values = _require_list(resources, field_path)
    if len(values) != 1:
        raise ProfileDriftError(
            f"{field_path}: expected exactly one {kind} {name!r}, got {len(values)}"
        )
    return _require_mapping(values[0], f"{field_path}[0]")


def _validate_batch_arn(
    value: object,
    *,
    resource_type: str,
    name: str,
    path: str,
) -> str:
    expected = f"arn:aws:batch:{REGION}:{ACCOUNT_ID}:{resource_type}/{name}"
    if not isinstance(value, str):
        raise _drift(path, expected, value)
    parts = value.split(":", 5)
    if len(parts) != 6:
        raise _drift(path, expected, value)
    prefix, partition, service, region, account, resource = parts
    resource_parts = resource.split("/", 1)
    if (
        prefix != "arn"
        or partition != "aws"
        or service != "batch"
        or region != REGION
        or account != ACCOUNT_ID
        or resource_parts != [resource_type, name]
    ):
        raise _drift(path, expected, value)
    return value


def _validate_ami_configuration(
    resources: Mapping[str, Any],
    expected: ComputeEnvironmentSpec,
) -> None:
    path = "computeResources."
    configuration = require_one(
        resources.get("ec2Configuration"),
        kind="AMI configuration",
        name=expected.name,
        path=f"{path}ec2Configuration",
    )
    _require_field(
        configuration,
        "imageType",
        expected.ami_family,
        path=f"{path}ec2Configuration[0].",
    )
    if "imageIdOverride" in configuration:
        _require_field(
            configuration,
            "imageIdOverride",
            "",
            path=f"{path}ec2Configuration[0].",
        )
    if "imageId" in resources:
        raise ProfileDriftError(
            f"{path}imageId: expected field to be absent, "
            f"got {resources['imageId']!r}"
        )
    if "launchTemplate" in resources:
        launch_template = _require_mapping(
            resources["launchTemplate"], f"{path}launchTemplate"
        )
        if launch_template:
            raise ProfileDriftError(
                f"{path}launchTemplate: expected empty mapping, "
                f"got {launch_template!r}"
            )


def _validate_compute_environment_order(
    actual: Mapping[str, Any],
    expected: QueueSpec,
    environment_arns: Mapping[str, str],
) -> None:
    path = "computeEnvironmentOrder"
    entries = _require_list(actual.get(path), path)
    if len(entries) != len(expected.compute_environments):
        raise _drift(path, f"{len(expected.compute_environments)} entries", entries)
    for index, (entry_value, environment_key) in enumerate(
        zip(entries, expected.compute_environments, strict=True)
    ):
        entry_path = f"{path}[{index}]"
        entry = _require_mapping(entry_value, entry_path)
        _require_field(entry, "order", index + 1, path=f"{entry_path}.")
        _require_field(
            entry,
            "computeEnvironment",
            environment_arns[environment_key],
            path=f"{entry_path}.",
        )


def validate_compute_environment(
    actual: Mapping[str, Any],
    expected: ComputeEnvironmentSpec,
    network: AwsNetworkSettings,
) -> str:
    _require_field(actual, "computeEnvironmentName", expected.name)
    _require_field(actual, "type", "MANAGED")
    _require_field(actual, "state", "ENABLED")
    _require_field(actual, "status", "VALID")
    resources = _require_mapping(actual.get("computeResources"), "computeResources")
    path = "computeResources."
    _require_field(resources, "type", "EC2", path=path)
    _require_field(resources, "minvCpus", 0, path=path)
    _require_field(resources, "maxvCpus", expected.max_vcpus, path=path)
    _require_field(resources, "instanceTypes", [expected.instance_type], path=path)
    _require_string_set(resources, "subnets", network.subnets, path=path)
    _require_string_set(
        resources,
        "securityGroupIds",
        network.security_group_ids,
        path=path,
    )
    _require_field(resources, "instanceRole", network.instance_role, path=path)
    _validate_ami_configuration(resources, expected)
    return _validate_batch_arn(
        actual.get("computeEnvironmentArn"),
        resource_type="compute-environment",
        name=expected.name,
        path="computeEnvironmentArn",
    )


def validate_queue(
    actual: Mapping[str, Any],
    expected: QueueSpec,
    environment_arns: Mapping[str, str],
) -> str:
    _require_field(actual, "jobQueueName", expected.name)
    _require_field(actual, "state", "ENABLED")
    _require_field(actual, "status", "VALID")
    _require_field(actual, "priority", expected.priority)
    _validate_compute_environment_order(actual, expected, environment_arns)
    return _validate_batch_arn(
        actual.get("jobQueueArn"),
        resource_type="job-queue",
        name=expected.name,
        path="jobQueueArn",
    )


class BatchTopologyValidator:
    def __init__(
        self,
        batch: Any,
        sts: Any,
        topology: BatchTopology = _TOPOLOGY,
        network: AwsNetworkSettings = DEFAULT_AWS_NETWORK_SETTINGS,
    ) -> None:
        self._batch = batch
        self._sts = sts
        self._topology = topology
        self._network = network

    def validate(self) -> ValidatedTopology:
        identity = _require_mapping(
            self._sts.get_caller_identity(), "sts.get_caller_identity"
        )
        account = identity.get("Account")
        meta = getattr(self._batch, "meta", None)
        region = getattr(meta, "region_name", None)
        if (
            type(account) is not str
            or account != ACCOUNT_ID
            or type(region) is not str
            or region != REGION
        ):
            raise ProfileDriftError(
                "sts.Account/batch.meta.region_name: expected "
                f"{ACCOUNT_ID}/{REGION}, got {account!r}/{region!r}"
            )

        environment_arns: dict[str, str] = {}
        for key, expected in self._topology.compute_environments.items():
            response = _require_mapping(
                self._batch.describe_compute_environments(
                    computeEnvironments=[expected.name]
                ),
                "describe_compute_environments",
            )
            actual = require_one(
                response.get("computeEnvironments"),
                kind="compute environment",
                name=expected.name,
                path="describe_compute_environments.computeEnvironments",
            )
            environment_arns[key] = validate_compute_environment(
                actual, expected, self._network
            )

        queue_arns: dict[str, str] = {}
        for key, expected in self._topology.queues.items():
            response = _require_mapping(
                self._batch.describe_job_queues(jobQueues=[expected.name]),
                "describe_job_queues",
            )
            actual = require_one(
                response.get("jobQueues"),
                kind="job queue",
                name=expected.name,
                path="describe_job_queues.jobQueues",
            )
            queue_arns[key] = validate_queue(actual, expected, environment_arns)

        return ValidatedTopology(
            compute_environment_arns=MappingProxyType(environment_arns),
            queue_arns=MappingProxyType(queue_arns),
        )
