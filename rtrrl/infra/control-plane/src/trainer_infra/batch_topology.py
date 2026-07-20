from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    if actual != expected:
        raise _drift(f"{path}{field}", expected, actual)


def _require_nonempty_string(
    resource: Mapping[str, Any], field: str, *, path: str = ""
) -> str:
    actual = resource.get(field)
    if not isinstance(actual, str) or not actual:
        raise ProfileDriftError(
            f"{path}{field}: expected non-empty string, got {actual!r}"
        )
    return actual


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


def require_one(
    resources: object,
    *,
    kind: str,
    name: str,
) -> Mapping[str, Any]:
    if (
        not isinstance(resources, Sequence)
        or isinstance(resources, (str, bytes))
        or len(resources) != 1
        or not isinstance(resources[0], Mapping)
    ):
        count = len(resources) if isinstance(resources, Sequence) else "non-sequence"
        raise ProfileDriftError(
            f"{kind} {name!r}: expected exactly one result, got {count}"
        )
    return resources[0]


def validate_compute_environment(
    actual: Mapping[str, Any],
    expected: ComputeEnvironmentSpec,
    network: AwsNetworkSettings,
) -> str:
    _require_field(actual, "computeEnvironmentName", expected.name)
    _require_field(actual, "type", "MANAGED")
    _require_field(actual, "state", "ENABLED")
    _require_field(actual, "status", "VALID")
    resources = actual.get("computeResources")
    if not isinstance(resources, Mapping):
        raise _drift("computeResources", "mapping", resources)
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
    configuration = require_one(
        resources.get("ec2Configuration"),
        kind="computeResources.ec2Configuration entry",
        name=expected.name,
    )
    _require_field(
        configuration,
        "imageType",
        expected.ami_family,
        path="computeResources.ec2Configuration[0].",
    )
    return _require_nonempty_string(actual, "computeEnvironmentArn")


def validate_queue(
    actual: Mapping[str, Any],
    expected: QueueSpec,
    environment_arns: Mapping[str, str],
) -> str:
    _require_field(actual, "jobQueueName", expected.name)
    _require_field(actual, "state", "ENABLED")
    _require_field(actual, "status", "VALID")
    _require_field(actual, "priority", expected.priority)
    expected_order = [
        {"order": order, "computeEnvironment": environment_arns[key]}
        for order, key in enumerate(expected.compute_environments, start=1)
    ]
    _require_field(actual, "computeEnvironmentOrder", expected_order)
    return _require_nonempty_string(actual, "jobQueueArn")


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
        account = self._sts.get_caller_identity()["Account"]
        region = self._batch.meta.region_name
        if account != ACCOUNT_ID or region != REGION:
            raise ProfileDriftError(
                f"expected {ACCOUNT_ID}/{REGION}, got {account}/{region}"
            )

        environment_arns: dict[str, str] = {}
        for key, expected in self._topology.compute_environments.items():
            response = self._batch.describe_compute_environments(
                computeEnvironments=[expected.name]
            )
            actual = require_one(
                response.get("computeEnvironments"),
                kind="compute environment",
                name=expected.name,
            )
            environment_arns[key] = validate_compute_environment(
                actual, expected, self._network
            )

        queue_arns: dict[str, str] = {}
        for key, expected in self._topology.queues.items():
            response = self._batch.describe_job_queues(jobQueues=[expected.name])
            actual = require_one(
                response.get("jobQueues"),
                kind="job queue",
                name=expected.name,
            )
            queue_arns[key] = validate_queue(actual, expected, environment_arns)

        return ValidatedTopology(
            compute_environment_arns=MappingProxyType(environment_arns),
            queue_arns=MappingProxyType(queue_arns),
        )
