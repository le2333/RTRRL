from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from trainer_infra.heavy_tests import (
    TEST_PROFILES,
    AwsNetworkSettings,
    ProfileDriftError,
    create_c7ax_if_missing,
    validate_test_profile,
)

NETWORK_SETTINGS = AwsNetworkSettings(
    subnets=("subnet-a", "subnet-b"),
    security_group_ids=("sg-a",),
    instance_role="arn:aws:iam::123456789012:instance-profile/batch",
)


class FakeBatch:
    def __init__(self, *, include_c7ax: bool = True) -> None:
        self.compute_environments = {
            name: self._compute_environment(name)
            for name in ("c7am", "g6x", *(("c7ax",) if include_c7ax else ()))
        }
        self.job_queues = {
            name: self._job_queue(name)
            for name in ("c7am", "g6x", *(("c7ax",) if include_c7ax else ()))
        }
        self.create_compute_environment_calls: list[dict[str, object]] = []
        self.create_job_queue_calls: list[dict[str, object]] = []
        self.update_calls: list[dict[str, object]] = []

    @staticmethod
    def _compute_environment(name: str) -> dict[str, object]:
        profile = TEST_PROFILES[name]
        return {
            "computeEnvironmentName": profile.compute_environment,
            "computeEnvironmentArn": (
                "arn:aws:batch:eu-north-1:123456789012:"
                f"compute-environment/{profile.compute_environment}"
            ),
            "type": "MANAGED",
            "state": "ENABLED",
            "status": "VALID",
            "computeResources": {
                "type": "EC2",
                "minvCpus": 0,
                "maxvCpus": 16,
                "desiredvCpus": 0,
                "instanceTypes": [profile.instance_type],
                "subnets": list(NETWORK_SETTINGS.subnets),
                "securityGroupIds": list(NETWORK_SETTINGS.security_group_ids),
                "instanceRole": NETWORK_SETTINGS.instance_role,
            },
        }

    def _job_queue(self, name: str) -> dict[str, object]:
        profile = TEST_PROFILES[name]
        compute_environment = self.compute_environments[name]
        return {
            "jobQueueName": profile.queue,
            "jobQueueArn": (
                f"arn:aws:batch:eu-north-1:123456789012:job-queue/{profile.queue}"
            ),
            "state": "ENABLED",
            "status": "VALID",
            "priority": 1,
            "computeEnvironmentOrder": [
                {
                    "order": 1,
                    "computeEnvironment": compute_environment["computeEnvironmentArn"],
                }
            ],
        }

    def describe_compute_environments(
        self, *, computeEnvironments: list[str]
    ) -> dict[str, object]:
        environments = [
            deepcopy(environment)
            for name, environment in self.compute_environments.items()
            if TEST_PROFILES[name].compute_environment in computeEnvironments
        ]
        return {"computeEnvironments": environments}

    def describe_job_queues(self, *, jobQueues: list[str]) -> dict[str, object]:
        queues = [
            deepcopy(queue)
            for name, queue in self.job_queues.items()
            if TEST_PROFILES[name].queue in jobQueues
        ]
        return {"jobQueues": queues}

    def create_compute_environment(self, **kwargs: object) -> dict[str, str]:
        self.create_compute_environment_calls.append(deepcopy(kwargs))
        profile = TEST_PROFILES["c7ax"]
        self.compute_environments["c7ax"] = self._compute_environment("c7ax")
        return {
            "computeEnvironmentName": profile.compute_environment,
            "computeEnvironmentArn": self.compute_environments["c7ax"][
                "computeEnvironmentArn"
            ],
        }

    def create_job_queue(self, **kwargs: object) -> dict[str, str]:
        self.create_job_queue_calls.append(deepcopy(kwargs))
        profile = TEST_PROFILES["c7ax"]
        self.job_queues["c7ax"] = self._job_queue("c7ax")
        return {
            "jobQueueName": profile.queue,
            "jobQueueArn": self.job_queues["c7ax"]["jobQueueArn"],
        }


class InvalidCreateArnBatch(FakeBatch):
    def __init__(self, response: dict[str, object]) -> None:
        super().__init__(include_c7ax=False)
        self.response = response

    def create_compute_environment(self, **kwargs: object) -> dict[str, object]:
        super().create_compute_environment(**kwargs)
        return self.response


@pytest.fixture
def fake_batch() -> FakeBatch:
    return FakeBatch()


def test_profiles_are_exact_and_immutable() -> None:
    assert set(TEST_PROFILES) == {"c7am", "c7ax", "g6x"}
    assert TEST_PROFILES["c7am"].queue == "rtrrl-cpu-c7am-queue"
    assert TEST_PROFILES["c7am"].compute_environment == "rtrrl-cpu-c7am-ce"
    assert TEST_PROFILES["c7am"].instance_type == "c7a.medium"
    assert TEST_PROFILES["c7am"].vcpus == 1
    assert TEST_PROFILES["c7am"].memory_mib == 1600
    assert TEST_PROFILES["c7am"].gpus == 0
    assert TEST_PROFILES["c7am"].gpu_model is None
    assert TEST_PROFILES["c7ax"].queue == "rtrrl-cpu-c7ax-queue"
    assert TEST_PROFILES["c7ax"].compute_environment == "rtrrl-cpu-c7ax-ce"
    assert TEST_PROFILES["c7ax"].instance_type == "c7a.xlarge"
    assert TEST_PROFILES["c7ax"].vcpus == 4
    assert TEST_PROFILES["c7ax"].memory_mib == 7168
    assert TEST_PROFILES["c7ax"].gpus == 0
    assert TEST_PROFILES["c7ax"].gpu_model is None
    assert TEST_PROFILES["g6x"].queue == "rtrrl-gpu-g6x-queue"
    assert TEST_PROFILES["g6x"].compute_environment == "rtrrl-gpu-g6x-ce"
    assert TEST_PROFILES["g6x"].instance_type == "g6.xlarge"
    assert TEST_PROFILES["g6x"].vcpus == 4
    assert TEST_PROFILES["g6x"].memory_mib == 12000
    assert TEST_PROFILES["g6x"].gpus == 1
    assert TEST_PROFILES["g6x"].gpu_model == "NVIDIA L4"

    with pytest.raises(TypeError):
        TEST_PROFILES["other"] = TEST_PROFILES["c7am"]
    with pytest.raises(FrozenInstanceError):
        TEST_PROFILES["c7am"].vcpus = 2


def test_validate_returns_exact_profile_and_resource_arns(fake_batch: FakeBatch) -> None:
    validated = validate_test_profile(fake_batch, "g6x", NETWORK_SETTINGS)

    assert validated.profile is TEST_PROFILES["g6x"]
    assert validated.queue_arn.endswith("/rtrrl-gpu-g6x-queue")
    assert validated.compute_environment_arn.endswith("/rtrrl-gpu-g6x-ce")
    assert fake_batch.update_calls == []


@pytest.mark.parametrize("name", ["c7am", "c7ax", "g6x"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("subnets", ["subnet-wrong"]),
        ("securityGroupIds", ["sg-wrong"]),
        ("instanceRole", "arn:aws:iam::123456789012:instance-profile/wrong"),
    ],
)
def test_every_profile_network_drift_fails_closed(
    fake_batch: FakeBatch, name: str, field: str, value: object
) -> None:
    compute_resources = fake_batch.compute_environments[name]["computeResources"]
    assert isinstance(compute_resources, dict)
    compute_resources[field] = value

    with pytest.raises(ProfileDriftError, match=field):
        validate_test_profile(fake_batch, name, NETWORK_SETTINGS)

    assert fake_batch.update_calls == []
    assert fake_batch.create_compute_environment_calls == []
    assert fake_batch.create_job_queue_calls == []


@pytest.mark.parametrize(
    ("resource", "field", "value"),
    [
        ("compute", "computeEnvironmentName", "wrong-ce"),
        ("compute", "type", "UNMANAGED"),
        ("compute", "state", "DISABLED"),
        ("compute", "status", "INVALID"),
        ("resources", "type", "SPOT"),
        ("resources", "minvCpus", 1),
        ("resources", "maxvCpus", 32),
        ("resources", "instanceTypes", ["c7a.large"]),
        ("queue", "jobQueueName", "wrong-queue"),
        ("queue", "state", "DISABLED"),
        ("queue", "status", "INVALID"),
        ("queue", "priority", 2),
        ("queue", "computeEnvironmentOrder", []),
    ],
)
def test_every_existing_profile_drift_fails_closed(
    fake_batch: FakeBatch, resource: str, field: str, value: object
) -> None:
    if resource == "compute":
        fake_batch.compute_environments["c7am"][field] = value
    elif resource == "resources":
        compute_resources = fake_batch.compute_environments["c7am"]["computeResources"]
        assert isinstance(compute_resources, dict)
        compute_resources[field] = value
    else:
        fake_batch.job_queues["c7am"][field] = value

    with pytest.raises(ProfileDriftError, match=field):
        validate_test_profile(fake_batch, "c7am", NETWORK_SETTINGS)

    assert fake_batch.update_calls == []
    assert fake_batch.create_compute_environment_calls == []
    assert fake_batch.create_job_queue_calls == []


def test_nonzero_desired_vcpus_is_not_profile_drift(fake_batch: FakeBatch) -> None:
    compute_resources = fake_batch.compute_environments["c7am"]["computeResources"]
    assert isinstance(compute_resources, dict)
    compute_resources["desiredvCpus"] = 4

    validated = validate_test_profile(fake_batch, "c7am", NETWORK_SETTINGS)

    assert validated.profile is TEST_PROFILES["c7am"]
    assert fake_batch.update_calls == []


def test_queue_binding_drift_fails_closed(fake_batch: FakeBatch) -> None:
    fake_batch.job_queues["c7am"]["computeEnvironmentOrder"] = [
        {"order": 2, "computeEnvironment": "arn:aws:batch:eu-north-1:123:compute-environment/wrong"}
    ]

    with pytest.raises(ProfileDriftError, match="computeEnvironmentOrder"):
        validate_test_profile(fake_batch, "c7am", NETWORK_SETTINGS)

    assert fake_batch.update_calls == []


@pytest.mark.parametrize("resource", ["compute", "queue"])
def test_missing_existing_profile_resource_fails_closed(
    fake_batch: FakeBatch, resource: str
) -> None:
    if resource == "compute":
        del fake_batch.compute_environments["g6x"]
    else:
        del fake_batch.job_queues["g6x"]

    with pytest.raises(ProfileDriftError, match="missing"):
        validate_test_profile(fake_batch, "g6x", NETWORK_SETTINGS)

    assert fake_batch.update_calls == []


def test_unknown_profile_is_rejected(fake_batch: FakeBatch) -> None:
    with pytest.raises(ValueError, match="unknown test profile"):
        validate_test_profile(fake_batch, "cpu", NETWORK_SETTINGS)


def test_existing_c7ax_is_validated_and_never_mutated(fake_batch: FakeBatch) -> None:
    fake_batch.compute_environments["c7ax"]["computeResources"]["instanceTypes"] = [
        "c7a.large"
    ]

    with pytest.raises(ProfileDriftError, match="instanceTypes"):
        create_c7ax_if_missing(
            fake_batch,
            NETWORK_SETTINGS,
        )

    assert fake_batch.update_calls == []
    assert fake_batch.create_compute_environment_calls == []
    assert fake_batch.create_job_queue_calls == []


def test_missing_c7ax_resources_are_created_exactly() -> None:
    fake_batch = FakeBatch(include_c7ax=False)
    settings = NETWORK_SETTINGS

    create_c7ax_if_missing(fake_batch, settings)

    assert fake_batch.create_compute_environment_calls == [
        {
            "computeEnvironmentName": "rtrrl-cpu-c7ax-ce",
            "type": "MANAGED",
            "state": "ENABLED",
            "computeResources": {
                "type": "EC2",
                "minvCpus": 0,
                "maxvCpus": 16,
                "desiredvCpus": 0,
                "instanceTypes": ["c7a.xlarge"],
                "subnets": ["subnet-a", "subnet-b"],
                "securityGroupIds": ["sg-a"],
                "instanceRole": "arn:aws:iam::123456789012:instance-profile/batch",
            },
        }
    ]
    compute_environment_arn = fake_batch.compute_environments["c7ax"][
        "computeEnvironmentArn"
    ]
    assert fake_batch.create_job_queue_calls == [
        {
            "jobQueueName": "rtrrl-cpu-c7ax-queue",
            "state": "ENABLED",
            "priority": 1,
            "computeEnvironmentOrder": [
                {"order": 1, "computeEnvironment": compute_environment_arn}
            ],
        }
    ]
    assert fake_batch.update_calls == []


@pytest.mark.parametrize("response", [{}, {"computeEnvironmentArn": ""}])
def test_invalid_created_compute_environment_arn_fails_closed(
    response: dict[str, object],
) -> None:
    fake_batch = InvalidCreateArnBatch(response)

    with pytest.raises(ProfileDriftError, match="computeEnvironmentArn"):
        create_c7ax_if_missing(fake_batch, NETWORK_SETTINGS)

    assert len(fake_batch.create_compute_environment_calls) == 1
    assert fake_batch.create_job_queue_calls == []
    assert fake_batch.update_calls == []


def test_only_missing_c7ax_queue_is_created(fake_batch: FakeBatch) -> None:
    del fake_batch.job_queues["c7ax"]

    create_c7ax_if_missing(
        fake_batch,
        NETWORK_SETTINGS,
    )

    assert fake_batch.create_compute_environment_calls == []
    assert len(fake_batch.create_job_queue_calls) == 1
    assert fake_batch.update_calls == []
