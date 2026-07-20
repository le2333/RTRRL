from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from trainer_infra.heavy_tests import (
    TEST_PROFILES,
    AwsNetworkSettings,
    ProfileDriftError,
    create_c7ax_if_missing,
    validate_test_profile,
)

NETWORK_SETTINGS = AwsNetworkSettings(
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
REPOSITORY_ROOT = Path(__file__).parents[4]
HEAVY_TEST_IMAGE_DIR = REPOSITORY_ROOT / "infra" / "batch" / "heavy-tests"


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
    validated = validate_test_profile(fake_batch, "g6x")

    assert validated.profile is TEST_PROFILES["g6x"]
    assert validated.queue_arn.endswith("/rtrrl-gpu-g6x-queue")
    assert validated.compute_environment_arn.endswith("/rtrrl-gpu-g6x-ce")
    assert fake_batch.update_calls == []


@pytest.mark.parametrize("name", ["c7am", "c7ax", "g6x"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("subnets", ["subnet-wrong"]),
        (
            "subnets",
            [
                NETWORK_SETTINGS.subnets[0],
                NETWORK_SETTINGS.subnets[0],
                *NETWORK_SETTINGS.subnets[1:],
            ],
        ),
        ("subnets", [*NETWORK_SETTINGS.subnets, 7]),
        ("securityGroupIds", ["sg-wrong"]),
        (
            "securityGroupIds",
            [
                NETWORK_SETTINGS.security_group_ids[0],
                NETWORK_SETTINGS.security_group_ids[0],
            ],
        ),
        ("securityGroupIds", [*NETWORK_SETTINGS.security_group_ids, None]),
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
        validate_test_profile(fake_batch, name)

    assert fake_batch.update_calls == []
    assert fake_batch.create_compute_environment_calls == []
    assert fake_batch.create_job_queue_calls == []


@pytest.mark.parametrize("field", ["subnets", "securityGroupIds"])
def test_network_list_order_is_not_profile_drift(
    fake_batch: FakeBatch, field: str
) -> None:
    settings = AwsNetworkSettings(
        subnets=("subnet-a", "subnet-b"),
        security_group_ids=("sg-a", "sg-b"),
        instance_role=NETWORK_SETTINGS.instance_role,
    )
    compute_resources = fake_batch.compute_environments["c7am"]["computeResources"]
    assert isinstance(compute_resources, dict)
    compute_resources["subnets"] = list(settings.subnets)
    compute_resources["securityGroupIds"] = list(settings.security_group_ids)
    compute_resources[field] = list(reversed(compute_resources[field]))

    validated = validate_test_profile(fake_batch, "c7am", settings=settings)

    assert validated.profile is TEST_PROFILES["c7am"]
    assert fake_batch.update_calls == []


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
        validate_test_profile(fake_batch, "c7am")

    assert fake_batch.update_calls == []
    assert fake_batch.create_compute_environment_calls == []
    assert fake_batch.create_job_queue_calls == []


def test_nonzero_desired_vcpus_is_not_profile_drift(fake_batch: FakeBatch) -> None:
    compute_resources = fake_batch.compute_environments["c7am"]["computeResources"]
    assert isinstance(compute_resources, dict)
    compute_resources["desiredvCpus"] = 4

    validated = validate_test_profile(fake_batch, "c7am")

    assert validated.profile is TEST_PROFILES["c7am"]
    assert fake_batch.update_calls == []


def test_queue_binding_drift_fails_closed(fake_batch: FakeBatch) -> None:
    fake_batch.job_queues["c7am"]["computeEnvironmentOrder"] = [
        {"order": 2, "computeEnvironment": "arn:aws:batch:eu-north-1:123:compute-environment/wrong"}
    ]

    with pytest.raises(ProfileDriftError, match="computeEnvironmentOrder"):
        validate_test_profile(fake_batch, "c7am")

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
        validate_test_profile(fake_batch, "g6x")

    assert fake_batch.update_calls == []


def test_unknown_profile_is_rejected(fake_batch: FakeBatch) -> None:
    with pytest.raises(ValueError, match="unknown test profile"):
        validate_test_profile(fake_batch, "cpu")


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
                "subnets": list(NETWORK_SETTINGS.subnets),
                "securityGroupIds": list(NETWORK_SETTINGS.security_group_ids),
                "instanceRole": NETWORK_SETTINGS.instance_role,
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


def test_heavy_test_builder_creates_a_filtered_temporary_context() -> None:
    builder = (HEAVY_TEST_IMAGE_DIR / "build-image.sh").read_text()

    assert 'mktemp -d' in builder
    assert '"${REPOSITORY_ROOT}/memo/"' in builder
    assert '"${BUILD_CONTEXT}/memo/"' in builder
    assert '"${REPOSITORY_ROOT}/training-sdk/"' in builder
    assert '"${BUILD_CONTEXT}/training-sdk/"' in builder
    assert '"${BUILD_CONTEXT}/Dockerfile"' in builder
    for excluded in (".git", ".venv", "__pycache__", ".cache", "cache", "logs", "*.log"):
        assert f"--exclude={excluded!r}" in builder

    assert "docker build" in builder
    assert '"${BUILD_CONTEXT}"' in builder
    assert '"${REPOSITORY_ROOT}"' not in builder.split("docker build", maxsplit=1)[1]


def test_heavy_test_overlay_installs_current_sources() -> None:
    dockerfile = (HEAVY_TEST_IMAGE_DIR / "Dockerfile").read_text()

    assert "ARG BASE_IMAGE" in dockerfile
    assert "FROM ${BASE_IMAGE}" in dockerfile
    assert "COPY training-sdk /workspace/training-sdk" in dockerfile
    assert "COPY memo /app" in dockerfile
    assert dockerfile.index("/opt/venv/bin/python -m ensurepip") < dockerfile.index(
        "/opt/venv/bin/python -m pip install"
    )
    assert (
        "RUN /opt/venv/bin/python -m pip install /workspace/training-sdk pytest"
        in dockerfile
    )
    assert "WORKDIR /app" in dockerfile
    assert "PYTHONPATH=/workspace/training-sdk/src:/app" in dockerfile
    assert "XLA_PYTHON_CLIENT_PREALLOCATE=false" in dockerfile
    assert "MALLOC_ARENA_MAX=2" in dockerfile
    assert 'ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]' in dockerfile


def test_heavy_test_builder_maps_profiles_and_emits_digest_reference() -> None:
    builder = (HEAVY_TEST_IMAGE_DIR / "build-image.sh").read_text()

    assert "--profile c7am|c7ax|g6x" in builder
    assert "memorax-rtrl-cpu" in builder
    assert "memorax-rtrl-gpu" in builder
    assert builder.count("aws ecr batch-get-image") == 2
    assert builder.index("BASE_DIGEST=") < builder.index("docker build")
    assert "aws ecr describe-images" not in builder
    assert '"image":' in builder
    assert "@${PUSHED_DIGEST}" in builder
