from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "migrate_shared_queues.py"


@pytest.fixture
def migration():
    spec = importlib.util.spec_from_file_location("migrate_shared_queues", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeSts:
    def __init__(self, account: str = "007122174918") -> None:
        self.account = account

    def get_caller_identity(self) -> dict[str, str]:
        return {"Account": self.account}


class FakeBatch:
    def __init__(self, migration: Any, *, matching: bool) -> None:
        self.migration = migration
        self.meta = SimpleNamespace(region_name="eu-north-1")
        self.environments: dict[str, dict[str, Any]] = {}
        self.queues: dict[str, dict[str, Any]] = {}
        self.jobs: dict[tuple[str, str], list[str]] = {}
        self.mutations: list[tuple[str, dict[str, Any]]] = []
        if matching:
            for environment in migration.ENVIRONMENTS:
                self.environments[environment["name"]] = self._environment(environment)
            for queue in migration.QUEUES:
                self.queues[queue["name"]] = self._queue(queue)
            self._add_old_resources()

    def _environment(self, expected: dict[str, Any]) -> dict[str, Any]:
        name = expected["name"]
        return {
            "computeEnvironmentName": name,
            "computeEnvironmentArn": self.environment_arn(name),
            "type": "MANAGED",
            "state": "ENABLED",
            "status": "VALID",
            "computeResources": {
                "type": "EC2",
                "minvCpus": 0,
                "maxvCpus": expected["max_vcpus"],
                "desiredvCpus": 0,
                "instanceTypes": [expected["instance_type"]],
                "subnets": list(self.migration.SUBNETS),
                "securityGroupIds": list(self.migration.SECURITY_GROUPS),
                "instanceRole": self.migration.INSTANCE_ROLE,
                "ec2Configuration": [{"imageType": expected["image_type"]}],
            },
        }

    def _queue(self, expected: dict[str, Any]) -> dict[str, Any]:
        name = expected["name"]
        return {
            "jobQueueName": name,
            "jobQueueArn": self.queue_arn(name),
            "state": "ENABLED",
            "status": "VALID",
            "priority": expected["priority"],
            "computeEnvironmentOrder": [
                {
                    "order": 1,
                    "computeEnvironment": self.environment_arn(expected["environment"]),
                }
            ],
        }

    def _add_old_resources(self) -> None:
        old_environment_names = {
            "rtrrl-cpu-ce",
            "rtrrl-cpu2-ce",
            "rtrrl-gpu-ce",
        }
        for name in old_environment_names:
            self.environments[name] = {
                "computeEnvironmentName": name,
                "computeEnvironmentArn": self.environment_arn(name),
                "type": "MANAGED",
                "state": "ENABLED",
                "status": "VALID",
                "computeResources": {},
            }
        old_bindings = {
            "rtrrl-cpu-queue": "rtrrl-cpu-ce",
            "rtrrl-cpu2-queue": "rtrrl-cpu2-ce",
            "rtrrl-gpu-queue": "rtrrl-gpu-ce",
        }
        for name in self.migration.OLD_QUEUES:
            environment = old_bindings.get(name)
            if environment is None:
                target = name.removeprefix("rtrrl-").removesuffix("-queue")
                environment = f"rtrrl-{target}-ce"
            self.queues[name] = {
                "jobQueueName": name,
                "jobQueueArn": self.queue_arn(name),
                "state": "ENABLED",
                "status": "VALID",
                "priority": 1,
                "computeEnvironmentOrder": [
                    {
                        "order": 1,
                        "computeEnvironment": self.environment_arn(environment),
                    }
                ],
            }

    @staticmethod
    def environment_arn(name: str) -> str:
        return (
            "arn:aws:batch:eu-north-1:007122174918:"
            f"compute-environment/{name}"
        )

    @staticmethod
    def queue_arn(name: str) -> str:
        return f"arn:aws:batch:eu-north-1:007122174918:job-queue/{name}"

    def describe_compute_environments(
        self, *, computeEnvironments: list[str]
    ) -> dict[str, Any]:
        return {
            "computeEnvironments": [
                deepcopy(self.environments[name])
                for name in computeEnvironments
                if name in self.environments
            ]
        }

    def create_compute_environment(self, **kwargs: Any) -> dict[str, str]:
        self.mutations.append(("create_compute_environment", deepcopy(kwargs)))
        expected = next(
            item
            for item in self.migration.ENVIRONMENTS
            if item["name"] == kwargs["computeEnvironmentName"]
        )
        self.environments[expected["name"]] = self._environment(expected)
        return {
            "computeEnvironmentName": expected["name"],
            "computeEnvironmentArn": self.environment_arn(expected["name"]),
        }

    def describe_job_queues(
        self,
        *,
        jobQueues: list[str] | None = None,
        nextToken: str | None = None,
    ) -> dict[str, Any]:
        assert nextToken is None
        if jobQueues is None:
            queues = list(self.queues.values())
        else:
            queues = [
                queue
                for queue in self.queues.values()
                if queue["jobQueueName"] in jobQueues
                or queue["jobQueueArn"] in jobQueues
            ]
        return {"jobQueues": deepcopy(queues)}

    def create_job_queue(self, **kwargs: Any) -> dict[str, str]:
        self.mutations.append(("create_job_queue", deepcopy(kwargs)))
        expected = next(
            item
            for item in self.migration.QUEUES
            if item["name"] == kwargs["jobQueueName"]
        )
        self.queues[expected["name"]] = self._queue(expected)
        return {
            "jobQueueName": expected["name"],
            "jobQueueArn": self.queue_arn(expected["name"]),
        }

    def list_jobs(
        self,
        *,
        jobQueue: str,
        jobStatus: str,
        nextToken: str | None = None,
    ) -> dict[str, Any]:
        assert nextToken is None
        return {
            "jobSummaryList": [
                {"jobId": job_id}
                for job_id in self.jobs.get((jobQueue, jobStatus), [])
            ]
        }

    def update_job_queue(self, **kwargs: Any) -> dict[str, str]:
        self.mutations.append(("update_job_queue", deepcopy(kwargs)))
        queue = self.queues[kwargs["jobQueue"]]
        queue["state"] = kwargs["state"]
        return {
            "jobQueueName": queue["jobQueueName"],
            "jobQueueArn": queue["jobQueueArn"],
        }

    def delete_job_queue(self, **kwargs: Any) -> dict[str, Any]:
        self.mutations.append(("delete_job_queue", deepcopy(kwargs)))
        self.queues.pop(kwargs["jobQueue"])
        return {}

    def update_compute_environment(self, **kwargs: Any) -> dict[str, str]:
        self.mutations.append(("update_compute_environment", deepcopy(kwargs)))
        environment = self.environments[kwargs["computeEnvironment"]]
        environment["state"] = kwargs["state"]
        return {
            "computeEnvironmentName": environment["computeEnvironmentName"],
            "computeEnvironmentArn": environment["computeEnvironmentArn"],
        }

    def delete_compute_environment(self, **kwargs: Any) -> dict[str, Any]:
        self.mutations.append(("delete_compute_environment", deepcopy(kwargs)))
        self.environments.pop(kwargs["computeEnvironment"])
        return {}

    @property
    def created_instance_types(self) -> list[str]:
        return [
            call["computeResources"]["instanceTypes"][0]
            for name, call in self.mutations
            if name == "create_compute_environment"
        ]

    @property
    def created_queue_names(self) -> list[str]:
        return [
            call["jobQueueName"]
            for name, call in self.mutations
            if name == "create_job_queue"
        ]

    @property
    def deleted_queues(self) -> list[str]:
        return [
            call["jobQueue"]
            for name, call in self.mutations
            if name == "delete_job_queue"
        ]

    @property
    def deleted_environments(self) -> list[str]:
        return [
            call["computeEnvironment"]
            for name, call in self.mutations
            if name == "delete_compute_environment"
        ]


@pytest.fixture
def empty_aws(migration: Any) -> SimpleNamespace:
    return SimpleNamespace(batch=FakeBatch(migration, matching=False), sts=FakeSts())


@pytest.fixture
def matching_aws(migration: Any) -> SimpleNamespace:
    return SimpleNamespace(batch=FakeBatch(migration, matching=True), sts=FakeSts())


def test_fixed_resource_tables(migration: Any) -> None:
    assert [item["name"] for item in migration.ENVIRONMENTS] == [
        "rtrrl-cpu-c7am-ce",
        "rtrrl-cpu-c7al-ce",
        "rtrrl-cpu-c7ax-ce",
        "rtrrl-gpu-g6x-ce",
    ]
    assert [item["instance_type"] for item in migration.ENVIRONMENTS] == [
        "c7a.medium",
        "c7a.large",
        "c7a.xlarge",
        "g6.xlarge",
    ]
    assert [item["max_vcpus"] for item in migration.ENVIRONMENTS] == [16, 32, 16, 32]
    assert [item["image_type"] for item in migration.ENVIRONMENTS] == [
        "ECS_AL2023",
        "ECS_AL2023",
        "ECS_AL2023",
        "ECS_AL2023_NVIDIA",
    ]
    assert [item["name"] for item in migration.QUEUES] == [
        "dev-cpu-c7am-queue",
        "dev-cpu-c7al-queue",
        "dev-cpu-c7ax-queue",
        "run-cpu-c7am-queue",
        "run-cpu-c7al-queue",
        "run-cpu-c7ax-queue",
        "dev-gpu-queue",
        "run-gpu-queue",
    ]
    assert [item["priority"] for item in migration.QUEUES] == [
        10,
        10,
        10,
        100,
        100,
        100,
        10,
        100,
    ]
    assert migration.OLD_QUEUES == (
        "rtrrl-cpu-c7am-queue",
        "rtrrl-cpu-c7al-queue",
        "rtrrl-cpu-c7ax-queue",
        "rtrrl-gpu-g6x-queue",
        "rtrrl-cpu-queue",
        "rtrrl-cpu2-queue",
        "rtrrl-gpu-queue",
    )
    assert migration.OLD_ENVIRONMENTS == (
        "rtrrl-cpu-ce",
        "rtrrl-cpu2-ce",
        "rtrrl-gpu-ce",
    )


def test_default_mode_only_reports_actions(migration: Any, empty_aws: Any) -> None:
    actions = migration.migrate(
        batch=empty_aws.batch,
        sts=empty_aws.sts,
        execute=False,
    )

    assert len([a for a in actions if a.startswith("create environment")]) == 4
    assert len([a for a in actions if a.startswith("create queue")]) == 8
    assert empty_aws.batch.mutations == []


def test_execute_creates_exact_resources(migration: Any, empty_aws: Any) -> None:
    migration.migrate(batch=empty_aws.batch, sts=empty_aws.sts, execute=True)

    assert empty_aws.batch.created_instance_types == [
        "c7a.medium",
        "c7a.large",
        "c7a.xlarge",
        "g6.xlarge",
    ]
    assert empty_aws.batch.created_queue_names == [
        "dev-cpu-c7am-queue",
        "dev-cpu-c7al-queue",
        "dev-cpu-c7ax-queue",
        "run-cpu-c7am-queue",
        "run-cpu-c7al-queue",
        "run-cpu-c7ax-queue",
        "dev-gpu-queue",
        "run-gpu-queue",
    ]


def test_matching_targets_are_reused_without_target_mutation(
    migration: Any, matching_aws: Any
) -> None:
    actions = migration.migrate(
        batch=matching_aws.batch,
        sts=matching_aws.sts,
        execute=False,
    )

    assert len([a for a in actions if a.startswith("reuse environment")]) == 4
    assert len([a for a in actions if a.startswith("reuse queue")]) == 8
    assert matching_aws.batch.mutations == []


@pytest.mark.parametrize(
    ("resource", "field", "value"),
    [
        ("environment", "type", "UNMANAGED"),
        ("environment", "state", "DISABLED"),
        ("environment", "status", "INVALID"),
        ("compute", "type", "SPOT"),
        ("compute", "instanceTypes", ["c7a.large"]),
        ("compute", "minvCpus", 1),
        ("compute", "maxvCpus", 99),
        ("compute", "subnets", ["subnet-wrong"]),
        ("compute", "securityGroupIds", ["sg-wrong"]),
        ("compute", "instanceRole", "wrong-role"),
        ("compute", "ec2Configuration", [{"imageType": "ECS_AL2"}]),
    ],
)
def test_existing_environment_mismatch_fails_without_mutation(
    migration: Any,
    matching_aws: Any,
    resource: str,
    field: str,
    value: Any,
) -> None:
    environment = matching_aws.batch.environments["rtrrl-cpu-c7am-ce"]
    target = environment if resource == "environment" else environment["computeResources"]
    target[field] = value

    with pytest.raises(ValueError, match="rtrrl-cpu-c7am-ce"):
        migration.migrate(
            batch=matching_aws.batch,
            sts=matching_aws.sts,
            execute=True,
        )

    assert matching_aws.batch.mutations == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("state", "DISABLED"),
        ("status", "INVALID"),
        ("priority", 999),
        ("computeEnvironmentOrder", []),
    ],
)
def test_existing_queue_mismatch_fails_without_mutation(
    migration: Any,
    matching_aws: Any,
    field: str,
    value: Any,
) -> None:
    matching_aws.batch.queues["dev-cpu-c7am-queue"][field] = value

    with pytest.raises(ValueError, match="dev-cpu-c7am-queue"):
        migration.migrate(
            batch=matching_aws.batch,
            sts=matching_aws.sts,
            execute=True,
        )

    assert matching_aws.batch.mutations == []


@pytest.mark.parametrize("state", ["SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING"])
def test_each_active_state_skips_only_that_old_queue(
    migration: Any, matching_aws: Any, state: str
) -> None:
    matching_aws.batch.jobs[("rtrrl-cpu-queue", state)] = ["job-1"]

    actions = migration.migrate(
        batch=matching_aws.batch,
        sts=matching_aws.sts,
        execute=True,
    )

    assert "skip active queue rtrrl-cpu-queue" in actions
    assert "rtrrl-cpu-queue" not in matching_aws.batch.deleted_queues
    assert "rtrrl-cpu-ce" not in matching_aws.batch.deleted_environments
    assert "rtrrl-cpu2-queue" in matching_aws.batch.deleted_queues
    assert "rtrrl-cpu2-ce" in matching_aws.batch.deleted_environments


def test_referenced_old_environment_is_preserved(
    migration: Any, matching_aws: Any
) -> None:
    matching_aws.batch.queues["external-queue"] = {
        "jobQueueName": "external-queue",
        "jobQueueArn": matching_aws.batch.queue_arn("external-queue"),
        "state": "ENABLED",
        "status": "VALID",
        "priority": 1,
        "computeEnvironmentOrder": [
            {
                "order": 1,
                "computeEnvironment": matching_aws.batch.environment_arn(
                    "rtrrl-cpu2-ce"
                ),
            }
        ],
    }

    actions = migration.migrate(
        batch=matching_aws.batch,
        sts=matching_aws.sts,
        execute=True,
    )

    assert "skip referenced environment rtrrl-cpu2-ce" in actions
    assert "rtrrl-cpu2-ce" not in matching_aws.batch.deleted_environments


def test_wrong_account_or_region_fails_before_batch_calls(
    migration: Any, empty_aws: Any
) -> None:
    with pytest.raises(ValueError, match="account"):
        migration.migrate(
            batch=empty_aws.batch,
            sts=FakeSts("123456789012"),
            execute=False,
        )
    empty_aws.batch.meta.region_name = "us-east-1"
    with pytest.raises(ValueError, match="region"):
        migration.migrate(
            batch=empty_aws.batch,
            sts=empty_aws.sts,
            execute=False,
        )

    assert empty_aws.batch.mutations == []


def test_aws_errors_propagate_unchanged(migration: Any, empty_aws: Any) -> None:
    error = RuntimeError("AWS unavailable")

    def fail(**_kwargs: Any) -> dict[str, Any]:
        raise error

    empty_aws.batch.describe_compute_environments = fail

    with pytest.raises(RuntimeError) as raised:
        migration.migrate(
            batch=empty_aws.batch,
            sts=empty_aws.sts,
            execute=False,
        )

    assert raised.value is error
