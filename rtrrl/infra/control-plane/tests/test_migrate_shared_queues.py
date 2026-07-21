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


def test_dry_run_reports_idle_queue_and_its_unreferenced_environment_deletion(
    migration: Any, matching_aws: Any
) -> None:
    actions = migration.migrate(
        batch=matching_aws.batch,
        sts=matching_aws.sts,
        execute=False,
    )

    assert "delete queue rtrrl-cpu2-queue" in actions
    assert "delete environment rtrrl-cpu2-ce" in actions
    assert matching_aws.batch.mutations == []


def test_dry_run_active_queue_preserves_its_referenced_environment(
    migration: Any, matching_aws: Any
) -> None:
    matching_aws.batch.jobs[("rtrrl-cpu2-queue", "RUNNING")] = ["job-1"]

    actions = migration.migrate(
        batch=matching_aws.batch,
        sts=matching_aws.sts,
        execute=False,
    )

    assert "skip active queue rtrrl-cpu2-queue" in actions
    assert "skip referenced environment rtrrl-cpu2-ce" in actions
    assert "delete environment rtrrl-cpu2-ce" not in actions
    assert matching_aws.batch.mutations == []


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


def test_existing_environment_allows_read_only_batch_image_status(
    migration: Any, matching_aws: Any
) -> None:
    resources = matching_aws.batch.environments["rtrrl-cpu-c7am-ce"][
        "computeResources"
    ]
    resources["ec2Configuration"] = [
        {
            "imageType": "ECS_AL2023",
            "batchImageStatus": "LATEST",
        }
    ]

    actions = migration.migrate(
        batch=matching_aws.batch,
        sts=matching_aws.sts,
        execute=False,
    )

    assert "reuse environment rtrrl-cpu-c7am-ce" in actions
    assert matching_aws.batch.mutations == []


def test_existing_environment_rejects_nonempty_image_override(
    migration: Any, matching_aws: Any
) -> None:
    resources = matching_aws.batch.environments["rtrrl-cpu-c7am-ce"][
        "computeResources"
    ]
    resources["ec2Configuration"] = [
        {
            "imageType": "ECS_AL2023",
            "batchImageStatus": "LATEST",
            "imageIdOverride": "ami-0123456789abcdef0",
        }
    ]

    with pytest.raises(ValueError, match="ec2Configuration"):
        migration.migrate(
            batch=matching_aws.batch,
            sts=matching_aws.sts,
            execute=True,
        )

    assert matching_aws.batch.mutations == []


@pytest.mark.parametrize("index", range(4))
def test_each_compute_environment_create_payload_is_exact(
    migration: Any, empty_aws: Any, index: int
) -> None:
    migration.migrate(batch=empty_aws.batch, sts=empty_aws.sts, execute=True)
    expected = migration.ENVIRONMENTS[index]
    calls = [
        call
        for name, call in empty_aws.batch.mutations
        if name == "create_compute_environment"
    ]

    assert calls[index] == {
        "computeEnvironmentName": expected["name"],
        "type": "MANAGED",
        "state": "ENABLED",
        "computeResources": {
            "type": "EC2",
            "minvCpus": 0,
            "maxvCpus": expected["max_vcpus"],
            "desiredvCpus": 0,
            "instanceTypes": [expected["instance_type"]],
            "subnets": list(migration.SUBNETS),
            "securityGroupIds": list(migration.SECURITY_GROUPS),
            "instanceRole": migration.INSTANCE_ROLE,
            "ec2Configuration": [{"imageType": expected["image_type"]}],
        },
    }
    assert "tags" not in calls[index]
    assert "tags" not in calls[index]["computeResources"]


@pytest.mark.parametrize("index", range(8))
def test_each_job_queue_create_payload_is_exact(
    migration: Any, empty_aws: Any, index: int
) -> None:
    migration.migrate(batch=empty_aws.batch, sts=empty_aws.sts, execute=True)
    expected = migration.QUEUES[index]
    calls = [
        call
        for name, call in empty_aws.batch.mutations
        if name == "create_job_queue"
    ]

    assert calls[index] == {
        "jobQueueName": expected["name"],
        "state": "ENABLED",
        "priority": expected["priority"],
        "computeEnvironmentOrder": [
            {
                "order": 1,
                "computeEnvironment": empty_aws.batch.environment_arn(
                    expected["environment"]
                ),
            }
        ],
    }
    assert "tags" not in calls[index]


def test_created_environment_uses_redescribed_arn_for_queue_binding(
    migration: Any, empty_aws: Any
) -> None:
    described_arn = "arn:aws:batch:eu-north-1:007122174918:compute-environment/redescribed"
    original_create = empty_aws.batch.create_compute_environment

    def create_with_different_response(**kwargs: Any) -> dict[str, str]:
        response = original_create(**kwargs)
        if kwargs["computeEnvironmentName"] == "rtrrl-cpu-c7am-ce":
            environment = empty_aws.batch.environments["rtrrl-cpu-c7am-ce"]
            environment["computeEnvironmentArn"] = described_arn
            response["computeEnvironmentArn"] = "arn:response-must-not-be-used"
        return response

    empty_aws.batch.create_compute_environment = create_with_different_response

    migration.migrate(batch=empty_aws.batch, sts=empty_aws.sts, execute=True)

    queue_call = next(
        call
        for name, call in empty_aws.batch.mutations
        if name == "create_job_queue" and call["jobQueueName"] == "dev-cpu-c7am-queue"
    )
    assert queue_call["computeEnvironmentOrder"] == [
        {"order": 1, "computeEnvironment": described_arn}
    ]


def test_active_job_on_second_list_page_blocks_queue_deletion(
    migration: Any, matching_aws: Any
) -> None:
    calls: list[dict[str, Any]] = []
    original_list_jobs = matching_aws.batch.list_jobs

    def paginated_list_jobs(**kwargs: Any) -> dict[str, Any]:
        calls.append(deepcopy(kwargs))
        if (
            kwargs["jobQueue"] == "rtrrl-cpu-queue"
            and kwargs["jobStatus"] == "RUNNING"
        ):
            if "nextToken" not in kwargs:
                return {"jobSummaryList": [], "nextToken": "page-2"}
            assert kwargs["nextToken"] == "page-2"
            return {"jobSummaryList": [{"jobId": "job-1"}]}
        return original_list_jobs(**kwargs)

    matching_aws.batch.list_jobs = paginated_list_jobs

    actions = migration.migrate(
        batch=matching_aws.batch,
        sts=matching_aws.sts,
        execute=True,
    )

    assert "skip active queue rtrrl-cpu-queue" in actions
    assert "rtrrl-cpu-queue" not in matching_aws.batch.deleted_queues
    assert any(call.get("nextToken") == "page-2" for call in calls)


def test_remaining_queue_references_are_collected_from_every_page(
    migration: Any, matching_aws: Any
) -> None:
    external_queue = {
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
    original_describe = matching_aws.batch.describe_job_queues
    all_queue_calls: list[dict[str, Any]] = []

    def paginated_describe(**kwargs: Any) -> dict[str, Any]:
        if "jobQueues" in kwargs:
            return original_describe(**kwargs)
        all_queue_calls.append(deepcopy(kwargs))
        if "nextToken" not in kwargs:
            return {"jobQueues": [], "nextToken": "page-2"}
        assert kwargs["nextToken"] == "page-2"
        return {"jobQueues": [external_queue]}

    matching_aws.batch.describe_job_queues = paginated_describe

    actions = migration.migrate(
        batch=matching_aws.batch,
        sts=matching_aws.sts,
        execute=True,
    )

    assert "skip referenced environment rtrrl-cpu2-ce" in actions
    assert "rtrrl-cpu2-ce" not in matching_aws.batch.deleted_environments
    assert all_queue_calls == [{}, {"nextToken": "page-2"}]


def test_disable_and_delete_wait_through_intermediate_states(
    migration: Any, matching_aws: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(migration.time, "sleep", lambda _seconds: None)
    queue_reads = {"disable": 0, "delete": 0}
    environment_reads = {"disable": 0, "delete": 0}
    original_describe_queue = matching_aws.batch.describe_job_queues
    original_describe_environment = matching_aws.batch.describe_compute_environments
    original_update_queue = matching_aws.batch.update_job_queue
    original_delete_queue = matching_aws.batch.delete_job_queue
    original_update_environment = matching_aws.batch.update_compute_environment
    original_delete_environment = matching_aws.batch.delete_compute_environment

    def update_queue(**kwargs: Any) -> dict[str, str]:
        response = original_update_queue(**kwargs)
        if kwargs["jobQueue"] == "rtrrl-cpu2-queue":
            matching_aws.batch.queues["rtrrl-cpu2-queue"]["status"] = "UPDATING"
        return response

    def delete_queue(**kwargs: Any) -> dict[str, Any]:
        if kwargs["jobQueue"] != "rtrrl-cpu2-queue":
            return original_delete_queue(**kwargs)
        matching_aws.batch.mutations.append(("delete_job_queue", deepcopy(kwargs)))
        matching_aws.batch.queues["rtrrl-cpu2-queue"]["status"] = "DELETING"
        return {}

    def describe_queue(**kwargs: Any) -> dict[str, Any]:
        if kwargs.get("jobQueues") == ["rtrrl-cpu2-queue"]:
            queue = matching_aws.batch.queues.get("rtrrl-cpu2-queue")
            if queue is not None and queue["status"] == "UPDATING":
                queue_reads["disable"] += 1
                if queue_reads["disable"] == 2:
                    queue["status"] = "VALID"
            elif queue is not None and queue["status"] == "DELETING":
                queue_reads["delete"] += 1
                if queue_reads["delete"] == 2:
                    matching_aws.batch.queues.pop("rtrrl-cpu2-queue")
        return original_describe_queue(**kwargs)

    def update_environment(**kwargs: Any) -> dict[str, str]:
        response = original_update_environment(**kwargs)
        if kwargs["computeEnvironment"] == "rtrrl-cpu2-ce":
            matching_aws.batch.environments["rtrrl-cpu2-ce"]["status"] = "UPDATING"
        return response

    def delete_environment(**kwargs: Any) -> dict[str, Any]:
        if kwargs["computeEnvironment"] != "rtrrl-cpu2-ce":
            return original_delete_environment(**kwargs)
        matching_aws.batch.mutations.append(
            ("delete_compute_environment", deepcopy(kwargs))
        )
        matching_aws.batch.environments["rtrrl-cpu2-ce"]["status"] = "DELETING"
        return {}

    def describe_environment(**kwargs: Any) -> dict[str, Any]:
        if kwargs["computeEnvironments"] == ["rtrrl-cpu2-ce"]:
            environment = matching_aws.batch.environments.get("rtrrl-cpu2-ce")
            if environment is not None and environment["status"] == "UPDATING":
                environment_reads["disable"] += 1
                if environment_reads["disable"] == 2:
                    environment["status"] = "VALID"
            elif environment is not None and environment["status"] == "DELETING":
                environment_reads["delete"] += 1
                if environment_reads["delete"] == 2:
                    matching_aws.batch.environments.pop("rtrrl-cpu2-ce")
        return original_describe_environment(**kwargs)

    matching_aws.batch.update_job_queue = update_queue
    matching_aws.batch.delete_job_queue = delete_queue
    matching_aws.batch.describe_job_queues = describe_queue
    matching_aws.batch.update_compute_environment = update_environment
    matching_aws.batch.delete_compute_environment = delete_environment
    matching_aws.batch.describe_compute_environments = describe_environment

    migration.migrate(
        batch=matching_aws.batch,
        sts=matching_aws.sts,
        execute=True,
    )

    assert queue_reads == {"disable": 2, "delete": 2}
    assert environment_reads == {"disable": 2, "delete": 2}
