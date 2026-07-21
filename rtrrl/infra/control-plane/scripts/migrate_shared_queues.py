from __future__ import annotations

import argparse
import time
from typing import Any, Callable

import boto3


ACCOUNT_ID = "007122174918"
REGION = "eu-north-1"
ACTIVE_STATES = ("SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING")
OLD_QUEUES = (
    "rtrrl-cpu-c7am-queue",
    "rtrrl-cpu-c7al-queue",
    "rtrrl-cpu-c7ax-queue",
    "rtrrl-gpu-g6x-queue",
    "rtrrl-cpu-queue",
    "rtrrl-cpu2-queue",
    "rtrrl-gpu-queue",
)
OLD_ENVIRONMENTS = ("rtrrl-cpu-ce", "rtrrl-cpu2-ce", "rtrrl-gpu-ce")
SUBNETS = (
    "subnet-08127d1c5d4de6ac2",
    "subnet-0b8c68ea0a9784758",
    "subnet-01a2aa195678f8411",
)
SECURITY_GROUPS = ("sg-0c0ed6b927c5113dc",)
INSTANCE_ROLE = (
    "arn:aws:iam::007122174918:instance-profile/rtrrl-ecs-instance-role"
)
ENVIRONMENTS = (
    {
        "name": "rtrrl-cpu-c7am-ce",
        "instance_type": "c7a.medium",
        "max_vcpus": 16,
        "image_type": "ECS_AL2023",
    },
    {
        "name": "rtrrl-cpu-c7al-ce",
        "instance_type": "c7a.large",
        "max_vcpus": 32,
        "image_type": "ECS_AL2023",
    },
    {
        "name": "rtrrl-cpu-c7ax-ce",
        "instance_type": "c7a.xlarge",
        "max_vcpus": 16,
        "image_type": "ECS_AL2023",
    },
    {
        "name": "rtrrl-gpu-g6x-ce",
        "instance_type": "g6.xlarge",
        "max_vcpus": 32,
        "image_type": "ECS_AL2023_NVIDIA",
    },
)
QUEUES = (
    {"name": "dev-cpu-c7am-queue", "priority": 10, "environment": "rtrrl-cpu-c7am-ce"},
    {"name": "dev-cpu-c7al-queue", "priority": 10, "environment": "rtrrl-cpu-c7al-ce"},
    {"name": "dev-cpu-c7ax-queue", "priority": 10, "environment": "rtrrl-cpu-c7ax-ce"},
    {"name": "run-cpu-c7am-queue", "priority": 100, "environment": "rtrrl-cpu-c7am-ce"},
    {"name": "run-cpu-c7al-queue", "priority": 100, "environment": "rtrrl-cpu-c7al-ce"},
    {"name": "run-cpu-c7ax-queue", "priority": 100, "environment": "rtrrl-cpu-c7ax-ce"},
    {"name": "dev-gpu-queue", "priority": 10, "environment": "rtrrl-gpu-g6x-ce"},
    {"name": "run-gpu-queue", "priority": 100, "environment": "rtrrl-gpu-g6x-ce"},
)

_WAIT_SECONDS = 300.0
_POLL_SECONDS = 5.0


def _describe_environment(batch: Any, name: str) -> dict[str, Any] | None:
    response = batch.describe_compute_environments(computeEnvironments=[name])
    environments = response.get("computeEnvironments", [])
    return environments[0] if environments else None


def _describe_queue(batch: Any, name: str) -> dict[str, Any] | None:
    response = batch.describe_job_queues(jobQueues=[name])
    queues = response.get("jobQueues", [])
    return queues[0] if queues else None


def _wait_for(
    description: str,
    read: Callable[[], dict[str, Any] | None],
    ready: Callable[[dict[str, Any] | None], bool],
) -> None:
    deadline = time.monotonic() + _WAIT_SECONDS
    while True:
        if ready(read()):
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"timed out waiting for {description}")
        time.sleep(min(_POLL_SECONDS, remaining))


def _environment_mismatch(
    environment: dict[str, Any], expected: dict[str, Any]
) -> str | None:
    name = expected["name"]
    resources = environment.get("computeResources")
    fields = (
        ("computeEnvironmentName", environment.get("computeEnvironmentName"), name),
        ("type", environment.get("type"), "MANAGED"),
        ("state", environment.get("state"), "ENABLED"),
        ("status", environment.get("status"), "VALID"),
    )
    for field, actual, wanted in fields:
        if actual != wanted:
            return f"{field}: expected {wanted!r}, got {actual!r}"
    if not isinstance(resources, dict):
        return f"computeResources: expected mapping, got {resources!r}"
    resource_fields = (
        ("type", resources.get("type"), "EC2"),
        ("instanceTypes", resources.get("instanceTypes"), [expected["instance_type"]]),
        ("minvCpus", resources.get("minvCpus"), 0),
        ("maxvCpus", resources.get("maxvCpus"), expected["max_vcpus"]),
        ("subnets", resources.get("subnets"), list(SUBNETS)),
        (
            "securityGroupIds",
            resources.get("securityGroupIds"),
            list(SECURITY_GROUPS),
        ),
        ("instanceRole", resources.get("instanceRole"), INSTANCE_ROLE),
        (
            "ec2Configuration",
            resources.get("ec2Configuration"),
            [{"imageType": expected["image_type"]}],
        ),
    )
    for field, actual, wanted in resource_fields:
        if actual != wanted:
            return f"{field}: expected {wanted!r}, got {actual!r}"
    return None


def _queue_mismatch(
    queue: dict[str, Any], expected: dict[str, Any], environment_arn: str
) -> str | None:
    fields = (
        ("jobQueueName", queue.get("jobQueueName"), expected["name"]),
        ("state", queue.get("state"), "ENABLED"),
        ("status", queue.get("status"), "VALID"),
        ("priority", queue.get("priority"), expected["priority"]),
        (
            "computeEnvironmentOrder",
            queue.get("computeEnvironmentOrder"),
            [{"order": 1, "computeEnvironment": environment_arn}],
        ),
    )
    for field, actual, wanted in fields:
        if actual != wanted:
            return f"{field}: expected {wanted!r}, got {actual!r}"
    return None


def _create_environment(batch: Any, expected: dict[str, Any]) -> None:
    batch.create_compute_environment(
        computeEnvironmentName=expected["name"],
        type="MANAGED",
        state="ENABLED",
        computeResources={
            "type": "EC2",
            "minvCpus": 0,
            "maxvCpus": expected["max_vcpus"],
            "desiredvCpus": 0,
            "instanceTypes": [expected["instance_type"]],
            "subnets": list(SUBNETS),
            "securityGroupIds": list(SECURITY_GROUPS),
            "instanceRole": INSTANCE_ROLE,
            "ec2Configuration": [{"imageType": expected["image_type"]}],
        },
    )
    _wait_for(
        f"environment {expected['name']} to be VALID/ENABLED",
        lambda: _describe_environment(batch, expected["name"]),
        lambda item: item is not None
        and item.get("status") == "VALID"
        and item.get("state") == "ENABLED",
    )


def _create_queue(
    batch: Any, expected: dict[str, Any], environment_arn: str
) -> None:
    batch.create_job_queue(
        jobQueueName=expected["name"],
        state="ENABLED",
        priority=expected["priority"],
        computeEnvironmentOrder=[
            {"order": 1, "computeEnvironment": environment_arn}
        ],
    )
    _wait_for(
        f"queue {expected['name']} to be VALID/ENABLED",
        lambda: _describe_queue(batch, expected["name"]),
        lambda item: item is not None
        and item.get("status") == "VALID"
        and item.get("state") == "ENABLED",
    )


def _queue_has_active_jobs(batch: Any, name: str) -> bool:
    for state in ACTIVE_STATES:
        token: str | None = None
        while True:
            arguments = {"jobQueue": name, "jobStatus": state}
            if token is not None:
                arguments["nextToken"] = token
            response = batch.list_jobs(**arguments)
            if response.get("jobSummaryList", []):
                return True
            next_token = response.get("nextToken")
            if not isinstance(next_token, str) or not next_token:
                break
            token = next_token
    return False


def _all_queues(batch: Any) -> list[dict[str, Any]]:
    queues: list[dict[str, Any]] = []
    token: str | None = None
    while True:
        arguments = {}
        if token is not None:
            arguments["nextToken"] = token
        response = batch.describe_job_queues(**arguments)
        queues.extend(response.get("jobQueues", []))
        next_token = response.get("nextToken")
        if not isinstance(next_token, str) or not next_token:
            return queues
        token = next_token


def migrate(*, batch: Any, sts: Any, execute: bool) -> tuple[str, ...]:
    account = sts.get_caller_identity()["Account"]
    if account != ACCOUNT_ID:
        raise ValueError(f"account: expected {ACCOUNT_ID!r}, got {account!r}")
    region = batch.meta.region_name
    if region != REGION:
        raise ValueError(f"region: expected {REGION!r}, got {region!r}")

    actions: list[str] = []
    environment_arns: dict[str, str] = {}
    for expected in ENVIRONMENTS:
        name = expected["name"]
        environment = _describe_environment(batch, name)
        if environment is None:
            actions.append(f"create environment {name}")
            if execute:
                _create_environment(batch, expected)
                environment = _describe_environment(batch, name)
        else:
            mismatch = _environment_mismatch(environment, expected)
            if mismatch is not None:
                raise ValueError(f"environment {name} mismatch: {mismatch}")
            actions.append(f"reuse environment {name}")
        if environment is not None:
            environment_arns[name] = environment["computeEnvironmentArn"]
        else:
            environment_arns[name] = (
                f"arn:aws:batch:{REGION}:{ACCOUNT_ID}:compute-environment/{name}"
            )

    for expected in QUEUES:
        name = expected["name"]
        environment_arn = environment_arns[expected["environment"]]
        queue = _describe_queue(batch, name)
        if queue is None:
            actions.append(f"create queue {name}")
            if execute:
                _create_queue(batch, expected, environment_arn)
        else:
            mismatch = _queue_mismatch(queue, expected, environment_arn)
            if mismatch is not None:
                raise ValueError(f"queue {name} mismatch: {mismatch}")
            actions.append(f"reuse queue {name}")

    for name in OLD_QUEUES:
        queue = _describe_queue(batch, name)
        if queue is None:
            continue
        if _queue_has_active_jobs(batch, name):
            actions.append(f"skip active queue {name}")
            continue
        actions.append(f"delete queue {name}")
        if execute:
            if queue.get("state") != "DISABLED":
                batch.update_job_queue(jobQueue=name, state="DISABLED")
                _wait_for(
                    f"queue {name} to be DISABLED",
                    lambda name=name: _describe_queue(batch, name),
                    lambda item: item is not None
                    and item.get("status") == "VALID"
                    and item.get("state") == "DISABLED",
                )
            batch.delete_job_queue(jobQueue=name)
            _wait_for(
                f"queue {name} deletion",
                lambda name=name: _describe_queue(batch, name),
                lambda item: item is None,
            )

    referenced_environments = {
        binding.get("computeEnvironment")
        for queue in _all_queues(batch)
        for binding in queue.get("computeEnvironmentOrder", [])
    }
    for name in OLD_ENVIRONMENTS:
        environment = _describe_environment(batch, name)
        if environment is None:
            continue
        arn = environment.get("computeEnvironmentArn")
        if arn in referenced_environments:
            actions.append(f"skip referenced environment {name}")
            continue
        actions.append(f"delete environment {name}")
        if execute:
            if environment.get("state") != "DISABLED":
                batch.update_compute_environment(
                    computeEnvironment=name,
                    state="DISABLED",
                )
                _wait_for(
                    f"environment {name} to be DISABLED",
                    lambda name=name: _describe_environment(batch, name),
                    lambda item: item is not None
                    and item.get("status") == "VALID"
                    and item.get("state") == "DISABLED",
                )
            batch.delete_compute_environment(computeEnvironment=name)
            _wait_for(
                f"environment {name} deletion",
                lambda name=name: _describe_environment(batch, name),
                lambda item: item is None,
            )
    return tuple(actions)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args(argv)
    actions = migrate(
        batch=boto3.client("batch", region_name=REGION),
        sts=boto3.client("sts", region_name=REGION),
        execute=arguments.execute,
    )
    for action in actions:
        print(action)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
