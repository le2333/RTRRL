from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import math
import time
from types import MappingProxyType
from typing import Any, Callable

from trainer_infra.batch_topology import (
    ACCOUNT_ID,
    REGION,
    BatchTopology,
    BatchTopologyValidator,
    ProfileDriftError,
    QueueSpec,
    expected_topology,
    require_one,
    validate_queue,
)

NONTERMINAL_JOB_STATUSES = (
    "SUBMITTED",
    "PENDING",
    "RUNNABLE",
    "STARTING",
    "RUNNING",
)
_ALL_JOB_STATUSES = (*NONTERMINAL_JOB_STATUSES, "SUCCEEDED", "FAILED")
_PAGE_SIZE = 100
_DEFAULT_TIMEOUT_SECONDS = 60.0
_WAIT_SECONDS = 1.0


@dataclass(frozen=True)
class BatchAdminServices:
    batch: Any
    sts: Any


@dataclass(frozen=True)
class InventoryQueue:
    name: str
    arn: str
    state: str
    status: str
    priority: int
    compute_environments: tuple[str, ...]


@dataclass(frozen=True)
class InventoryComputeEnvironment:
    name: str
    arn: str
    state: str
    status: str


@dataclass(frozen=True)
class InventoryJob:
    job_id: str
    job_name: str
    queue: str
    status: str


@dataclass(frozen=True)
class BatchInventory:
    captured_at: datetime
    queues: tuple[InventoryQueue, ...]
    compute_environments: tuple[InventoryComputeEnvironment, ...]
    nonterminal_jobs: tuple[InventoryJob, ...]


@dataclass(frozen=True)
class DeploymentReport:
    execute: bool
    reused: tuple[str, ...]
    created: tuple[str, ...]
    create_queues: tuple[str, ...]
    rolled_back: tuple[str, ...]
    topology_valid: bool
    rollback_errors: tuple[str, ...] = ()
    rollback_deferred: tuple[str, ...] = ()


class DeploymentError(RuntimeError):
    def __init__(self, message: str, report: DeploymentReport) -> None:
        super().__init__(message)
        self.report = report


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfileDriftError(f"{path}: expected mapping, got {value!r}")
    return value


def _list(value: object, path: str) -> list[Any]:
    if type(value) is not list:
        raise ProfileDriftError(f"{path}: expected list, got {value!r}")
    return value


def _string(value: object, path: str) -> str:
    if type(value) is not str:
        raise ProfileDriftError(f"{path}: expected string, got {value!r}")
    return value


def _integer(value: object, path: str) -> int:
    if type(value) is not int:
        raise ProfileDriftError(f"{path}: expected integer, got {value!r}")
    return value


def _validated_timeout(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(
            "timeout_seconds must be a finite positive number, "
            f"got {value!r}"
        )
    return float(value)


def _new_deadline(timeout_seconds: float, monotonic: Callable[[], float]) -> float:
    now = monotonic()
    if not math.isfinite(now):
        raise ValueError(f"monotonic must return a finite number, got {now!r}")
    return now + timeout_seconds


def _remaining(
    deadline: float,
    monotonic: Callable[[], float],
    context: str,
) -> float:
    now = monotonic()
    if not math.isfinite(now):
        raise ValueError(f"monotonic must return a finite number, got {now!r}")
    remaining = deadline - now
    if remaining <= 0:
        raise TimeoutError(f"timed out {context}")
    return remaining


def _sleep_until_next_poll(
    deadline: float,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
    context: str,
) -> None:
    sleep(min(_WAIT_SECONDS, _remaining(deadline, monotonic, context)))


def _verify_identity(services: BatchAdminServices) -> None:
    identity = _mapping(services.sts.get_caller_identity(), "sts.get_caller_identity")
    account = identity.get("Account")
    meta = getattr(services.batch, "meta", None)
    region = getattr(meta, "region_name", None)
    if type(account) is not str or account != ACCOUNT_ID or region != REGION:
        raise ProfileDriftError(
            "sts.Account/batch.meta.region_name: expected "
            f"{ACCOUNT_ID}/{REGION}, got {account!r}/{region!r}"
        )


def _paginated(
    method: Any,
    *,
    result_key: str,
    path: str,
    arguments: Mapping[str, object] | None = None,
    deadline: float | None = None,
    monotonic: Callable[[], float] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    values: list[Mapping[str, Any]] = []
    token: str | None = None
    seen_tokens: set[str] = set()
    while True:
        request = dict(arguments or {})
        request["maxResults"] = _PAGE_SIZE
        if token is not None:
            request["nextToken"] = token
        raw_response = method(**request)
        if deadline is not None:
            if monotonic is None:
                raise ValueError("monotonic is required with deadline")
            _remaining(deadline, monotonic, f"waiting for {path}")
        response = _mapping(raw_response, path)
        page = _list(response.get(result_key), f"{path}.{result_key}")
        for index, value in enumerate(page):
            values.append(_mapping(value, f"{path}.{result_key}[{index}]"))
        next_token = response.get("nextToken")
        if next_token is None:
            return tuple(values)
        token = _string(next_token, f"{path}.nextToken")
        if not token or token in seen_tokens:
            raise ProfileDriftError(
                f"{path}.nextToken: expected a new non-empty token, got {token!r}"
            )
        seen_tokens.add(token)


def _inventory_queue(value: Mapping[str, Any]) -> InventoryQueue:
    entries = _list(value.get("computeEnvironmentOrder"), "computeEnvironmentOrder")
    environments = tuple(
        _string(
            _mapping(entry, f"computeEnvironmentOrder[{index}]").get(
                "computeEnvironment"
            ),
            f"computeEnvironmentOrder[{index}].computeEnvironment",
        )
        for index, entry in enumerate(entries)
    )
    return InventoryQueue(
        name=_string(value.get("jobQueueName"), "jobQueueName"),
        arn=_string(value.get("jobQueueArn"), "jobQueueArn"),
        state=_string(value.get("state"), "state"),
        status=_string(value.get("status"), "status"),
        priority=_integer(value.get("priority"), "priority"),
        compute_environments=environments,
    )


def _inventory_compute_environment(
    value: Mapping[str, Any],
) -> InventoryComputeEnvironment:
    return InventoryComputeEnvironment(
        name=_string(value.get("computeEnvironmentName"), "computeEnvironmentName"),
        arn=_string(value.get("computeEnvironmentArn"), "computeEnvironmentArn"),
        state=_string(value.get("state"), "state"),
        status=_string(value.get("status"), "status"),
    )


def _inventory_job(value: Mapping[str, Any], *, queue: str, status: str) -> InventoryJob:
    return InventoryJob(
        job_id=_string(value.get("jobId"), "jobId"),
        job_name=_string(value.get("jobName"), "jobName"),
        queue=_string(value.get("jobQueue", queue), "jobQueue"),
        status=_string(value.get("status", status), "status"),
    )


def inventory(services: BatchAdminServices) -> BatchInventory:
    _verify_identity(services)
    queue_values = _paginated(
        services.batch.describe_job_queues,
        result_key="jobQueues",
        path="describe_job_queues",
    )
    environment_values = _paginated(
        services.batch.describe_compute_environments,
        result_key="computeEnvironments",
        path="describe_compute_environments",
    )
    queues = tuple(sorted((_inventory_queue(value) for value in queue_values), key=lambda x: x.name))
    environments = tuple(
        sorted(
            (_inventory_compute_environment(value) for value in environment_values),
            key=lambda x: x.name,
        )
    )
    jobs: list[InventoryJob] = []
    for queue in queues:
        for status in NONTERMINAL_JOB_STATUSES:
            values = _paginated(
                services.batch.list_jobs,
                result_key="jobSummaryList",
                path=f"list_jobs[{queue.name},{status}]",
                arguments={"jobQueue": queue.arn, "jobStatus": status},
            )
            jobs.extend(
                _inventory_job(value, queue=queue.name, status=status)
                for value in values
            )
    return BatchInventory(
        captured_at=datetime.now(timezone.utc),
        queues=queues,
        compute_environments=environments,
        nonterminal_jobs=tuple(
            sorted(jobs, key=lambda x: (x.queue, x.status, x.job_id))
        ),
    )


def _describe_named_queue(
    services: BatchAdminServices, name: str
) -> Mapping[str, Any] | None:
    response = _mapping(
        services.batch.describe_job_queues(jobQueues=[name]),
        "describe_job_queues",
    )
    values = _list(
        response.get("jobQueues"), "describe_job_queues.jobQueues"
    )
    if not values:
        return None
    return require_one(
        values,
        kind="job queue",
        name=name,
        path="describe_job_queues.jobQueues",
    )


def _inspect_expected_resources(
    services: BatchAdminServices,
) -> tuple[dict[str, str], dict[str, Mapping[str, Any]]]:
    topology = expected_topology()
    compute_only = BatchTopology(
        compute_environments=topology.compute_environments,
        profiles=topology.profiles,
        queues=MappingProxyType({}),
    )
    validated = BatchTopologyValidator(
        services.batch, services.sts, topology=compute_only
    ).validate()
    environment_arns = dict(validated.compute_environment_arns)

    queues: dict[str, Mapping[str, Any]] = {}
    for key, expected in topology.queues.items():
        actual = _describe_named_queue(services, expected.name)
        if actual is not None:
            validate_queue(actual, expected, environment_arns)
            queues[key] = actual
    return environment_arns, queues


def _wait_for_valid_queue(
    services: BatchAdminServices,
    expected: QueueSpec,
    environment_arns: Mapping[str, str],
    *,
    deadline: float,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> None:
    last_status = "not visible"
    while True:
        actual = _describe_named_queue(services, expected.name)
        _remaining(
            deadline,
            monotonic,
            f"waiting for job queue {expected.name!r} to become VALID; "
            f"last status {last_status}",
        )
        if actual is not None:
            status = actual.get("status")
            last_status = repr(status)
            if status == "INVALID":
                raise ProfileDriftError(
                    f"job queue {expected.name!r} became INVALID"
                )
            if status == "VALID":
                validate_queue(actual, expected, environment_arns)
                return
        _sleep_until_next_poll(
            deadline,
            monotonic,
            sleep,
            f"waiting for job queue {expected.name!r} to become VALID; "
            f"last status {last_status}",
        )


def _create_arguments(
    expected: QueueSpec, environment_arns: Mapping[str, str]
) -> dict[str, object]:
    return {
        "jobQueueName": expected.name,
        "state": "ENABLED",
        "priority": expected.priority,
        "computeEnvironmentOrder": [
            {
                "order": index,
                "computeEnvironment": environment_arns[key],
            }
            for index, key in enumerate(expected.compute_environments, start=1)
        ],
    }


def _validate_create_response(value: object, expected: QueueSpec) -> str:
    response = _mapping(value, "create_job_queue")
    name = _string(response.get("jobQueueName"), "create_job_queue.jobQueueName")
    if name != expected.name:
        raise ProfileDriftError(
            "create_job_queue.jobQueueName: expected "
            f"{expected.name!r}, got {name!r}"
        )
    arn = _string(response.get("jobQueueArn"), "create_job_queue.jobQueueArn")
    expected_arn = (
        f"arn:aws:batch:{REGION}:{ACCOUNT_ID}:job-queue/{expected.name}"
    )
    if arn != expected_arn:
        raise ProfileDriftError(
            "create_job_queue.jobQueueArn: expected "
            f"{expected_arn!r}, got {arn!r}"
        )
    return arn


def _queue_has_any_jobs(
    services: BatchAdminServices,
    name: str,
    *,
    deadline: float,
    monotonic: Callable[[], float],
) -> bool:
    found = False
    for status in _ALL_JOB_STATUSES:
        jobs = _paginated(
            services.batch.list_jobs,
            result_key="jobSummaryList",
            path=f"rollback.list_jobs[{name},{status}]",
            arguments={"jobQueue": name, "jobStatus": status},
            deadline=deadline,
            monotonic=monotonic,
        )
        if jobs:
            found = True
    return found


def _wait_for_queue_state(
    services: BatchAdminServices,
    name: str,
    state: str,
    *,
    deadline: float,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> None:
    context = (
        f"waiting for job queue {name!r} state {state} and status VALID"
    )
    while True:
        actual = _describe_named_queue(services, name)
        _remaining(deadline, monotonic, context)
        if (
            actual is not None
            and actual.get("state") == state
            and actual.get("status") == "VALID"
        ):
            return
        _sleep_until_next_poll(deadline, monotonic, sleep, context)


def _wait_for_queue_deletion(
    services: BatchAdminServices,
    name: str,
    *,
    deadline: float,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> None:
    context = f"waiting for job queue {name!r} deletion"
    while True:
        if _describe_named_queue(services, name) is None:
            _remaining(deadline, monotonic, context)
            return
        _remaining(deadline, monotonic, context)
        _sleep_until_next_poll(deadline, monotonic, sleep, context)


def _rollback_created(
    services: BatchAdminServices,
    created: tuple[str, ...],
    *,
    timeout_seconds: float,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    rolled_back: list[str] = []
    errors: list[str] = []
    deferred: list[str] = []
    for name in reversed(created):
        try:
            deadline = _new_deadline(timeout_seconds, monotonic)
            services.batch.update_job_queue(jobQueue=name, state="DISABLED")
            _remaining(deadline, monotonic, f"disabling job queue {name!r}")
            _wait_for_queue_state(
                services,
                name,
                "DISABLED",
                deadline=deadline,
                monotonic=monotonic,
                sleep=sleep,
            )
            # AWS Batch exposes jobs by queue but no general reverse-reference API.
            # Ownership comes from the validated create response; this post-disable
            # seven-state scan is the final deletion safety gate.
            job_deadline = _new_deadline(timeout_seconds, monotonic)
            if _queue_has_any_jobs(
                services,
                name,
                deadline=job_deadline,
                monotonic=monotonic,
            ):
                deferred.append(name)
                errors.append(
                    f"{name}: deferred deletion: fresh post-disable job "
                    "inventory is not empty"
                )
                continue
            deadline = _new_deadline(timeout_seconds, monotonic)
            services.batch.delete_job_queue(jobQueue=name)
            _remaining(deadline, monotonic, f"deleting job queue {name!r}")
            _wait_for_queue_deletion(
                services,
                name,
                deadline=deadline,
                monotonic=monotonic,
                sleep=sleep,
            )
            rolled_back.append(name)
        except Exception as error:
            errors.append(f"{name}: {type(error).__name__}: {error}")
    return tuple(rolled_back), tuple(errors), tuple(deferred)


def deploy_queues(
    services: BatchAdminServices,
    *,
    execute: bool = False,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    monotonic: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> DeploymentReport:
    timeout = _validated_timeout(timeout_seconds)
    monotonic_fn = monotonic or time.monotonic
    sleep_fn = sleep or time.sleep
    environment_arns, existing = _inspect_expected_resources(services)
    topology = expected_topology()
    reused = tuple(
        expected.name for key, expected in topology.queues.items() if key in existing
    )
    create_queues = tuple(
        expected.name
        for key, expected in topology.queues.items()
        if key not in existing
    )
    report = DeploymentReport(
        execute=execute,
        reused=reused,
        created=(),
        create_queues=create_queues,
        rolled_back=(),
        topology_valid=True,
    )
    if not execute:
        return report

    created: list[str] = []
    try:
        for key, expected in topology.queues.items():
            if key in existing:
                continue
            deadline = _new_deadline(timeout, monotonic_fn)
            response = services.batch.create_job_queue(
                **_create_arguments(expected, environment_arns)
            )
            _validate_create_response(response, expected)
            created.append(expected.name)
            report = replace(report, created=tuple(created))
            _remaining(
                deadline,
                monotonic_fn,
                f"creating job queue {expected.name!r}",
            )
            _wait_for_valid_queue(
                services,
                expected,
                environment_arns,
                deadline=deadline,
                monotonic=monotonic_fn,
                sleep=sleep_fn,
            )
            _inspect_expected_resources(services)
        BatchTopologyValidator(services.batch, services.sts).validate()
        return report
    except Exception as error:
        rolled_back, rollback_errors, rollback_deferred = _rollback_created(
            services,
            tuple(created),
            timeout_seconds=timeout,
            monotonic=monotonic_fn,
            sleep=sleep_fn,
        )
        failed_report = replace(
            report,
            created=tuple(created),
            rolled_back=rolled_back,
            rollback_errors=rollback_errors,
            rollback_deferred=rollback_deferred,
            topology_valid=False,
        )
        raise DeploymentError(str(error), failed_report) from error
