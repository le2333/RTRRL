from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from types import SimpleNamespace

import pytest

from trainer_infra import batch_admin_cli
from trainer_infra.batch_admin import (
    NONTERMINAL_JOB_STATUSES,
    BatchAdminServices,
    DeploymentError,
    deploy_queues,
    inventory,
)
from trainer_infra.batch_topology import (
    ACCOUNT_ID,
    DEFAULT_AWS_NETWORK_SETTINGS,
    REGION,
    ProfileDriftError,
    expected_topology,
)

ROLLBACK_JOB_STATUSES = (
    "SUBMITTED",
    "PENDING",
    "RUNNABLE",
    "STARTING",
    "RUNNING",
    "SUCCEEDED",
    "FAILED",
)
DEFAULT_CREATE_RESPONSE = object()


class FakeSts:
    def __init__(self, account: str = ACCOUNT_ID) -> None:
        self.account = account
        self.calls = 0

    def get_caller_identity(self) -> dict[str, str]:
        self.calls += 1
        return {"Account": self.account}


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeBatch:
    def __init__(self) -> None:
        topology = expected_topology()
        network = DEFAULT_AWS_NETWORK_SETTINGS
        self.meta = SimpleNamespace(region_name=REGION)
        self.compute_environments = {
            key: {
                "computeEnvironmentName": spec.name,
                "computeEnvironmentArn": (
                    f"arn:aws:batch:{REGION}:{ACCOUNT_ID}:"
                    f"compute-environment/{spec.name}"
                ),
                "type": "MANAGED",
                "state": "ENABLED",
                "status": "VALID",
                "computeResources": {
                    "type": "EC2",
                    "minvCpus": 0,
                    "maxvCpus": spec.max_vcpus,
                    "instanceTypes": [spec.instance_type],
                    "subnets": list(network.subnets),
                    "securityGroupIds": list(network.security_group_ids),
                    "instanceRole": network.instance_role,
                    "ec2Configuration": [{"imageType": spec.ami_family}],
                },
            }
            for key, spec in topology.compute_environments.items()
        }
        self.queues: dict[str, dict[str, object]] = {}
        self.jobs: dict[tuple[str, str], list[dict[str, object]]] = {}
        self.create_calls: dict[str, dict[str, object]] = {}
        self.update_calls: list[dict[str, object]] = []
        self.delete_calls: list[dict[str, object]] = []
        self.mutation_calls: list[tuple[str, dict[str, object]]] = []
        self.batch_api_calls = 0
        self.failed_create: str | None = None
        self.create_response: object = DEFAULT_CREATE_RESPONSE
        self.failed_delete: str | None = None
        self.job_after_disable: tuple[str, str] | None = None
        self.list_job_calls: list[tuple[str, str, str | None]] = []
        self.async_lifecycle = False
        self._describe_counts: dict[str, int] = {}
        self._pending_delete: str | None = None

    def binding(self, key: str, *, order: int) -> dict[str, object]:
        return {
            "order": order,
            "computeEnvironment": self.compute_environments[key][
                "computeEnvironmentArn"
            ],
        }

    def add_queue(self, name: str, **overrides: object) -> None:
        spec = next(
            (
                item
                for item in expected_topology().queues.values()
                if item.name == name
            ),
            None,
        )
        key = spec.compute_environments[0] if spec is not None else "c7am"
        self.queues[name] = {
            "jobQueueName": name,
            "jobQueueArn": (
                f"arn:aws:batch:{REGION}:{ACCOUNT_ID}:job-queue/{name}"
            ),
            "state": "ENABLED",
            "status": "VALID",
            "priority": spec.priority if spec is not None else 1,
            "computeEnvironmentOrder": [self.binding(key, order=1)],
            **overrides,
        }

    def fail_create(self, name: str) -> None:
        self.failed_create = name

    def describe_compute_environments(
        self,
        *,
        computeEnvironments: list[str] | None = None,
        maxResults: int | None = None,
        nextToken: str | None = None,
    ) -> dict[str, object]:
        del maxResults
        self.batch_api_calls += 1
        values = list(self.compute_environments.values())
        if computeEnvironments is not None:
            values = [
                item
                for item in values
                if item["computeEnvironmentName"] in computeEnvironments
            ]
        return self._page("computeEnvironments", values, nextToken)

    def describe_job_queues(
        self,
        *,
        jobQueues: list[str] | None = None,
        maxResults: int | None = None,
        nextToken: str | None = None,
    ) -> dict[str, object]:
        del maxResults
        self.batch_api_calls += 1
        if jobQueues is not None and len(jobQueues) == 1:
            name = jobQueues[0]
            count = self._describe_counts.get(name, 0) + 1
            self._describe_counts[name] = count
            queue = self.queues.get(name)
            if queue is not None and count >= 2:
                if queue["status"] in {"CREATING", "UPDATING"}:
                    queue["status"] = "VALID"
                elif queue["status"] == "DELETING":
                    del self.queues[name]
                    self._pending_delete = None
        values = list(self.queues.values())
        if jobQueues is not None:
            values = [
                item
                for item in values
                if item["jobQueueName"] in jobQueues
                or item["jobQueueArn"] in jobQueues
            ]
        return self._page("jobQueues", values, nextToken)

    def list_jobs(
        self,
        *,
        jobQueue: str,
        jobStatus: str,
        maxResults: int | None = None,
        nextToken: str | None = None,
    ) -> dict[str, object]:
        del maxResults
        self.batch_api_calls += 1
        name = jobQueue.rsplit("/", 1)[-1]
        self.list_job_calls.append((name, jobStatus, nextToken))
        return self._page(
            "jobSummaryList", self.jobs.get((name, jobStatus), []), nextToken
        )

    @staticmethod
    def _page(
        field: str, values: list[dict[str, object]], token: str | None
    ) -> dict[str, object]:
        offset = int(token or "0")
        page = {field: deepcopy(values[offset : offset + 1])}
        if offset + 1 < len(values):
            page["nextToken"] = str(offset + 1)
        return page

    def create_job_queue(self, **kwargs: object) -> object:
        name = str(kwargs["jobQueueName"])
        call = deepcopy(kwargs)
        self.create_calls[name] = call
        self.mutation_calls.append(("create_job_queue", call))
        if name == self.failed_create:
            raise RuntimeError(f"failed create {name}")
        self.add_queue(name, **kwargs)
        if self.async_lifecycle:
            self.queues[name]["status"] = "CREATING"
            self._describe_counts[name] = 0
        if self.create_response is not DEFAULT_CREATE_RESPONSE:
            return deepcopy(self.create_response)
        return {
            "jobQueueName": name,
            "jobQueueArn": str(self.queues[name]["jobQueueArn"]),
        }

    def update_job_queue(self, **kwargs: object) -> dict[str, str]:
        call = deepcopy(kwargs)
        self.update_calls.append(call)
        self.mutation_calls.append(("update_job_queue", call))
        name = str(kwargs["jobQueue"])
        self.queues[name]["state"] = kwargs["state"]
        if self.job_after_disable == (name, "ANY"):
            raise AssertionError("test must specify a concrete job status")
        if (
            self.job_after_disable is not None
            and self.job_after_disable[0] == name
        ):
            status = self.job_after_disable[1]
            self.jobs[(name, status)] = [
                {
                    "jobId": "late-job-a",
                    "jobName": "late-a",
                    "jobQueue": name,
                    "status": status,
                },
                {
                    "jobId": "late-job-b",
                    "jobName": "late-b",
                    "jobQueue": name,
                    "status": status,
                },
            ]
        if self.async_lifecycle:
            self.queues[name]["status"] = "UPDATING"
            self._describe_counts[name] = 0
        return {"jobQueueArn": str(self.queues[name]["jobQueueArn"])}

    def delete_job_queue(self, **kwargs: object) -> None:
        call = deepcopy(kwargs)
        self.delete_calls.append(call)
        self.mutation_calls.append(("delete_job_queue", call))
        name = str(kwargs["jobQueue"])
        if name == self.failed_delete:
            raise RuntimeError(f"failed delete {name}")
        if self.queues[name]["status"] != "VALID":
            raise RuntimeError(f"queue {name} is not stable")
        if self.async_lifecycle:
            self.queues[name]["status"] = "DELETING"
            self._describe_counts[name] = 0
            self._pending_delete = name
        else:
            del self.queues[name]


@pytest.fixture
def fake_services() -> BatchAdminServices:
    return BatchAdminServices(batch=FakeBatch(), sts=FakeSts())


def test_inventory_paginates_resources_and_all_nonterminal_job_states(
    fake_services: BatchAdminServices,
) -> None:
    batch = fake_services.batch
    for spec in expected_topology().queues.values():
        batch.add_queue(spec.name)
    for index, status in enumerate(NONTERMINAL_JOB_STATUSES):
        batch.jobs[("dev-cpu-c7am-queue", status)] = [
            {
                "jobId": f"job-{index}-a",
                "jobName": f"name-{index}-a",
                "jobQueue": "dev-cpu-c7am-queue",
                "status": status,
            },
            {
                "jobId": f"job-{index}-b",
                "jobName": f"name-{index}-b",
                "jobQueue": "dev-cpu-c7am-queue",
                "status": status,
            },
        ]

    result = inventory(fake_services)

    assert len(result.compute_environments) == 4
    assert len(result.queues) == 8
    assert len(result.nonterminal_jobs) == 10
    assert {job.status for job in result.nonterminal_jobs} == set(
        NONTERMINAL_JOB_STATUSES
    )
    with pytest.raises(FrozenInstanceError):
        result.queues = ()


def test_inventory_rejects_wrong_account_before_batch_api(
    fake_services: BatchAdminServices,
) -> None:
    fake_services.sts.account = "123456789012"
    with pytest.raises(ProfileDriftError, match=f"{ACCOUNT_ID}/{REGION}"):
        inventory(fake_services)
    assert fake_services.batch.batch_api_calls == 0


def test_deploy_dry_run_has_no_mutation(
    fake_services: BatchAdminServices,
) -> None:
    report = deploy_queues(fake_services, execute=False)
    assert report.create_queues == (
        "dev-cpu-c7am-queue",
        "dev-cpu-c7al-queue",
        "dev-cpu-c7ax-queue",
        "run-cpu-c7am-queue",
        "run-cpu-c7al-queue",
        "run-cpu-c7ax-queue",
        "dev-gpu-queue",
        "run-gpu-queue",
    )
    assert report.created == ()
    assert report.topology_valid
    assert fake_services.batch.mutation_calls == []


@pytest.mark.parametrize(
    "queue_key",
    (
        "dev-c7am",
        "dev-c7al",
        "dev-c7ax",
        "run-c7am",
        "run-c7al",
        "run-c7ax",
        "dev-g6x",
        "run-g6x",
    ),
)
def test_deploy_creates_each_exact_shared_binding_without_tags(
    fake_services: BatchAdminServices, queue_key: str
) -> None:
    spec = expected_topology().queues[queue_key]
    report = deploy_queues(fake_services, execute=True)
    assert report.created == tuple(
        item.name for item in expected_topology().queues.values()
    )
    assert fake_services.batch.create_calls[spec.name] == {
        "jobQueueName": spec.name,
        "state": "ENABLED",
        "priority": spec.priority,
        "computeEnvironmentOrder": [
            fake_services.batch.binding(environment, order=index)
            for index, environment in enumerate(
                spec.compute_environments, start=1
            )
        ],
    }
    assert "tags" not in fake_services.batch.create_calls[spec.name]
    assert not any(
        operation.endswith("compute_environment")
        for operation, _ in fake_services.batch.mutation_calls
    )


def test_deploy_reuses_exact_existing_queue(
    fake_services: BatchAdminServices,
) -> None:
    fake_services.batch.add_queue("dev-cpu-c7am-queue")
    report = deploy_queues(fake_services, execute=True)
    assert report.reused == ("dev-cpu-c7am-queue",)
    assert "dev-cpu-c7am-queue" not in fake_services.batch.create_calls


@pytest.mark.parametrize(
    ("response", "path"),
    [
        (None, "create_job_queue"),
        ([], "create_job_queue"),
        ({}, "jobQueueName"),
        ({"jobQueueName": "dev-cpu-c7am-queue"}, "jobQueueArn"),
        (
            {
                "jobQueueName": "wrong-name",
                "jobQueueArn": (
                    f"arn:aws:batch:{REGION}:{ACCOUNT_ID}:"
                    "job-queue/dev-cpu-c7am-queue"
                ),
            },
            "jobQueueName",
        ),
        (
            {
                "jobQueueName": "dev-cpu-c7am-queue",
                "jobQueueArn": (
                    f"arn:aws:batch:us-east-1:{ACCOUNT_ID}:"
                    "job-queue/dev-cpu-c7am-queue"
                ),
            },
            "jobQueueArn",
        ),
        (
            {
                "jobQueueName": "dev-cpu-c7am-queue",
                "jobQueueArn": (
                    f"arn:aws:batch:{REGION}:123456789012:"
                    "job-queue/dev-cpu-c7am-queue"
                ),
            },
            "jobQueueArn",
        ),
        (
            {
                "jobQueueName": "dev-cpu-c7am-queue",
                "jobQueueArn": (
                    f"arn:aws:batch:{REGION}:{ACCOUNT_ID}:"
                    "compute-environment/dev-cpu-c7am-queue"
                ),
            },
            "jobQueueArn",
        ),
        (
            {
                "jobQueueName": "dev-cpu-c7am-queue",
                "jobQueueArn": (
                    f"arn:aws:batch:{REGION}:{ACCOUNT_ID}:"
                    "job-queue/run-cpu-c7am-queue"
                ),
            },
            "jobQueueArn",
        ),
    ],
)
def test_invalid_create_response_is_never_owned_or_rolled_back(
    fake_services: BatchAdminServices, response: object, path: str
) -> None:
    fake_services.batch.create_response = response

    with pytest.raises(DeploymentError, match=path) as raised:
        deploy_queues(fake_services, execute=True)

    assert raised.value.report.created == ()
    assert raised.value.report.rolled_back == ()
    assert fake_services.batch.update_calls == []
    assert fake_services.batch.delete_calls == []
    assert "dev-cpu-c7am-queue" in fake_services.batch.queues


def test_existing_drift_is_never_updated(
    fake_services: BatchAdminServices,
) -> None:
    fake_services.batch.add_queue("dev-cpu-c7am-queue", priority=99)
    with pytest.raises(ProfileDriftError, match="priority"):
        deploy_queues(fake_services, execute=True)
    assert fake_services.batch.update_calls == []
    assert fake_services.batch.mutation_calls == []


def test_partial_creation_removes_only_new_unreferenced_queues(
    fake_services: BatchAdminServices,
) -> None:
    fake_services.batch.add_queue("preexisting-unrelated", priority=7)
    fake_services.batch.fail_create("run-gpu-queue")

    with pytest.raises(DeploymentError) as raised:
        deploy_queues(fake_services, execute=True)

    expected = tuple(spec.name for spec in expected_topology().queues.values())[:-1]
    assert set(raised.value.report.rolled_back) == set(expected)
    assert raised.value.report.rollback_errors == ()
    assert "preexisting-unrelated" in fake_services.batch.queues
    assert all(call["jobQueue"] in expected for call in fake_services.batch.delete_calls)


@pytest.mark.parametrize("status", ROLLBACK_JOB_STATUSES)
def test_rollback_disables_then_rechecks_all_statuses_and_defers_late_job(
    fake_services: BatchAdminServices, status: str
) -> None:
    fake_services.batch.fail_create("run-gpu-queue")
    fake_services.batch.job_after_disable = ("dev-cpu-c7am-queue", status)

    with pytest.raises(DeploymentError) as raised:
        deploy_queues(fake_services, execute=True)

    assert "dev-cpu-c7am-queue" not in raised.value.report.rolled_back
    assert raised.value.report.rollback_deferred == ("dev-cpu-c7am-queue",)
    assert any(
        "dev-cpu-c7am-queue" in error
        and "deferred" in error
        and "job" in error
        for error in raised.value.report.rollback_errors
    )
    assert "dev-cpu-c7am-queue" in fake_services.batch.queues
    assert fake_services.batch.queues["dev-cpu-c7am-queue"]["state"] == "DISABLED"
    assert not any(
        call["jobQueue"] == "dev-cpu-c7am-queue"
        for call in fake_services.batch.delete_calls
    )
    calls = [
        job_status
        for name, job_status, _ in fake_services.batch.list_job_calls
        if name == "dev-cpu-c7am-queue"
    ]
    assert set(calls) == set(ROLLBACK_JOB_STATUSES)
    assert calls.count(status) == 2


def test_partial_rollback_error_is_preserved_in_audit_report(
    fake_services: BatchAdminServices,
) -> None:
    fake_services.batch.fail_create("run-gpu-queue")
    fake_services.batch.failed_delete = "dev-cpu-c7am-queue"

    with pytest.raises(DeploymentError) as raised:
        deploy_queues(fake_services, execute=True)

    assert any(
        "dev-cpu-c7am-queue" in error and "failed delete" in error
        for error in raised.value.report.rollback_errors
    )
    assert raised.value.__cause__ is not None


def test_rollback_waits_through_async_disable_and_delete_states(
    fake_services: BatchAdminServices,
) -> None:
    clock = FakeClock()
    fake_services.batch.async_lifecycle = True
    fake_services.batch.fail_create("run-gpu-queue")

    with pytest.raises(DeploymentError) as raised:
        deploy_queues(
            fake_services,
            execute=True,
            timeout_seconds=5.0,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

    assert len(raised.value.report.rolled_back) == 7
    assert raised.value.report.rollback_errors == ()


def test_create_wait_is_bounded_and_rolls_back(
    fake_services: BatchAdminServices,
) -> None:
    clock = FakeClock()
    original = fake_services.batch.describe_job_queues

    def never_visible(**kwargs: object) -> dict[str, object]:
        if kwargs.get("jobQueues"):
            return {"jobQueues": []}
        return original(**kwargs)

    fake_services.batch.describe_job_queues = never_visible

    with pytest.raises(DeploymentError, match="timed out") as raised:
        deploy_queues(
            fake_services,
            execute=True,
            timeout_seconds=2.0,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

    assert raised.value.report.created == ("dev-cpu-c7am-queue",)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_deploy_rejects_non_finite_or_non_positive_timeout_before_aws(
    fake_services: BatchAdminServices, timeout: object
) -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        deploy_queues(fake_services, execute=True, timeout_seconds=timeout)
    assert fake_services.sts.calls == 0
    assert fake_services.batch.batch_api_calls == 0


def test_wait_checks_absolute_deadline_after_aws_call(
    fake_services: BatchAdminServices,
) -> None:
    clock = FakeClock()
    original = fake_services.batch.describe_job_queues

    def slow_describe(**kwargs: object) -> dict[str, object]:
        response = original(**kwargs)
        requested = kwargs.get("jobQueues")
        if (
            requested == ["dev-cpu-c7am-queue"]
            and "dev-cpu-c7am-queue" in fake_services.batch.queues
        ):
            clock.now += 2.1
        return response

    fake_services.batch.describe_job_queues = slow_describe

    with pytest.raises(DeploymentError, match="timed out"):
        deploy_queues(
            fake_services,
            execute=True,
            timeout_seconds=2.0,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

    assert clock.sleeps == []


def test_wait_sleep_is_limited_to_remaining_deadline(
    fake_services: BatchAdminServices,
) -> None:
    clock = FakeClock()
    fake_services.batch.async_lifecycle = True
    original = fake_services.batch.describe_job_queues

    def slow_describe(**kwargs: object) -> dict[str, object]:
        response = original(**kwargs)
        requested = kwargs.get("jobQueues")
        if (
            requested == ["dev-cpu-c7am-queue"]
            and "dev-cpu-c7am-queue" in fake_services.batch.queues
        ):
            clock.now += 0.75
        return response

    fake_services.batch.describe_job_queues = slow_describe

    with pytest.raises(DeploymentError, match="timed out"):
        deploy_queues(
            fake_services,
            execute=True,
            timeout_seconds=1.0,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

    assert clock.sleeps[0] == pytest.approx(0.25)
    assert all(value <= 1.0 for value in clock.sleeps)


def test_cli_inventory_uses_fixed_region_and_prints_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    batch = FakeBatch()
    sts = FakeSts()
    calls: list[tuple[str, str]] = []

    def client(service: str, *, region_name: str) -> object:
        calls.append((service, region_name))
        return batch if service == "batch" else sts

    monkeypatch.setattr(batch_admin_cli.boto3, "client", client)
    assert batch_admin_cli.main(["inventory"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["queues"] == []
    assert calls == [("batch", REGION), ("sts", REGION)]


def test_cli_deploy_defaults_to_dry_run_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    batch = FakeBatch()
    sts = FakeSts()
    monkeypatch.setattr(
        batch_admin_cli.boto3,
        "client",
        lambda service, *, region_name: batch if service == "batch" else sts,
    )

    assert batch_admin_cli.main(["deploy"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["execute"] is False
    assert len(output["create_queues"]) == 8
    assert batch.mutation_calls == []


def test_cli_execute_creates_queues(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    batch = FakeBatch()
    sts = FakeSts()
    monkeypatch.setattr(
        batch_admin_cli.boto3,
        "client",
        lambda service, *, region_name: batch if service == "batch" else sts,
    )

    assert batch_admin_cli.main(["deploy", "--execute"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["execute"] is True
    assert len(output["created"]) == 8


def test_cli_emits_deployment_audit_report_on_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    batch = FakeBatch()
    batch.fail_create("run-gpu-queue")
    sts = FakeSts()
    monkeypatch.setattr(
        batch_admin_cli.boto3,
        "client",
        lambda service, *, region_name: batch if service == "batch" else sts,
    )

    assert batch_admin_cli.main(["deploy", "--execute"]) == 1
    error = json.loads(capsys.readouterr().err)
    assert error["error"] == "deployment_error"
    assert len(error["report"]["rolled_back"]) == 7
