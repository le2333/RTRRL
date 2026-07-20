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


class FakeSts:
    def __init__(self, account: str = ACCOUNT_ID) -> None:
        self.account = account
        self.calls = 0

    def get_caller_identity(self) -> dict[str, str]:
        self.calls += 1
        return {"Account": self.account}


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
        self.failed_delete: str | None = None
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

    def create_job_queue(self, **kwargs: object) -> dict[str, str]:
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
        return {"jobQueueArn": str(self.queues[name]["jobQueueArn"])}

    def update_job_queue(self, **kwargs: object) -> dict[str, str]:
        call = deepcopy(kwargs)
        self.update_calls.append(call)
        self.mutation_calls.append(("update_job_queue", call))
        name = str(kwargs["jobQueue"])
        self.queues[name]["state"] = kwargs["state"]
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


def test_deploy_creates_exact_shared_bindings(
    fake_services: BatchAdminServices,
) -> None:
    report = deploy_queues(fake_services, execute=True)
    assert report.created == tuple(
        spec.name for spec in expected_topology().queues.values()
    )
    assert fake_services.batch.create_calls["run-cpu-c7al-queue"] == {
        "jobQueueName": "run-cpu-c7al-queue",
        "state": "ENABLED",
        "priority": 100,
        "computeEnvironmentOrder": [
            fake_services.batch.binding("c7al", order=1),
        ],
    }
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


def test_rollback_rechecks_paginated_jobs_and_keeps_referenced_queue(
    fake_services: BatchAdminServices,
) -> None:
    fake_services.batch.fail_create("run-gpu-queue")
    fake_services.batch.jobs[("dev-cpu-c7am-queue", "RUNNING")] = [
        {
            "jobId": "job-1",
            "jobName": "audit",
            "jobQueue": "dev-cpu-c7am-queue",
            "status": "RUNNING",
        }
    ]

    with pytest.raises(DeploymentError) as raised:
        deploy_queues(fake_services, execute=True)

    assert "dev-cpu-c7am-queue" not in raised.value.report.rolled_back
    assert any(
        "dev-cpu-c7am-queue" in error and "job" in error
        for error in raised.value.report.rollback_errors
    )
    assert "dev-cpu-c7am-queue" in fake_services.batch.queues


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
    fake_services: BatchAdminServices, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_services.batch.async_lifecycle = True
    fake_services.batch.fail_create("run-gpu-queue")
    monkeypatch.setattr("trainer_infra.batch_admin.time.sleep", lambda _: None)

    with pytest.raises(DeploymentError) as raised:
        deploy_queues(fake_services, execute=True)

    assert len(raised.value.report.rolled_back) == 7
    assert raised.value.report.rollback_errors == ()


def test_create_wait_is_bounded_and_rolls_back(
    fake_services: BatchAdminServices, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = fake_services.batch.describe_job_queues

    def never_visible(**kwargs: object) -> dict[str, object]:
        if kwargs.get("jobQueues"):
            return {"jobQueues": []}
        return original(**kwargs)

    fake_services.batch.describe_job_queues = never_visible
    monkeypatch.setattr("trainer_infra.batch_admin.time.sleep", lambda _: None)

    with pytest.raises(DeploymentError, match="timed out") as raised:
        deploy_queues(fake_services, execute=True)

    assert raised.value.report.created == ("dev-cpu-c7am-queue",)


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
