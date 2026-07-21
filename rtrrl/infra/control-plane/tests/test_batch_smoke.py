from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from trainer_infra import batch_admin_cli, batch_smoke
from trainer_infra.batch_smoke import SmokeServices, run_smoke, smoke_plan
from trainer_infra.batch_topology import (
    ACCOUNT_ID,
    REGION,
    expected_topology,
)

from test_heavy_tests import FakeBatch, FakeLogs, FakeSts, IMAGE

CPU_IMAGE = IMAGE
GPU_IMAGE = (
    "007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl-gpu@sha256:" + "b" * 64
)


class SmokeBatch(FakeBatch):
    def __init__(self) -> None:
        super().__init__()
        self.fail_submission_number: int | None = None
        self.malformed_register_response = False
        self.untrusted_register_arn = False
        self.actual_queue_override: str | None = None
        self.multiple_success_attempts = False
        self.no_success_attempt = False
        self.fail_definition_reads_after_registration = False
        for profile, environment in self.compute_environments.items():
            environment["ecsClusterArn"] = (
                f"arn:aws:ecs:{REGION}:{ACCOUNT_ID}:cluster/{profile}"
            )

    def register_job_definition(self, **kwargs: object) -> dict[str, object]:
        response = super().register_job_definition(**kwargs)
        if self.untrusted_register_arn:
            response["jobDefinitionArn"] = str(
                response["jobDefinitionArn"]
            ).replace(REGION, "us-east-1")
        if self.malformed_register_response:
            return {"jobDefinitionArn": response["jobDefinitionArn"]}
        return response

    def describe_job_definitions(self, **kwargs: object) -> dict[str, object]:
        if self.fail_definition_reads_after_registration and self.job_definitions:
            raise RuntimeError("definition read unavailable")
        return super().describe_job_definitions(**kwargs)

    def submit_job(self, **kwargs: object) -> dict[str, str]:
        next_number = len(self.submit_job_calls) + 1
        if self.fail_submission_number == next_number:
            raise RuntimeError("submission rejected")
        response = super().submit_job(**kwargs)
        job_id = response["jobId"]
        job_name = str(kwargs["jobName"])
        parts = job_name.split("-")
        purpose = parts[2]
        profile = parts[3]
        definition_arn = str(kwargs["jobDefinition"])
        definition = next(
            item
            for item in self.job_definitions
            if item["jobDefinitionArn"] == definition_arn
        )
        container_properties = definition["containerProperties"]
        assert isinstance(container_properties, dict)
        stream = f"stream/{purpose}/{profile}"
        container_instance_arn = (
            f"arn:aws:ecs:{REGION}:{ACCOUNT_ID}:"
            f"container-instance/{profile}/container-{purpose}-{profile}"
        )
        attempt = {
            "container": {
                "containerInstanceArn": container_instance_arn,
                "exitCode": 1 if self.no_success_attempt else 0,
                "logStreamName": stream,
            }
        }
        attempts = [attempt]
        if self.multiple_success_attempts:
            attempts.append(deepcopy(attempt))
        self.jobs[job_id] = {
            "jobId": job_id,
            "jobName": job_name,
            "jobQueue": self.actual_queue_override or kwargs["jobQueue"],
            "jobDefinition": definition_arn,
            "status": "SUCCEEDED",
            "attempts": attempts,
            "container": {
                "image": container_properties["image"],
                "resourceRequirements": deepcopy(
                    container_properties["resourceRequirements"]
                ),
                "logStreamName": "wrong/top-level/stream",
                "exitCode": 0,
            },
        }
        return response


class SmokeLogs(FakeLogs):
    def __init__(self) -> None:
        super().__init__()
        self.missing_l4: set[tuple[str, str]] = set()
        self.malformed_event = False
        self.cyclic_tokens = False

    def get_log_events(self, **kwargs: object) -> dict[str, object]:
        stream = str(kwargs["logStreamName"])
        _, purpose, profile = stream.split("/")
        messages = [
            f"trainer_smoke_profile={profile}",
            f"trainer_smoke_purpose={purpose}",
            "Maximum resident set size (kbytes): 1234",
        ]
        if profile == "g6x":
            messages.append("JAX devices: [CudaDevice(id=0)]")
            if (purpose, profile) not in self.missing_l4:
                messages.append("NVIDIA L4, 23034 MiB")
        token = kwargs.get("nextToken")
        if self.cyclic_tokens:
            next_token = {"done-a": "done-b", "done-b": "done-a"}.get(
                token, "done-a"
            )
            return {
                "events": [{"message": message} for message in messages],
                "nextForwardToken": next_token,
            }
        if token is not None and not self.cyclic_tokens:
            return {"events": [], "nextForwardToken": token}
        events: list[object] = [{"message": message} for message in messages]
        if self.malformed_event:
            events.append({"message": 123})
        return {"events": events, "nextForwardToken": "done"}


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeEcs:
    def __init__(self) -> None:
        self.instance_types = {
            profile: environment.instance_type
            for profile, environment in expected_topology().compute_environments.items()
        }
        self.calls: list[dict[str, object]] = []
        self.duplicate = False
        self.unexpected = False

    def describe_container_instances(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(deepcopy(kwargs))
        cluster = str(kwargs["cluster"])
        profile = cluster.rsplit("/", 1)[-1]
        requested = kwargs["containerInstances"]
        assert isinstance(requested, list)
        arn = str(requested[0])
        arn_profile = arn.split("/")[-2]
        if profile != arn_profile and not self.duplicate:
            return {"containerInstances": [], "failures": []}
        instances = [
                {
                    "containerInstanceArn": arn,
                    "ec2InstanceId": f"i-{arn_profile}",
                }
            ]
        if self.unexpected:
            instances.append(
                {
                    "containerInstanceArn": f"{arn}-unexpected",
                    "ec2InstanceId": "i-unexpected",
                }
            )
        return {
            "containerInstances": instances,
            "failures": [],
        }


class FakeEc2:
    def __init__(self, ecs: FakeEcs) -> None:
        self.ecs = ecs
        self.calls: list[list[str]] = []
        self.paginate = False

    def describe_instances(
        self,
        *,
        InstanceIds: list[str],
        NextToken: str | None = None,
    ) -> dict[str, object]:
        self.calls.append(list(InstanceIds))
        if self.paginate and NextToken is None:
            return {"Reservations": [], "NextToken": "page-2"}
        instance_id = InstanceIds[0]
        profile = instance_id.removeprefix("i-")
        return {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": instance_id,
                            "InstanceType": self.ecs.instance_types[profile],
                        }
                    ]
                }
            ]
        }


def _fake_services() -> SmokeServices:
    batch = SmokeBatch()
    logs = SmokeLogs()
    ecs = FakeEcs()
    return SmokeServices(
        batch=batch,
        logs=logs,
        sts=FakeSts(),
        ecs=ecs,
        ec2=FakeEc2(ecs),
    )


@pytest.fixture
def fake_services() -> SmokeServices:
    return _fake_services()


def test_smoke_matrix_is_exact_and_has_exact_resources() -> None:
    cases = smoke_plan(CPU_IMAGE, GPU_IMAGE)
    assert [(case.purpose.value, case.profile) for case in cases] == [
        ("dev", "c7am"),
        ("dev", "c7al"),
        ("dev", "c7ax"),
        ("run", "c7am"),
        ("run", "c7al"),
        ("run", "c7ax"),
        ("dev", "g6x"),
        ("run", "g6x"),
    ]
    assert len({case.smoke_name for case in cases}) == 8
    assert all(case.smoke_name.startswith("trainer-smoke-") for case in cases)
    assert cases[1].queue_name == "dev-cpu-c7al-queue"
    assert cases[1].resource_requirements == (("VCPU", "2"), ("MEMORY", "3200"))
    assert cases[-1].resource_requirements == (
        ("VCPU", "4"),
        ("MEMORY", "12000"),
        ("GPU", "1"),
    )


def test_dry_run_has_zero_aws_mutations_and_writes_nothing(
    fake_services: SmokeServices,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    report = run_smoke(
        fake_services,
        cpu_image=CPU_IMAGE,
        gpu_image=GPU_IMAGE,
    )
    assert report.execute is False
    assert report.failure_text == ""
    assert all(not case.evidence_errors for case in report.cases)
    assert len(report.cases) == 8
    assert fake_services.batch.register_job_definition_calls == []
    assert fake_services.batch.submit_job_calls == []
    assert not (tmp_path / ".trainer").exists()


def test_execute_collects_all_evidence_and_reuses_definition_revisions(
    fake_services: SmokeServices,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    report = run_smoke(
        fake_services,
        cpu_image=CPU_IMAGE,
        gpu_image=GPU_IMAGE,
        execute=True,
        sleep=lambda _: None,
    )
    assert report.passed, report.failure_text
    assert len(report.execution_scope) == 32
    assert report.execution_scope[12] == "4"
    assert report.execution_scope[16] in "89ab"
    assert all(character in "0123456789abcdef" for character in report.execution_scope)
    assert len(report.cases) == 8
    assert len(fake_services.batch.submit_job_calls) == 8
    assert len(fake_services.batch.register_job_definition_calls) == 4
    commands = {
        str(call["jobName"]): str(call["containerOverrides"])
        for call in fake_services.batch.submit_job_calls
    }
    assert all("trainer_smoke_profile=" in command for command in commands.values())
    assert all("trainer_smoke_purpose=" in command for command in commands.values())
    assert all(
        "nvidia-smi --query-gpu=name,memory.total" in command
        for name, command in commands.items()
        if "-g6x-" in name
    )
    by_case = {(item.purpose.value, item.profile): item for item in report.cases}
    plan = {(item.purpose.value, item.profile): item for item in smoke_plan(CPU_IMAGE, GPU_IMAGE)}
    for key, item in by_case.items():
        expected = plan[key]
        assert item.queue_name == expected.queue_name
        assert item.queue_arn.endswith(f"/{expected.queue_name}")
        assert item.image == expected.image
        assert item.resource_requirements == tuple(
            sorted(expected.resource_requirements)
        )
        assert item.log_stream_name == f"stream/{key[0]}/{key[1]}"
        assert item.profile_marker_lines == (f"trainer_smoke_profile={key[1]}",)
        assert item.purpose_marker_lines == (f"trainer_smoke_purpose={key[0]}",)
    for profile, instance_type in (
        ("c7am", "c7a.medium"),
        ("c7al", "c7a.large"),
        ("c7ax", "c7a.xlarge"),
        ("g6x", "g6.xlarge"),
    ):
        dev = by_case[("dev", profile)]
        run = by_case[("run", profile)]
        assert dev.job_definition_arn == run.job_definition_arn
        assert dev.job_definition_revision == run.job_definition_revision
        assert dev.instance_type == run.instance_type == instance_type
        assert dev.exit_code == run.exit_code == 0
        assert dev.maximum_rss_lines and run.maximum_rss_lines
    for purpose in ("dev", "run"):
        assert by_case[(purpose, "g6x")].gpu_lines
        assert by_case[(purpose, "g6x")].jax_gpu_lines
    path = tmp_path / ".trainer/smoke/trainer-smoke-shared-queues.json"
    payload = json.loads(path.read_text())
    assert payload["passed"] is True
    assert payload["execution_scope"] == report.execution_scope
    assert payload["owned_job_definition_arns"] == list(
        report.owned_job_definition_arns
    )
    assert len(payload["cases"]) == 8
    assert payload["job_definition_arns"] == list(report.job_definition_arns)
    assert payload["log_stream_names"] == list(report.log_stream_names)
    assert not list(path.parent.glob("*.tmp"))


def test_ec2_evidence_paginates_to_unique_instance(
    fake_services: SmokeServices,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert isinstance(fake_services.ec2, FakeEc2)
    fake_services.ec2.paginate = True
    report = run_smoke(
        fake_services,
        cpu_image=CPU_IMAGE,
        gpu_image=GPU_IMAGE,
        execute=True,
        sleep=lambda _: None,
    )
    assert report.passed
    assert len(fake_services.ec2.calls) == 16


def test_queue_observation_timestamp_is_after_topology_validation(
    fake_services: SmokeServices,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    captured = datetime(2026, 7, 20, 20, 0, tzinfo=timezone.utc)
    observed = datetime(2026, 7, 20, 20, 1, tzinfo=timezone.utc)
    values = iter((captured, observed))
    monkeypatch.setattr(batch_smoke, "_utc_now", lambda: next(values))
    report = run_smoke(
        fake_services,
        cpu_image=CPU_IMAGE,
        gpu_image=GPU_IMAGE,
        execute=True,
    )
    assert report.captured_at == captured
    assert report.queue_deployment_observed_at == observed


def test_wrong_instance_and_missing_gpu_evidence_fail_closed(
    fake_services: SmokeServices,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert isinstance(fake_services.ecs, FakeEcs)
    assert isinstance(fake_services.logs, SmokeLogs)
    fake_services.ecs.instance_types["c7al"] = "c7a.xlarge"
    fake_services.logs.missing_l4.add(("run", "g6x"))
    report = run_smoke(
        fake_services,
        cpu_image=CPU_IMAGE,
        gpu_image=GPU_IMAGE,
        execute=True,
    )
    assert not report.passed
    assert "instance type" in report.failure_text
    assert "NVIDIA L4" in report.failure_text


def test_ambiguous_ecs_match_fails_closed(
    fake_services: SmokeServices,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert isinstance(fake_services.ecs, FakeEcs)
    fake_services.ecs.duplicate = True
    report = run_smoke(
        fake_services,
        cpu_image=CPU_IMAGE,
        gpu_image=GPU_IMAGE,
        execute=True,
    )
    assert not report.passed
    assert "exactly one ECS container instance" in report.failure_text


def test_unexpected_ecs_match_fails_closed(
    fake_services: SmokeServices,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert isinstance(fake_services.ecs, FakeEcs)
    fake_services.ecs.unexpected = True
    report = run_smoke(
        fake_services,
        cpu_image=CPU_IMAGE,
        gpu_image=GPU_IMAGE,
        execute=True,
    )
    assert not report.passed
    assert "unexpected container instance" in report.failure_text


def test_partial_submission_report_retains_created_job_ids(
    fake_services: SmokeServices,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert isinstance(fake_services.batch, SmokeBatch)
    fake_services.batch.fail_submission_number = 4
    report = run_smoke(
        fake_services,
        cpu_image=CPU_IMAGE,
        gpu_image=GPU_IMAGE,
        execute=True,
    )
    assert not report.passed
    assert [case.job_id for case in report.cases[:3]] == ["job-1", "job-2", "job-3"]
    assert all(case.job_id is None for case in report.cases[3:])
    assert report.cases[3].job_definition_arn is not None
    assert report.cases[3].job_definition_revision == 1
    assert report.cases[3].job_definition_arn in report.owned_job_definition_arns
    payload = json.loads(
        (tmp_path / ".trainer/smoke/trainer-smoke-shared-queues.json").read_text()
    )
    assert [case["job_id"] for case in payload["cases"][:3]] == [
        "job-1",
        "job-2",
        "job-3",
    ]


def test_each_execution_uses_a_unique_definition_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    first = run_smoke(
        _fake_services(),
        cpu_image=CPU_IMAGE,
        gpu_image=GPU_IMAGE,
        execute=True,
    )
    second = run_smoke(
        _fake_services(),
        cpu_image=CPU_IMAGE,
        gpu_image=GPU_IMAGE,
        execute=True,
    )
    assert first.execution_scope != second.execution_scope
    assert set(first.job_definition_arns).isdisjoint(second.job_definition_arns)
    for report in (first, second):
        assert len(report.owned_job_definition_arns) == 4
        assert all(
            f"trainer-smoke-{report.execution_scope}-" in arn
            for arn in report.owned_job_definition_arns
        )


def test_scope_collision_is_retried_before_registration(
    fake_services: SmokeServices,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    first_scope = "11111111111141118111111111111111"
    second_scope = "22222222222242228222222222222222"
    digest = CPU_IMAGE.rsplit("@sha256:", 1)[1]
    assert isinstance(fake_services.batch, SmokeBatch)
    family = f"trainer-smoke-{first_scope}-c7am-{digest}"
    original_describe = fake_services.batch.describe_job_definitions
    collision_tokens: list[object] = []

    def paginated_describe(**kwargs: object) -> dict[str, object]:
        if kwargs["jobDefinitionName"] != family:
            return original_describe(**kwargs)
        collision_tokens.append(kwargs.get("nextToken"))
        if "nextToken" not in kwargs:
            return {"jobDefinitions": [], "nextToken": "collision-page"}
        assert kwargs["nextToken"] == "collision-page"
        return {"jobDefinitions": [{"jobDefinitionName": family}]}

    monkeypatch.setattr(
        fake_services.batch,
        "describe_job_definitions",
        paginated_describe,
    )
    scopes = iter((first_scope, second_scope))
    monkeypatch.setattr(
        batch_smoke, "_new_execution_scope", lambda: next(scopes)
    )
    report = run_smoke(
        fake_services,
        cpu_image=CPU_IMAGE,
        gpu_image=GPU_IMAGE,
        execute=True,
    )
    assert report.passed, report.failure_text
    assert report.execution_scope == second_scope
    assert collision_tokens == [None, "collision-page"]
    assert all(
        f"trainer-smoke-{second_scope}-" in str(call["jobDefinitionName"])
        for call in fake_services.batch.register_job_definition_calls
    )


def test_scope_collision_retry_is_bounded_before_mutation(
    fake_services: SmokeServices,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    colliding_scope = "11111111111141118111111111111111"
    digest = CPU_IMAGE.rsplit("@sha256:", 1)[1]
    assert isinstance(fake_services.batch, SmokeBatch)
    fake_services.batch.job_definitions.append(
        {
            "jobDefinitionName": (
                f"trainer-smoke-{colliding_scope}-c7am-{digest}"
            ),
            "jobDefinitionArn": (
                f"arn:aws:batch:{REGION}:{ACCOUNT_ID}:job-definition/"
                f"trainer-smoke-{colliding_scope}-c7am-{digest}:1"
            ),
        }
    )
    monkeypatch.setattr(
        batch_smoke, "_new_execution_scope", lambda: colliding_scope
    )
    report = run_smoke(
        fake_services,
        cpu_image=CPU_IMAGE,
        gpu_image=GPU_IMAGE,
        execute=True,
    )
    assert not report.passed
    assert "collision" in report.failure_text
    assert fake_services.batch.register_job_definition_calls == []
    assert fake_services.batch.submit_job_calls == []


def test_actual_queue_drift_is_recorded_not_replaced_by_plan(
    fake_services: SmokeServices,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert isinstance(fake_services.batch, SmokeBatch)
    fake_services.batch.actual_queue_override = (
        f"arn:aws:batch:{REGION}:{ACCOUNT_ID}:job-queue/run-gpu-queue"
    )
    report = run_smoke(
        fake_services,
        cpu_image=CPU_IMAGE,
        gpu_image=GPU_IMAGE,
        execute=True,
    )
    assert not report.passed
    first = report.cases[0]
    assert first.expected_queue_name == "dev-cpu-c7am-queue"
    assert first.queue_name == "run-gpu-queue"
    assert first.queue_arn.endswith("/run-gpu-queue")
    assert "queue mismatch" in report.failure_text


def test_definition_read_failure_keeps_registered_identity(
    fake_services: SmokeServices,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert isinstance(fake_services.batch, SmokeBatch)
    fake_services.batch.fail_definition_reads_after_registration = True
    report = run_smoke(
        fake_services,
        cpu_image=CPU_IMAGE,
        gpu_image=GPU_IMAGE,
        execute=True,
        sleep=lambda _: None,
    )
    assert not report.passed
    assert report.owned_job_definition_arns
    assert report.cases[0].job_definition_arn in report.owned_job_definition_arns
    assert report.cases[0].job_definition_revision == 1
    assert report.cases[0].job_id is None
    assert fake_services.batch.submit_job_calls == []
    assert "definition read unavailable" in report.failure_text


def test_malformed_register_response_fails_before_submit_and_writes_report(
    fake_services: SmokeServices,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert isinstance(fake_services.batch, SmokeBatch)
    fake_services.batch.malformed_register_response = True
    report = run_smoke(
        fake_services,
        cpu_image=CPU_IMAGE,
        gpu_image=GPU_IMAGE,
        execute=True,
    )
    assert not report.passed
    assert fake_services.batch.submit_job_calls == []
    assert "revision" in report.failure_text
    assert len(report.job_definition_arns) == 1
    assert report.job_definition_arns == report.owned_job_definition_arns
    assert report.cases[0].job_definition_arn == report.job_definition_arns[0]
    assert report.cases[0].job_definition_revision == 1
    assert report.untrusted_job_definition_identifiers == ()
    assert (tmp_path / ".trainer/smoke/trainer-smoke-shared-queues.json").is_file()


def test_untrusted_registration_arn_is_audit_only(
    fake_services: SmokeServices,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert isinstance(fake_services.batch, SmokeBatch)
    fake_services.batch.untrusted_register_arn = True
    report = run_smoke(
        fake_services,
        cpu_image=CPU_IMAGE,
        gpu_image=GPU_IMAGE,
        execute=True,
    )
    assert not report.passed
    assert report.job_definition_arns == ()
    assert report.owned_job_definition_arns == ()
    assert len(report.untrusted_job_definition_identifiers) == 1
    assert "us-east-1" in report.untrusted_job_definition_identifiers[0]


@pytest.mark.parametrize("attempt_mode", ("none", "multiple"))
def test_logs_require_one_successful_attempt(
    fake_services: SmokeServices,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attempt_mode: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert isinstance(fake_services.batch, SmokeBatch)
    if attempt_mode == "none":
        fake_services.batch.no_success_attempt = True
    else:
        fake_services.batch.multiple_success_attempts = True
    report = run_smoke(
        fake_services,
        cpu_image=CPU_IMAGE,
        gpu_image=GPU_IMAGE,
        execute=True,
    )
    assert not report.passed
    assert "exactly one successful attempt" in report.failure_text
    assert all(case.log_stream_name is None for case in report.cases)


@pytest.mark.parametrize("boundary", ("malformed-event", "token-cycle"))
def test_log_boundaries_fail_closed(
    fake_services: SmokeServices,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert isinstance(fake_services.logs, SmokeLogs)
    if boundary == "malformed-event":
        fake_services.logs.malformed_event = True
    else:
        fake_services.logs.cyclic_tokens = True
    report = run_smoke(
        fake_services,
        cpu_image=CPU_IMAGE,
        gpu_image=GPU_IMAGE,
        execute=True,
        sleep=lambda _: None,
    )
    assert not report.passed
    assert (
        "event message must be a string" in report.failure_text
        or "token cycle" in report.failure_text
    )


def test_global_deadline_stops_after_sdk_return_and_preserves_artifacts(
    fake_services: SmokeServices,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    clock = FakeClock()
    original = fake_services.batch.describe_jobs

    def slow_describe(*, jobs: list[str]) -> dict[str, object]:
        response = original(jobs=jobs)
        clock.now = 2.0
        return response

    fake_services.batch.describe_jobs = slow_describe
    report = run_smoke(
        fake_services,
        cpu_image=CPU_IMAGE,
        gpu_image=GPU_IMAGE,
        execute=True,
        timeout_seconds=1.0,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert not report.passed
    assert "deadline" in report.failure_text
    assert len(report.owned_job_definition_arns) == 4
    assert [case.job_id for case in report.cases] == [
        f"job-{index}" for index in range(1, 9)
    ]
    assert (tmp_path / ".trainer/smoke/trainer-smoke-shared-queues.json").is_file()


def test_deadline_after_registration_retains_owned_definition_on_case(
    fake_services: SmokeServices,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    clock = FakeClock()
    original = fake_services.batch.register_job_definition

    def slow_register(**kwargs: object) -> dict[str, object]:
        response = original(**kwargs)
        clock.now = 2.0
        return response

    fake_services.batch.register_job_definition = slow_register
    report = run_smoke(
        fake_services,
        cpu_image=CPU_IMAGE,
        gpu_image=GPU_IMAGE,
        execute=True,
        timeout_seconds=1.0,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert not report.passed
    assert len(report.owned_job_definition_arns) == 1
    assert report.cases[0].job_definition_arn == report.owned_job_definition_arns[0]
    assert report.cases[0].job_definition_revision == 1
    assert report.cases[0].job_id is None


@pytest.mark.parametrize("timeout", (0, -1, float("inf"), float("nan"), True))
def test_invalid_timeout_fails_before_aws(
    fake_services: SmokeServices,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timeout: object,
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="timeout_seconds"):
        run_smoke(
            fake_services,
            cpu_image=CPU_IMAGE,
            gpu_image=GPU_IMAGE,
            execute=True,
            timeout_seconds=timeout,  # type: ignore[arg-type]
        )
    assert fake_services.batch.register_job_definition_calls == []
    assert fake_services.batch.submit_job_calls == []


def test_cli_smoke_constructs_all_clients_in_fixed_region(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, str]] = []

    def client(service: str, *, region_name: str) -> object:
        calls.append((service, region_name))
        if service == "batch":
            return SmokeBatch()
        if service == "logs":
            return SmokeLogs()
        if service == "sts":
            return FakeSts()
        if service == "ecs":
            return FakeEcs()
        return SimpleNamespace()

    monkeypatch.setattr(batch_admin_cli.boto3, "client", client)
    assert (
        batch_admin_cli.main(
            ["smoke", "--cpu-image", CPU_IMAGE, "--gpu-image", GPU_IMAGE]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["execute"] is False
    assert calls == [
        ("batch", REGION),
        ("logs", REGION),
        ("sts", REGION),
        ("ecs", REGION),
        ("ec2", REGION),
    ]


def test_cli_execute_failure_is_strict_json_on_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    planned = run_smoke(
        SmokeServices(
            batch=SimpleNamespace(),
            logs=SimpleNamespace(),
            sts=SimpleNamespace(),
            ecs=SimpleNamespace(),
            ec2=SimpleNamespace(),
        ),
        cpu_image=CPU_IMAGE,
        gpu_image=GPU_IMAGE,
    )
    failed = replace(
        planned,
        execute=True,
        passed=False,
        failure_text="job failed",
    )
    monkeypatch.setattr(batch_admin_cli, "_smoke_services", lambda: object())
    monkeypatch.setattr(batch_admin_cli, "run_smoke", lambda *args, **kwargs: failed)
    assert (
        batch_admin_cli.main(
            [
                "smoke",
                "--cpu-image",
                CPU_IMAGE,
                "--gpu-image",
                GPU_IMAGE,
                "--execute",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "smoke_failed"
    assert payload["report"]["failure_text"] == "job failed"


@pytest.mark.parametrize(
    "image",
    ("repo:latest", "repo@sha256:abc", "repo@sha256:" + "A" * 64),
)
def test_smoke_rejects_non_digest_images_before_aws(
    fake_services: SmokeServices,
    image: str,
) -> None:
    with pytest.raises(ValueError, match="digest"):
        smoke_plan(image, GPU_IMAGE)
    assert fake_services.batch.register_job_definition_calls == []
    assert fake_services.batch.submit_job_calls == []
