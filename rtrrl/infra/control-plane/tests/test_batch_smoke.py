from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from trainer_infra import batch_admin_cli
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
        for profile, environment in self.compute_environments.items():
            environment["ecsClusterArn"] = (
                f"arn:aws:ecs:{REGION}:{ACCOUNT_ID}:cluster/{profile}"
            )

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
        self.jobs[job_id] = {
            "jobId": job_id,
            "jobName": job_name,
            "jobQueue": kwargs["jobQueue"],
            "jobDefinition": definition_arn,
            "status": "SUCCEEDED",
            "attempts": [
                {
                    "container": {
                        "containerInstanceArn": container_instance_arn,
                        "exitCode": 0,
                    }
                }
            ],
            "container": {
                "image": container_properties["image"],
                "resourceRequirements": deepcopy(
                    container_properties["resourceRequirements"]
                ),
                "logStreamName": stream,
                "exitCode": 0,
            },
        }
        return response


class SmokeLogs(FakeLogs):
    def __init__(self) -> None:
        super().__init__()
        self.missing_l4: set[tuple[str, str]] = set()

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
        return {
            "events": [{"message": message} for message in messages],
            "nextForwardToken": "done",
        }


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

    def describe_instances(self, *, InstanceIds: list[str]) -> dict[str, object]:
        self.calls.append(list(InstanceIds))
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


@pytest.fixture
def fake_services() -> SmokeServices:
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
    )
    assert report.passed
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
    assert by_case[("dev", "g6x")].gpu_lines
    assert by_case[("run", "g6x")].jax_gpu_lines
    path = tmp_path / ".trainer/smoke/trainer-smoke-shared-queues.json"
    payload = json.loads(path.read_text())
    assert payload["passed"] is True
    assert len(payload["cases"]) == 8
    assert payload["job_definition_arns"] == list(report.job_definition_arns)
    assert payload["log_stream_names"] == list(report.log_stream_names)
    assert not list(path.parent.glob("*.tmp"))


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
    payload = json.loads(
        (tmp_path / ".trainer/smoke/trainer-smoke-shared-queues.json").read_text()
    )
    assert [case["job_id"] for case in payload["cases"][:3]] == [
        "job-1",
        "job-2",
        "job-3",
    ]


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
