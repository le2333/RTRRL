from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import pytest

from trainer_infra import heavy_test_cli
from trainer_infra.batch_topology import (
    ACCOUNT_ID,
    DEFAULT_AWS_NETWORK_SETTINGS,
    REGION,
    ExecutionPurpose,
    ProfileDriftError,
    expected_topology,
    queue_for,
)
from trainer_infra.heavy_tests import (
    AggregateJobFailure,
    HeavyTestRunner,
    JobEvidence,
    PartialSubmissionError,
    ResourceRequirement,
    SubmittedTestJob,
)

IMAGE = "007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@sha256:" + "a" * 64
REPOSITORY_ROOT = Path(__file__).parents[4]
BUILDER = REPOSITORY_ROOT / "infra" / "batch" / "heavy-tests" / "build-image.sh"


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
        self.queues = {
            key: {
                "jobQueueName": spec.name,
                "jobQueueArn": (
                    f"arn:aws:batch:{REGION}:{ACCOUNT_ID}:job-queue/{spec.name}"
                ),
                "state": "ENABLED",
                "status": "VALID",
                "priority": spec.priority,
                "computeEnvironmentOrder": [
                    self.binding(spec.compute_environments[0], order=1)
                ],
            }
            for key, spec in topology.queues.items()
        }
        self.job_definitions: list[dict[str, object]] = []
        self.jobs: dict[str, dict[str, object]] = {}
        self.register_job_definition_calls: list[dict[str, object]] = []
        self.submit_job_calls: list[dict[str, object]] = []
        self.describe_jobs_calls: list[list[str]] = []
        self.describe_compute_environment_calls: list[list[str]] = []
        self.describe_job_queue_calls: list[list[str]] = []

    def binding(self, profile: str, *, order: int) -> dict[str, object]:
        return {
            "order": order,
            "computeEnvironment": self.compute_environments[profile][
                "computeEnvironmentArn"
            ],
        }

    def queue(self, name: str) -> dict[str, object]:
        return next(
            queue for queue in self.queues.values() if queue["jobQueueName"] == name
        )

    def describe_compute_environments(
        self, *, computeEnvironments: list[str]
    ) -> dict[str, object]:
        self.describe_compute_environment_calls.append(computeEnvironments)
        return {
            "computeEnvironments": [
                deepcopy(environment)
                for environment in self.compute_environments.values()
                if environment["computeEnvironmentName"] in computeEnvironments
            ]
        }

    def describe_job_queues(self, *, jobQueues: list[str]) -> dict[str, object]:
        self.describe_job_queue_calls.append(jobQueues)
        return {
            "jobQueues": [
                deepcopy(queue)
                for queue in self.queues.values()
                if queue["jobQueueName"] in jobQueues
                or queue["jobQueueArn"] in jobQueues
            ]
        }

    def describe_job_definitions(self, **kwargs: object) -> dict[str, object]:
        name = kwargs["jobDefinitionName"]
        return {
            "jobDefinitions": [
                deepcopy(definition)
                for definition in self.job_definitions
                if definition["jobDefinitionName"] == name
            ]
        }

    def register_job_definition(self, **kwargs: object) -> dict[str, object]:
        self.register_job_definition_calls.append(deepcopy(kwargs))
        revision = 1 + sum(
            definition["jobDefinitionName"] == kwargs["jobDefinitionName"]
            for definition in self.job_definitions
        )
        name = str(kwargs["jobDefinitionName"])
        definition = {
            **deepcopy(kwargs),
            "revision": revision,
            "jobDefinitionArn": (
                f"arn:aws:batch:{REGION}:{ACCOUNT_ID}:"
                f"job-definition/{name}:{revision}"
            ),
        }
        self.job_definitions.append(definition)
        return deepcopy(definition)

    def submit_job(self, **kwargs: object) -> dict[str, str]:
        self.submit_job_calls.append(deepcopy(kwargs))
        job_id = f"job-{len(self.submit_job_calls)}"
        return {"jobId": job_id}

    def describe_jobs(self, *, jobs: list[str]) -> dict[str, object]:
        self.describe_jobs_calls.append(list(jobs))
        return {"jobs": [deepcopy(self.jobs[job_id]) for job_id in jobs]}

    def add_identity_job(
        self,
        job_id: str,
        profile: str,
        *,
        prefix: str = "trainer-heavy-test",
        purpose: ExecutionPurpose = ExecutionPurpose.DEV,
        status: str = "SUCCEEDED",
        stream: str = "stream/job",
    ) -> None:
        digest = IMAGE.rsplit("@sha256:", 1)[1]
        definition_name = f"{prefix}-{profile}-{digest}"
        definition_arn = (
            f"arn:aws:batch:{REGION}:{ACCOUNT_ID}:"
            f"job-definition/{definition_name}:1"
        )
        profile_spec = expected_topology().profiles[profile]
        resources = [
            {"type": kind, "value": value}
            for kind, value in profile_spec.resource_requirements
        ]
        if not any(
            definition["jobDefinitionArn"] == definition_arn
            for definition in self.job_definitions
        ):
            self.job_definitions.append(
                {
                    "jobDefinitionName": definition_name,
                    "jobDefinitionArn": definition_arn,
                    "revision": 1,
                    "type": "container",
                    "platformCapabilities": ["EC2"],
                    "containerProperties": {
                        "image": IMAGE,
                        "command": ["bash", "-lc", "exit 64"],
                        "resourceRequirements": deepcopy(resources),
                        "logConfiguration": {"logDriver": "awslogs"},
                    },
                }
            )
        queue = self.queues[f"{purpose.value}-{profile}"]
        self.jobs[job_id] = {
            "jobId": job_id,
            "jobName": f"{prefix}-{purpose.value}-{profile}-test-{'b' * 12}",
            "jobQueue": queue["jobQueueArn"],
            "jobDefinition": definition_arn,
            "status": status,
            "container": {
                "image": IMAGE,
                "resourceRequirements": resources,
                "logStreamName": stream,
            },
        }


class FakeLogs:
    def __init__(self, messages: dict[str, list[str]] | None = None) -> None:
        self.messages = messages or {}

    def get_log_events(self, **kwargs: object) -> dict[str, object]:
        stream = str(kwargs["logStreamName"])
        return {
            "events": [
                {"message": message} for message in self.messages.get(stream, [])
            ],
            "nextForwardToken": "done",
        }


@pytest.fixture
def services() -> tuple[FakeBatch, FakeSts, HeavyTestRunner]:
    batch = FakeBatch()
    sts = FakeSts()
    return batch, sts, HeavyTestRunner(
        batch, FakeLogs(), sts, sleep=lambda _: None
    )


@pytest.mark.parametrize(
    ("profile", "queue"),
    [
        ("c7am", "dev-cpu-c7am-queue"),
        ("c7al", "dev-cpu-c7al-queue"),
        ("c7ax", "dev-cpu-c7ax-queue"),
        ("g6x", "dev-gpu-queue"),
    ],
)
def test_heavy_tests_route_only_to_dev_queues(
    services: tuple[FakeBatch, FakeSts, HeavyTestRunner],
    profile: str,
    queue: str,
) -> None:
    batch, _, runner = services
    submitted = runner.submit(
        profile=profile,
        image=IMAGE,
        tests=["memo/tests/online_ac/test_eval_trace.py"],
    )
    assert submitted[0].purpose == ExecutionPurpose.DEV
    assert submitted[0].queue_name == queue
    assert batch.submit_job_calls[0]["jobQueue"] == batch.queue(queue)["jobQueueArn"]


def test_submit_validates_all_four_environments_and_eight_queues(
    services: tuple[FakeBatch, FakeSts, HeavyTestRunner],
) -> None:
    batch, sts, runner = services
    runner.submit(
        profile="c7al",
        image=IMAGE,
        tests=["memo/tests/online_ac/test_eval_trace.py"],
    )
    assert sts.calls == 1
    assert len(batch.describe_compute_environment_calls) == 4
    assert len(batch.describe_job_queue_calls) == 8


def test_submit_rejects_wrong_queue_binding_before_paid_work(
    services: tuple[FakeBatch, FakeSts, HeavyTestRunner],
) -> None:
    batch, _, runner = services
    batch.queue("dev-cpu-c7al-queue")["computeEnvironmentOrder"] = [
        batch.binding("c7ax", order=1)
    ]
    with pytest.raises(ProfileDriftError, match="computeEnvironmentOrder"):
        runner.submit(
            profile="c7al",
            image=IMAGE,
            tests=["memo/tests/online_ac/test_eval_trace.py"],
        )
    assert batch.register_job_definition_calls == []
    assert batch.submit_job_calls == []


def test_wait_validates_complete_topology_before_describing_jobs(
    services: tuple[FakeBatch, FakeSts, HeavyTestRunner],
) -> None:
    batch, _, runner = services
    batch.queue("run-gpu-queue")["priority"] = 10
    with pytest.raises(ProfileDriftError, match="priority"):
        runner.wait(["job-1"])
    assert len(batch.describe_compute_environment_calls) == 4
    assert len(batch.describe_job_queue_calls) == 8
    assert batch.describe_jobs_calls == []


def test_account_and_region_are_validated_before_submit() -> None:
    batch = FakeBatch()
    batch.meta.region_name = "us-east-1"
    runner = HeavyTestRunner(batch, FakeLogs(), FakeSts(), sleep=lambda _: None)
    with pytest.raises(ProfileDriftError, match=f"{ACCOUNT_ID}/{REGION}"):
        runner.submit(
            profile="c7ax",
            image=IMAGE,
            tests=["memo/tests/online_ac/test_eval_trace.py"],
        )
    assert batch.register_job_definition_calls == []


def test_one_exact_pytest_file_per_job_and_purpose_in_job_identity(
    services: tuple[FakeBatch, FakeSts, HeavyTestRunner],
) -> None:
    batch, _, runner = services
    jobs = runner.submit(
        purpose=ExecutionPurpose.RUN,
        profile="c7al",
        image=IMAGE,
        tests=[
            "memo/tests/online_ac/test_eval_trace.py",
            "memo/tests/online_ac/test_jit_contract.py",
        ],
    )
    assert len(jobs) == len(batch.submit_job_calls) == 2
    assert all(job.purpose is ExecutionPurpose.RUN for job in jobs)
    assert all(job.kind == "heavy-test" for job in jobs)
    assert all(job.name_prefix == "trainer-heavy-test" for job in jobs)
    assert jobs[0].queue_name == queue_for(ExecutionPurpose.RUN, "c7al").name
    assert all(
        str(call["jobName"]).startswith("trainer-heavy-test-run-c7al-")
        for call in batch.submit_job_calls
    )
    assert str(batch.register_job_definition_calls[0]["jobDefinitionName"]).startswith(
        "trainer-heavy-test-c7al-"
    )
    assert "-run-" not in str(
        batch.register_job_definition_calls[0]["jobDefinitionName"]
    )
    assert all(
        len(str(call["jobName"])) <= 128 for call in batch.submit_job_calls
    )
    assert (
        len(str(batch.register_job_definition_calls[0]["jobDefinitionName"])) <= 128
    )
    assert "test_eval_trace.py -q" in jobs[0].command_text
    assert "test_jit_contract.py -q" in jobs[1].command_text
    assert jobs[0].resource_requirements == (
        ResourceRequirement(type="MEMORY", value="3200"),
        ResourceRequirement(type="VCPU", value="2"),
    )


def test_wait_accepts_new_identity_and_digest_bound_evidence() -> None:
    batch = FakeBatch()
    batch.add_identity_job("job", "c7al", purpose=ExecutionPurpose.DEV)
    runner = HeavyTestRunner(
        batch,
        FakeLogs({"stream/job": ["Maximum resident set size (kbytes): 123"]}),
        FakeSts(),
        sleep=lambda _: None,
    )
    evidence = runner.wait(["job"])
    assert evidence[0].kind == "heavy-test"
    assert evidence[0].name_prefix == "trainer-heavy-test"
    assert evidence[0].purpose is ExecutionPurpose.DEV
    assert evidence[0].profile == "c7al"
    assert evidence[0].queue_name == "dev-cpu-c7al-queue"
    assert evidence[0].image == IMAGE


def test_trainer_smoke_prefix_round_trips_through_wait() -> None:
    batch = FakeBatch()
    runner = HeavyTestRunner(batch, FakeLogs(), FakeSts(), sleep=lambda _: None)
    submitted = runner.submit(
        profile="c7al",
        image=IMAGE,
        tests=["memo/tests/online_ac/test_eval_trace.py"],
        name_prefix="trainer-smoke",
    )
    assert submitted[0].kind == "smoke"
    assert submitted[0].name_prefix == "trainer-smoke"
    assert str(batch.submit_job_calls[0]["jobName"]).startswith(
        "trainer-smoke-dev-c7al-"
    )
    assert str(batch.register_job_definition_calls[0]["jobDefinitionName"]).startswith(
        "trainer-smoke-c7al-"
    )
    assert len(str(batch.submit_job_calls[0]["jobName"])) <= 128
    assert len(str(batch.register_job_definition_calls[0]["jobDefinitionName"])) <= 128

    batch.add_identity_job("smoke", "c7al", prefix="trainer-smoke")
    runner = HeavyTestRunner(
        batch,
        FakeLogs({"stream/job": ["Maximum resident set size (kbytes): 123"]}),
        FakeSts(),
        sleep=lambda _: None,
    )
    evidence = runner.wait(["smoke"])
    assert evidence[0].kind == "smoke"
    assert evidence[0].name_prefix == "trainer-smoke"


def test_dev_and_run_reuse_purpose_neutral_definition_revision() -> None:
    batch = FakeBatch()
    runner = HeavyTestRunner(batch, FakeLogs(), FakeSts(), sleep=lambda _: None)
    dev = runner.submit(
        purpose=ExecutionPurpose.DEV,
        profile="c7al",
        image=IMAGE,
        tests=["memo/tests/online_ac/test_eval_trace.py"],
        name_prefix="trainer-smoke",
    )[0]
    run = runner.submit(
        purpose=ExecutionPurpose.RUN,
        profile="c7al",
        image=IMAGE,
        tests=["memo/tests/online_ac/test_eval_trace.py"],
        name_prefix="trainer-smoke",
    )[0]

    assert dev.job_definition_arn == run.job_definition_arn
    assert dev.job_definition_revision == run.job_definition_revision
    assert len(batch.register_job_definition_calls) == 1
    assert str(batch.submit_job_calls[0]["jobName"]).startswith(
        "trainer-smoke-dev-c7al-"
    )
    assert str(batch.submit_job_calls[1]["jobName"]).startswith(
        "trainer-smoke-run-c7al-"
    )


@pytest.mark.parametrize(
    "prefix",
    [
        "trainer-other",
        "trainer_smoke",
        "Trainer-smoke",
        "trainer-smoke!",
        "x" * 129,
        "",
        None,
    ],
)
def test_submit_rejects_unapproved_or_invalid_prefix_before_aws(
    prefix: object,
) -> None:
    batch = FakeBatch()
    sts = FakeSts()
    runner = HeavyTestRunner(batch, FakeLogs(), sts)
    with pytest.raises(ValueError, match="name_prefix"):
        runner.submit(
            profile="c7al",
            image=IMAGE,
            tests=["memo/tests/online_ac/test_eval_trace.py"],
            name_prefix=prefix,  # type: ignore[arg-type]
        )
    assert sts.calls == 0
    assert batch.register_job_definition_calls == []
    assert batch.submit_job_calls == []


def test_wait_rejects_old_job_name_without_purpose() -> None:
    batch = FakeBatch()
    batch.add_identity_job("job", "c7ax")
    batch.jobs["job"]["jobName"] = f"trainer-heavy-test-c7ax-test-{'b' * 12}"
    runner = HeavyTestRunner(
        batch,
        FakeLogs({"stream/job": ["Maximum resident set size (kbytes): 123"]}),
        FakeSts(),
        evidence_max_attempts=1,
        sleep=lambda _: None,
    )
    with pytest.raises(AggregateJobFailure) as raised:
        runner.wait(["job"])
    assert any(
        "not a trainer-heavy-test" in error
        for error in raised.value.evidence[0].evidence_errors
    )


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("profile", "profile"),
        ("prefix", "kind/prefix"),
        ("queue", "jobQueue"),
        ("definition-digest-shape", "digest-bound"),
        ("definition-revision-shape", "digest-bound"),
        ("job-image", "job container image"),
        ("job-resources", "resourceRequirements"),
        ("definition-image", "image digest"),
        ("definition-resources", "container image/resources"),
    ],
)
def test_deterministic_identity_drift_fails_without_sleep(
    drift: str,
    message: str,
) -> None:
    batch = FakeBatch()
    batch.add_identity_job("job", "c7ax")
    job = batch.jobs["job"]
    container = job["container"]
    assert isinstance(container, dict)
    definition = batch.job_definitions[0]
    definition_container = definition["containerProperties"]
    assert isinstance(definition_container, dict)
    if drift == "profile":
        job["jobName"] = f"trainer-heavy-test-dev-c7al-test-{'b' * 12}"
        job["jobQueue"] = batch.queues["dev-c7al"]["jobQueueArn"]
    elif drift == "prefix":
        job["jobName"] = f"trainer-smoke-dev-c7ax-test-{'b' * 12}"
    elif drift == "queue":
        job["jobQueue"] = batch.queues["run-c7ax"]["jobQueueArn"]
    elif drift == "definition-digest-shape":
        job["jobDefinition"] = str(job["jobDefinition"]).replace("a" * 64, "abc")
    elif drift == "definition-revision-shape":
        job["jobDefinition"] = str(job["jobDefinition"]).rsplit(":", 1)[0] + ":0"
    elif drift == "job-image":
        container["image"] = IMAGE.replace("a" * 64, "b" * 64)
    elif drift == "job-resources":
        container["resourceRequirements"] = [
            {"type": "VCPU", "value": "4"},
            {"type": "MEMORY", "value": "9999"},
        ]
    elif drift == "definition-image":
        definition_container["image"] = IMAGE.replace("a" * 64, "b" * 64)
    else:
        definition_container["resourceRequirements"] = [
            {"type": "VCPU", "value": "4"},
            {"type": "MEMORY", "value": "9999"},
        ]
    sleeps: list[float] = []
    runner = HeavyTestRunner(
        batch,
        FakeLogs(),
        FakeSts(),
        evidence_max_attempts=3,
        retry_delay_seconds=7,
        sleep=sleeps.append,
    )
    with pytest.raises(AggregateJobFailure) as raised:
        runner.wait(["job"])
    assert any(
        message in error for error in raised.value.evidence[0].evidence_errors
    )
    assert sleeps == []


def test_missing_definition_is_retried_as_eventually_consistent() -> None:
    batch = FakeBatch()
    batch.add_identity_job("job", "c7ax")
    batch.job_definitions.clear()
    sleeps: list[float] = []
    runner = HeavyTestRunner(
        batch,
        FakeLogs({"stream/job": ["Maximum resident set size (kbytes): 123"]}),
        FakeSts(),
        evidence_max_attempts=3,
        retry_delay_seconds=7,
        sleep=sleeps.append,
    )
    with pytest.raises(AggregateJobFailure):
        runner.wait(["job"])
    assert sleeps == [7, 7]


def test_g6x_wait_requires_and_returns_complete_gpu_evidence() -> None:
    batch = FakeBatch()
    batch.add_identity_job("gpu", "g6x", stream="stream/gpu")
    runner = HeavyTestRunner(
        batch,
        FakeLogs(
            {
                "stream/gpu": [
                    "[CudaDevice(id=0)]",
                    "NVIDIA L4, 23034 MiB",
                    "Maximum resident set size (kbytes): 5678",
                ]
            }
        ),
        FakeSts(),
        sleep=lambda _: None,
    )
    evidence = runner.wait(["gpu"])
    assert evidence[0].jax_gpu_lines == ("[CudaDevice(id=0)]",)
    assert evidence[0].gpu_lines == ("NVIDIA L4, 23034 MiB",)
    assert evidence[0].maximum_rss_lines == (
        "Maximum resident set size (kbytes): 5678",
    )
    assert evidence[0].resource_requirements == (
        ResourceRequirement(type="GPU", value="1"),
        ResourceRequirement(type="MEMORY", value="12000"),
        ResourceRequirement(type="VCPU", value="4"),
    )


@pytest.mark.parametrize(
    "image", ["repo:latest", "repo@sha256:abc", "repo@sha256:" + "A" * 64]
)
def test_submit_requires_digest_bound_image(image: str) -> None:
    runner = HeavyTestRunner(FakeBatch(), FakeLogs(), FakeSts())
    with pytest.raises(ValueError, match="digest"):
        runner.submit(
            profile="c7ax",
            image=image,
            tests=["memo/tests/online_ac/test_eval_trace.py"],
        )


def test_partial_submission_retains_successful_job() -> None:
    class FailSecond(FakeBatch):
        def submit_job(self, **kwargs: object) -> dict[str, str]:
            if self.submit_job_calls:
                raise RuntimeError("rejected")
            return super().submit_job(**kwargs)

    batch = FailSecond()
    runner = HeavyTestRunner(batch, FakeLogs(), FakeSts(), sleep=lambda _: None)
    with pytest.raises(PartialSubmissionError) as raised:
        runner.submit(
            profile="c7ax",
            image=IMAGE,
            tests=[
                "memo/tests/online_ac/test_eval_trace.py",
                "memo/tests/online_ac/test_jit_contract.py",
            ],
        )
    assert [job.job_id for job in raised.value.submitted] == ["job-1"]


def test_gpu_submission_keeps_probe_and_resource_contract(
    services: tuple[FakeBatch, FakeSts, HeavyTestRunner],
) -> None:
    batch, _, runner = services
    job = runner.submit(
        profile="g6x",
        image=IMAGE,
        tests=["memo/tests/online_ac/test_jit_contract.py"],
    )[0]
    assert job.command_text.index("jax.devices()") < job.command_text.index(
        "/usr/bin/time -v"
    )
    resources = batch.register_job_definition_calls[0]["containerProperties"][
        "resourceRequirements"
    ]
    assert {"type": "GPU", "value": "1"} in resources


def test_heavy_test_cli_is_dev_only_and_accepts_all_profiles() -> None:
    parser = heavy_test_cli._parser()
    for forbidden in (("--purpose", "run"), ("--name-prefix", "trainer-smoke")):
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "submit",
                    *forbidden,
                    "--profile",
                    "c7ax",
                    "--image",
                    IMAGE,
                    "memo/tests/online_ac/test_eval_trace.py",
                ]
            )
    for profile in ("c7am", "c7al", "c7ax", "g6x"):
        parsed = parser.parse_args(
            [
                "submit",
                "--profile",
                profile,
                "--image",
                IMAGE,
                "memo/tests/online_ac/test_eval_trace.py",
            ]
        )
        assert parsed.profile == profile


def test_cli_constructs_all_clients_in_fixed_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def client(service: str, *, region_name: str) -> object:
        calls.append((service, region_name))
        return object()

    monkeypatch.setattr(heavy_test_cli.boto3, "client", client)
    heavy_test_cli._runner()
    assert calls == [
        ("batch", REGION),
        ("logs", REGION),
        ("sts", REGION),
    ]


def test_submit_cli_passes_dev_purpose(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Runner:
        def submit(self, **kwargs: object) -> tuple[SubmittedTestJob, ...]:
            assert kwargs["purpose"] is ExecutionPurpose.DEV
            return ()

    monkeypatch.setattr(heavy_test_cli, "_runner", Runner)
    assert (
        heavy_test_cli.main(
            [
                "submit",
                "--profile",
                "c7al",
                "--image",
                IMAGE,
                "memo/tests/online_ac/test_eval_trace.py",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == ""


def test_wait_cli_keeps_failure_evidence_machine_readable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    evidence = JobEvidence(
        job_id="job-1",
        status="FAILED",
        log_stream_name="stream/job-1",
        maximum_rss_lines=(),
        gpu_lines=(),
        log_lines=("assert False",),
    )

    class Runner:
        def wait(self, job_ids: list[str]) -> tuple[JobEvidence, ...]:
            raise AggregateJobFailure([evidence])

    monkeypatch.setattr(heavy_test_cli, "_runner", Runner)
    assert heavy_test_cli.main(["wait", "job-1"]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["log_lines"] == ["assert False"]
    assert json.loads(captured.err)["error"] == "aggregate_job_failure"


def test_c7al_builder_uses_cpu_base_image_path() -> None:
    builder = BUILDER.read_text()
    assert "c7am|c7al|c7ax)" in builder
    assert 'base_tag="memorax-rtrl-cpu"' in builder
    assert "profile must be one of: c7am, c7al, c7ax, g6x" in builder
    assert re.search(r"c7am\|c7al\|c7ax\).*memorax-rtrl-cpu", builder)


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def test_c7al_builder_success_stdout_is_one_json_document(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    digest = f"sha256:{'c' * 64}"
    _write_executable(
        fake_bin / "aws",
        f"""#!/usr/bin/env bash
set -eu
case "$*" in
  *get-login-password*) printf 'password\\n' ;;
  *batch-get-image*) printf '%s\\n' '{{"images":[{{"imageId":{{"imageDigest":"{digest}"}}}}],"failures":[]}}' ;;
  *) printf 'unexpected aws call: %s\\n' "$*" >&2; exit 1 ;;
esac
""",
    )
    _write_executable(
        fake_bin / "docker",
        """#!/usr/bin/env bash
set -eu
case "$1" in
  info) exit 0 ;;
  login) read -r password; printf 'login:%s\\n' "$password" >&2 ;;
  build|push|run) printf 'docker:%s\\n' "$1" >&2 ;;
  *) exit 1 ;;
esac
""",
    )
    result = subprocess.run(
        [str(BUILDER), "--profile", "c7al"],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "ACCOUNT_ID": ACCOUNT_ID,
            "ECR_RETRY_DELAY_SECONDS": "0",
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
    )
    assert len(result.stdout.splitlines()) == 1
    payload = json.loads(result.stdout)
    assert payload == {
        "tag": payload["tag"],
        "digest": digest,
        "image": f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/rtrrl@{digest}",
    }
    assert "heavy-test-c7al-" in payload["tag"]
    assert "docker:build" in result.stderr
    assert "docker:push" in result.stderr
    assert "docker:run" in result.stderr
