from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
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
        purpose: ExecutionPurpose = ExecutionPurpose.DEV,
        status: str = "SUCCEEDED",
        stream: str = "stream/job",
    ) -> None:
        digest = IMAGE.rsplit("@sha256:", 1)[1]
        definition_name = (
            f"trainer-heavy-test-{purpose.value}-{profile}-{digest}"
        )
        definition_arn = (
            f"arn:aws:batch:{REGION}:{ACCOUNT_ID}:"
            f"job-definition/{definition_name}:1"
        )
        profile_spec = expected_topology().profiles[profile]
        resources = [
            {"type": kind, "value": value}
            for kind, value in profile_spec.resource_requirements
        ]
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
            "jobName": (
                f"trainer-heavy-test-{purpose.value}-{profile}-test-{'b' * 12}"
            ),
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


def test_one_exact_pytest_file_per_job_and_purpose_in_identity(
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
    assert jobs[0].queue_name == queue_for(ExecutionPurpose.RUN, "c7al").name
    assert all(
        str(call["jobName"]).startswith("trainer-heavy-test-run-c7al-")
        for call in batch.submit_job_calls
    )
    assert str(batch.register_job_definition_calls[0]["jobDefinitionName"]).startswith(
        "trainer-heavy-test-run-c7al-"
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
    assert evidence[0].purpose is ExecutionPurpose.DEV
    assert evidence[0].profile == "c7al"
    assert evidence[0].queue_name == "dev-cpu-c7al-queue"
    assert evidence[0].image == IMAGE


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
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "submit",
                "--purpose",
                "run",
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
