from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import PurePosixPath
import re
import shlex
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence


class ProfileDriftError(RuntimeError):
    """Raised when an AWS Batch resource does not match its fixed profile."""


@dataclass(frozen=True)
class HeavyTestProfile:
    queue: str
    compute_environment: str
    instance_type: str
    vcpus: int
    memory_mib: int
    gpus: int
    gpu_model: str | None = None


@dataclass(frozen=True)
class ValidatedTestProfile:
    profile: HeavyTestProfile
    queue_arn: str
    compute_environment_arn: str


@dataclass(frozen=True)
class AwsNetworkSettings:
    subnets: tuple[str, ...]
    security_group_ids: tuple[str, ...]
    instance_role: str


@dataclass(frozen=True)
class SubmittedTestJob:
    job_id: str
    test_file: str
    profile: str
    image: str
    job_definition_arn: str
    job_definition_revision: int
    command_text: str


@dataclass(frozen=True)
class JobEvidence:
    job_id: str
    status: str
    log_stream_name: str | None
    maximum_rss_lines: tuple[str, ...]
    gpu_lines: tuple[str, ...]
    log_lines: tuple[str, ...]
    status_reason: str | None = None
    container_reason: str | None = None
    exit_code: int | None = None


class AggregateJobFailure(RuntimeError):
    """Raised after all requested jobs finish when at least one did not succeed."""

    def __init__(self, evidence: Sequence[JobEvidence]) -> None:
        self.evidence = tuple(evidence)
        failed = ", ".join(
            f"{item.job_id}={item.status}"
            for item in self.evidence
            if item.status != "SUCCEEDED"
        )
        super().__init__(f"heavy-test jobs did not all succeed: {failed}")


DEFAULT_AWS_NETWORK_SETTINGS = AwsNetworkSettings(
    subnets=(
        "subnet-08127d1c5d4de6ac2",
        "subnet-0b8c68ea0a9784758",
        "subnet-01a2aa195678f8411",
    ),
    security_group_ids=("sg-0c0ed6b927c5113dc",),
    instance_role=(
        "arn:aws:iam::007122174918:instance-profile/rtrrl-ecs-instance-role"
    ),
)


TEST_PROFILES: Mapping[str, HeavyTestProfile] = MappingProxyType(
    {
        "c7am": HeavyTestProfile(
            queue="rtrrl-cpu-c7am-queue",
            compute_environment="rtrrl-cpu-c7am-ce",
            instance_type="c7a.medium",
            vcpus=1,
            memory_mib=1600,
            gpus=0,
        ),
        "c7ax": HeavyTestProfile(
            queue="rtrrl-cpu-c7ax-queue",
            compute_environment="rtrrl-cpu-c7ax-ce",
            instance_type="c7a.xlarge",
            vcpus=4,
            memory_mib=7168,
            gpus=0,
        ),
        "g6x": HeavyTestProfile(
            queue="rtrrl-gpu-g6x-queue",
            compute_environment="rtrrl-gpu-g6x-ce",
            instance_type="g6.xlarge",
            vcpus=4,
            memory_mib=12000,
            gpus=1,
            gpu_model="NVIDIA L4",
        ),
    }
)

_COMPUTE_RESOURCE_FIELDS: Mapping[str, object] = MappingProxyType(
    {
        "type": "EC2",
        "minvCpus": 0,
        "maxvCpus": 32,
    }
)
_QUEUE_FIELDS: Mapping[str, object] = MappingProxyType(
    {
        "state": "ENABLED",
        "status": "VALID",
        "priority": 1,
    }
)
_DIGEST_IMAGE_RE = re.compile(r".+@sha256:[0-9a-f]{64}")
_TERMINAL_JOB_STATES = frozenset({"SUCCEEDED", "FAILED"})
_LOG_GROUP = "/aws/batch/job"
_JOB_DEFINITION_COMMAND = ["bash", "-lc", "exit 64"]


def _require_field(resource: Mapping[str, Any], field: str, expected: object) -> None:
    actual = resource.get(field)
    if actual != expected:
        raise ProfileDriftError(f"{field}: expected {expected!r}, got {actual!r}")


def _require_nonempty_string(resource: Mapping[str, Any], field: str) -> str:
    value = resource.get(field)
    if not isinstance(value, str) or not value:
        raise ProfileDriftError(
            f"{field}: expected non-empty string, got {value!r}"
        )
    return value


def _require_string_set(
    resource: Mapping[str, Any], field: str, expected: tuple[str, ...]
) -> None:
    actual = resource.get(field)
    if not isinstance(actual, list) or any(type(value) is not str for value in actual):
        raise ProfileDriftError(
            f"{field}: expected a list of strings, got {actual!r}"
        )
    if len(actual) != len(set(actual)):
        raise ProfileDriftError(
            f"{field}: duplicate values are not allowed: {actual!r}"
        )
    if len(expected) != len(set(expected)):
        raise ProfileDriftError(
            f"{field}: duplicate expected values are not allowed: {expected!r}"
        )
    if set(actual) != set(expected):
        raise ProfileDriftError(
            f"{field}: expected elements {expected!r}, got {actual!r}"
        )


def _describe_compute_environment(
    batch: Any, profile: HeavyTestProfile
) -> Mapping[str, Any] | None:
    response = batch.describe_compute_environments(
        computeEnvironments=[profile.compute_environment]
    )
    environments = response.get("computeEnvironments", [])
    return environments[0] if environments else None


def _describe_job_queue(batch: Any, profile: HeavyTestProfile) -> Mapping[str, Any] | None:
    response = batch.describe_job_queues(jobQueues=[profile.queue])
    queues = response.get("jobQueues", [])
    return queues[0] if queues else None


def _validate_compute_environment(
    environment: Mapping[str, Any] | None,
    profile: HeavyTestProfile,
    settings: AwsNetworkSettings,
) -> str:
    if environment is None:
        raise ProfileDriftError(
            f"missing compute environment {profile.compute_environment!r}"
        )

    _require_field(
        environment, "computeEnvironmentName", profile.compute_environment
    )
    _require_field(environment, "type", "MANAGED")
    _require_field(environment, "state", "ENABLED")
    _require_field(environment, "status", "VALID")
    resources = environment.get("computeResources")
    if not isinstance(resources, Mapping):
        raise ProfileDriftError(
            f"computeResources: expected mapping, got {resources!r}"
        )
    for field, expected in _COMPUTE_RESOURCE_FIELDS.items():
        _require_field(resources, field, expected)
    _require_field(resources, "instanceTypes", [profile.instance_type])
    _require_string_set(resources, "subnets", settings.subnets)
    _require_string_set(
        resources, "securityGroupIds", settings.security_group_ids
    )
    _require_field(resources, "instanceRole", settings.instance_role)

    return _require_nonempty_string(environment, "computeEnvironmentArn")


def _validate_job_queue(
    queue: Mapping[str, Any] | None,
    profile: HeavyTestProfile,
    compute_environment_arn: str,
) -> str:
    if queue is None:
        raise ProfileDriftError(f"missing job queue {profile.queue!r}")

    _require_field(queue, "jobQueueName", profile.queue)
    for field, expected in _QUEUE_FIELDS.items():
        _require_field(queue, field, expected)
    _require_field(
        queue,
        "computeEnvironmentOrder",
        [{"order": 1, "computeEnvironment": compute_environment_arn}],
    )

    return _require_nonempty_string(queue, "jobQueueArn")


def _get_profile(name: str) -> HeavyTestProfile:
    try:
        return TEST_PROFILES[name]
    except KeyError as error:
        expected = ", ".join(TEST_PROFILES)
        raise ValueError(
            f"unknown test profile {name!r}; expected one of: {expected}"
        ) from error


def validate_test_profile(
    batch: Any,
    name: str,
    *,
    settings: AwsNetworkSettings = DEFAULT_AWS_NETWORK_SETTINGS,
) -> ValidatedTestProfile:
    profile = _get_profile(name)
    compute_environment_arn = _validate_compute_environment(
        _describe_compute_environment(batch, profile), profile, settings
    )
    queue_arn = _validate_job_queue(
        _describe_job_queue(batch, profile),
        profile,
        compute_environment_arn,
    )
    return ValidatedTestProfile(
        profile=profile,
        queue_arn=queue_arn,
        compute_environment_arn=compute_environment_arn,
    )


def create_c7ax_if_missing(batch: Any, settings: AwsNetworkSettings) -> None:
    profile = TEST_PROFILES["c7ax"]
    environment = _describe_compute_environment(batch, profile)
    if environment is None:
        response = batch.create_compute_environment(
            computeEnvironmentName=profile.compute_environment,
            type="MANAGED",
            state="ENABLED",
            computeResources={
                "type": "EC2",
                "minvCpus": 0,
                "maxvCpus": 32,
                "desiredvCpus": 0,
                "instanceTypes": [profile.instance_type],
                "subnets": list(settings.subnets),
                "securityGroupIds": list(settings.security_group_ids),
                "instanceRole": settings.instance_role,
            },
        )
        compute_environment_arn = _require_nonempty_string(
            response, "computeEnvironmentArn"
        )
    else:
        compute_environment_arn = _validate_compute_environment(
            environment, profile, settings
        )

    queue = _describe_job_queue(batch, profile)
    if queue is None:
        batch.create_job_queue(
            jobQueueName=profile.queue,
            state="ENABLED",
            priority=1,
            computeEnvironmentOrder=[
                {
                    "order": 1,
                    "computeEnvironment": compute_environment_arn,
                }
            ],
        )
    else:
        _validate_job_queue(queue, profile, compute_environment_arn)


def _validate_digest_image(image: str) -> None:
    if _DIGEST_IMAGE_RE.fullmatch(image) is None:
        raise ValueError("image must be an exact lowercase sha256 digest reference")


def _validate_test_path(test_file: str) -> str:
    path = PurePosixPath(test_file)
    parts = path.parts
    if (
        path.is_absolute()
        or ".." in parts
        or len(parts) < 3
        or parts[:2] != ("memo", "tests")
        or path.suffix != ".py"
    ):
        raise ValueError(
            f"test path must be a Python file below memo/tests: {test_file!r}"
        )
    return path.as_posix()


def _resource_requirements(profile: HeavyTestProfile) -> list[dict[str, str]]:
    requirements = [
        {"type": "VCPU", "value": str(profile.vcpus)},
        {"type": "MEMORY", "value": str(profile.memory_mib)},
    ]
    if profile.gpus:
        requirements.append({"type": "GPU", "value": str(profile.gpus)})
    return requirements


def _container_properties(
    profile: HeavyTestProfile, image: str
) -> dict[str, object]:
    return {
        "image": image,
        "command": list(_JOB_DEFINITION_COMMAND),
        "resourceRequirements": _resource_requirements(profile),
        "logConfiguration": {"logDriver": "awslogs"},
    }


def _definition_name(profile_name: str, image: str) -> str:
    digest = image.rsplit("@sha256:", 1)[1]
    return f"rtrrl-heavy-test-{profile_name}-{digest}"


def _definition_matches(
    definition: Mapping[str, Any], expected_container: Mapping[str, object]
) -> bool:
    if definition.get("type") != "container":
        return False
    if definition.get("platformCapabilities") != ["EC2"]:
        return False
    container = definition.get("containerProperties")
    if not isinstance(container, Mapping):
        return False
    return all(container.get(key) == value for key, value in expected_container.items())


def _job_definition_identity(
    definition: Mapping[str, Any],
) -> tuple[str, int]:
    arn = definition.get("jobDefinitionArn")
    revision = definition.get("revision")
    if not isinstance(arn, str) or not arn:
        raise RuntimeError("Batch returned a job definition without an ARN")
    if not isinstance(revision, int):
        raise RuntimeError("Batch returned a job definition without a revision")
    return arn, revision


def _command_text(profile_name: str, test_file: str) -> str:
    pytest_command = " ".join(
        (
            "/usr/bin/time -v env",
            "XLA_PYTHON_CLIENT_PREALLOCATE=false",
            "MALLOC_ARENA_MAX=2",
            "python -m pytest",
            shlex.quote(test_file),
            "-q",
        )
    )
    if profile_name != "g6x":
        return pytest_command

    probe = (
        "python -c 'import jax; print(jax.devices())'"
        " && gpu_info=\"$(nvidia-smi --query-gpu=name,memory.total"
        " --format=csv,noheader)\""
        " && printf '%s\\n' \"$gpu_info\""
        " && printf '%s\\n' \"$gpu_info\" | grep -F 'NVIDIA L4' >/dev/null"
    )
    return f"{probe} && {pytest_command}"


class HeavyTestRunner:
    """Submit isolated pytest files and retain their Batch/CloudWatch evidence."""

    def __init__(
        self,
        batch: Any,
        logs: Any,
        *,
        sleep: Callable[[float], None] = time.sleep,
        poll_interval_seconds: float = 15.0,
    ) -> None:
        self._batch = batch
        self._logs = logs
        self._sleep = sleep
        self._poll_interval_seconds = poll_interval_seconds

    def _get_or_register_definition(
        self, profile_name: str, profile: HeavyTestProfile, image: str
    ) -> tuple[str, int]:
        name = _definition_name(profile_name, image)
        container = _container_properties(profile, image)
        response = self._batch.describe_job_definitions(
            jobDefinitionName=name,
            status="ACTIVE",
        )
        definitions = response.get("jobDefinitions", [])
        matching = [
            definition
            for definition in definitions
            if isinstance(definition, Mapping)
            and _definition_matches(definition, container)
        ]
        if matching:
            latest = max(matching, key=lambda item: item.get("revision", -1))
            return _job_definition_identity(latest)

        registered = self._batch.register_job_definition(
            jobDefinitionName=name,
            type="container",
            platformCapabilities=["EC2"],
            containerProperties=container,
        )
        return _job_definition_identity(registered)

    def submit(
        self, *, profile: str, image: str, tests: Sequence[str]
    ) -> tuple[SubmittedTestJob, ...]:
        _validate_digest_image(image)
        test_files = tuple(_validate_test_path(test_file) for test_file in tests)
        if not test_files:
            raise ValueError("at least one memo/tests file is required")
        validated = validate_test_profile(self._batch, profile)
        definition_arn, definition_revision = self._get_or_register_definition(
            profile, validated.profile, image
        )

        submitted = []
        for test_file in test_files:
            command = _command_text(profile, test_file)
            stem = re.sub(r"[^A-Za-z0-9_-]+", "-", PurePosixPath(test_file).stem)
            unique = hashlib.sha256(
                f"{time.time_ns()}:{test_file}".encode()
            ).hexdigest()[:12]
            response = self._batch.submit_job(
                jobName=f"heavy-{profile}-{stem}-{unique}"[:128],
                jobQueue=validated.queue_arn,
                jobDefinition=definition_arn,
                containerOverrides={"command": ["bash", "-c", command]},
            )
            job_id = response.get("jobId")
            if not isinstance(job_id, str) or not job_id:
                raise RuntimeError("Batch submit_job returned no jobId")
            submitted.append(
                SubmittedTestJob(
                    job_id=job_id,
                    test_file=test_file,
                    profile=profile,
                    image=image,
                    job_definition_arn=definition_arn,
                    job_definition_revision=definition_revision,
                    command_text=command,
                )
            )
        return tuple(submitted)

    def _wait_for_terminal_jobs(
        self, job_ids: Sequence[str]
    ) -> list[Mapping[str, Any]]:
        pending = set(job_ids)
        terminal: dict[str, Mapping[str, Any]] = {}
        while pending:
            response = self._batch.describe_jobs(jobs=sorted(pending))
            described = response.get("jobs", [])
            seen: set[str] = set()
            for job in described:
                if not isinstance(job, Mapping):
                    continue
                job_id = job.get("jobId")
                status = job.get("status")
                if not isinstance(job_id, str):
                    continue
                seen.add(job_id)
                if status in _TERMINAL_JOB_STATES:
                    terminal[job_id] = job
                    pending.discard(job_id)
            missing = pending - seen
            if missing:
                raise RuntimeError(
                    f"Batch did not return requested jobs: {', '.join(sorted(missing))}"
                )
            if pending:
                self._sleep(self._poll_interval_seconds)
        return [terminal[job_id] for job_id in job_ids]

    def _read_log_lines(self, stream: str | None) -> tuple[str, ...]:
        if stream is None:
            return ()
        lines: list[str] = []
        token: str | None = None
        while True:
            arguments: dict[str, object] = {
                "logGroupName": _LOG_GROUP,
                "logStreamName": stream,
                "startFromHead": True,
            }
            if token is not None:
                arguments["nextToken"] = token
            response = self._logs.get_log_events(**arguments)
            next_token = response.get("nextForwardToken")
            if token is not None and next_token == token:
                break
            lines.extend(
                str(event.get("message", ""))
                for event in response.get("events", [])
                if isinstance(event, Mapping)
            )
            if not isinstance(next_token, str):
                break
            token = next_token
        return tuple(lines)

    def wait(self, job_ids: Sequence[str]) -> tuple[JobEvidence, ...]:
        if not job_ids:
            raise ValueError("at least one job ID is required")
        jobs = self._wait_for_terminal_jobs(tuple(job_ids))
        evidence = []
        for job in jobs:
            container = job.get("container")
            if not isinstance(container, Mapping):
                container = {}
            stream = container.get("logStreamName")
            if not isinstance(stream, str):
                stream = None
            log_lines = self._read_log_lines(stream)
            evidence.append(
                JobEvidence(
                    job_id=str(job["jobId"]),
                    status=str(job.get("status", "UNKNOWN")),
                    log_stream_name=stream,
                    maximum_rss_lines=tuple(
                        line
                        for line in log_lines
                        if "Maximum resident set size (kbytes):" in line
                    ),
                    gpu_lines=tuple(
                        line for line in log_lines if "NVIDIA L4" in line
                    ),
                    log_lines=log_lines,
                    status_reason=job.get("statusReason")
                    if isinstance(job.get("statusReason"), str)
                    else None,
                    container_reason=container.get("reason")
                    if isinstance(container.get("reason"), str)
                    else None,
                    exit_code=container.get("exitCode")
                    if isinstance(container.get("exitCode"), int)
                    else None,
                )
            )
        result = tuple(evidence)
        if any(item.status != "SUCCEEDED" for item in result):
            raise AggregateJobFailure(result)
        return result
