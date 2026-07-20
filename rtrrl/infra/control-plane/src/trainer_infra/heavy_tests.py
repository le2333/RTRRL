from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
from pathlib import Path, PurePosixPath
import re
import shlex
import tempfile
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
    jax_gpu_lines: tuple[str, ...] = ()
    evidence_errors: tuple[str, ...] = ()


class AggregateJobFailure(RuntimeError):
    """Raised after all requested jobs finish when at least one did not succeed."""

    def __init__(self, evidence: Sequence[JobEvidence]) -> None:
        self.evidence = tuple(evidence)
        failed = ", ".join(
            f"{item.job_id}={item.status}"
            for item in self.evidence
            if item.status != "SUCCEEDED" or item.evidence_errors
        )
        super().__init__(f"heavy-test jobs did not all succeed: {failed}")


class PartialSubmissionError(RuntimeError):
    """Raised when a multi-file submission fails after one or more jobs exist."""

    def __init__(
        self,
        *,
        submitted: Sequence[SubmittedTestJob],
        failed_test: str,
        cause: Exception,
    ) -> None:
        self.submitted = tuple(submitted)
        self.failed_test = failed_test
        self.cause = f"{type(cause).__name__}: {cause}"
        super().__init__(
            f"submission failed for {failed_test!r} after "
            f"{len(self.submitted)} successful jobs: {self.cause}"
        )


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
_IMAGE_COMPONENT = r"[a-z0-9]+(?:[._-][a-z0-9]+)*"
_REGISTRY_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_REGISTRY = rf"(?:localhost|{_REGISTRY_LABEL}(?:\.{_REGISTRY_LABEL})*)(?::[0-9]+)?"
_DIGEST_IMAGE_RE = re.compile(
    rf"(?:{_REGISTRY}/)?{_IMAGE_COMPONENT}(?:/{_IMAGE_COMPONENT})*"
    r"@sha256:[0-9a-f]{64}"
)
_TERMINAL_JOB_STATES = frozenset({"SUCCEEDED", "FAILED"})
_LOG_GROUP = "/aws/batch/job"
_JOB_DEFINITION_COMMAND = ["bash", "-lc", "exit 64"]
_DEFAULT_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_DEFINITION_LOCK_DIR = (
    Path(tempfile.gettempdir()) / "trainer-heavy-test-definition-locks"
)
_EMPTY_CONTAINER_DEFAULTS = (
    "environment",
    "mountPoints",
    "secrets",
    "ulimits",
    "volumes",
)


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


def _validate_test_path(test_file: str, repository_root: Path) -> str:
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
            "test path must be an existing regular .py file below "
            f"memo/tests: {test_file!r}"
        )
    candidate = repository_root.joinpath(*parts)
    current = repository_root
    for part in parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"test path must not contain symlinks: {test_file!r}")
    if not candidate.is_file():
        raise ValueError(
            f"test path must be an existing regular .py file below memo/tests: {test_file!r}"
        )
    tests_root = (repository_root / "memo" / "tests").resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(tests_root)
    except ValueError as error:
        raise ValueError(
            f"resolved test path must remain below memo/tests: {test_file!r}"
        ) from error
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
    return f"trainer-heavy-test-{profile_name}-{digest}"


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
    return _canonical_container(container) == _canonical_container(expected_container)


def _canonical_container(container: Mapping[str, Any]) -> dict[str, Any]:
    canonical = dict(container)
    for field in _EMPTY_CONTAINER_DEFAULTS:
        if canonical.get(field) == []:
            canonical.pop(field)
    log_configuration = canonical.get("logConfiguration")
    if isinstance(log_configuration, Mapping):
        normalized_log_configuration = dict(log_configuration)
        if normalized_log_configuration.get("options") == {}:
            normalized_log_configuration.pop("options")
        if normalized_log_configuration.get("secretOptions") == []:
            normalized_log_configuration.pop("secretOptions")
        canonical["logConfiguration"] = normalized_log_configuration
    return canonical


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
        repository_root: Path = _DEFAULT_REPOSITORY_ROOT,
        definition_lock_dir: Path = _DEFAULT_DEFINITION_LOCK_DIR,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        poll_interval_seconds: float = 15.0,
        wait_timeout_seconds: float = 3600.0,
        evidence_max_attempts: int = 5,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        self._batch = batch
        self._logs = logs
        self._repository_root = repository_root.resolve(strict=True)
        self._definition_lock_dir = definition_lock_dir
        self._sleep = sleep
        self._monotonic = monotonic
        self._poll_interval_seconds = poll_interval_seconds
        self._wait_timeout_seconds = wait_timeout_seconds
        self._evidence_max_attempts = evidence_max_attempts
        self._retry_delay_seconds = retry_delay_seconds
        if wait_timeout_seconds <= 0:
            raise ValueError("wait timeout must be positive")
        if evidence_max_attempts < 1:
            raise ValueError("evidence attempts must be at least one")

    def _get_or_register_definition(
        self, profile_name: str, profile: HeavyTestProfile, image: str
    ) -> tuple[str, int]:
        name = _definition_name(profile_name, image)
        container = _container_properties(profile, image)
        self._definition_lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self._definition_lock_dir / f"{name}.lock"
        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            matching = self._find_matching_definitions(name, container)
            if matching:
                latest = max(matching, key=lambda item: item.get("revision", -1))
                return _job_definition_identity(latest)

            self._batch.register_job_definition(
                jobDefinitionName=name,
                type="container",
                platformCapabilities=["EC2"],
                containerProperties=container,
            )
            for attempt in range(self._evidence_max_attempts):
                matching = self._find_matching_definitions(name, container)
                if matching:
                    latest = max(
                        matching, key=lambda item: item.get("revision", -1)
                    )
                    return _job_definition_identity(latest)
                if attempt + 1 < self._evidence_max_attempts:
                    self._sleep(self._retry_delay_seconds)
            raise RuntimeError(
                "registered job definition could not be re-read as an exact match"
            )

    def _find_matching_definitions(
        self, name: str, container: Mapping[str, object]
    ) -> list[Mapping[str, Any]]:
        definitions: list[Mapping[str, Any]] = []
        token: str | None = None
        while True:
            arguments: dict[str, object] = {
                "jobDefinitionName": name,
                "status": "ACTIVE",
            }
            if token is not None:
                arguments["nextToken"] = token
            response = self._batch.describe_job_definitions(**arguments)
            definitions.extend(
                definition
                for definition in response.get("jobDefinitions", [])
                if isinstance(definition, Mapping)
            )
            next_token = response.get("nextToken")
            if not isinstance(next_token, str) or not next_token:
                break
            token = next_token
        return [
            definition
            for definition in definitions
            if _definition_matches(definition, container)
        ]

    def submit(
        self, *, profile: str, image: str, tests: Sequence[str]
    ) -> tuple[SubmittedTestJob, ...]:
        _validate_digest_image(image)
        test_files = tuple(
            _validate_test_path(test_file, self._repository_root) for test_file in tests
        )
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
            try:
                response = self._batch.submit_job(
                    jobName=f"trainer-heavy-test-{profile}-{stem}-{unique}"[:128],
                    jobQueue=validated.queue_arn,
                    jobDefinition=definition_arn,
                    containerOverrides={"command": ["bash", "-c", command]},
                )
                job_id = response.get("jobId")
                if not isinstance(job_id, str) or not job_id:
                    raise RuntimeError("Batch submit_job returned no jobId")
            except Exception as error:
                raise PartialSubmissionError(
                    submitted=submitted,
                    failed_test=test_file,
                    cause=error,
                ) from error
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

    def _describe_job_chunks(
        self, job_ids: Sequence[str]
    ) -> tuple[dict[str, Mapping[str, Any]], dict[str, list[str]]]:
        described: dict[str, Mapping[str, Any]] = {}
        errors: dict[str, list[str]] = {}
        for offset in range(0, len(job_ids), 100):
            chunk = list(job_ids[offset : offset + 100])
            last_error: Exception | None = None
            response: Mapping[str, Any] | None = None
            for attempt in range(self._evidence_max_attempts):
                try:
                    candidate = self._batch.describe_jobs(jobs=chunk)
                    if not isinstance(candidate, Mapping):
                        raise RuntimeError("Batch describe_jobs returned a non-mapping")
                    response = candidate
                    break
                except Exception as error:
                    last_error = error
                    if attempt + 1 < self._evidence_max_attempts:
                        self._sleep(self._retry_delay_seconds)
            if response is None:
                message = f"describe_jobs failed: {last_error}"
                for job_id in chunk:
                    errors.setdefault(job_id, []).append(message)
                continue
            for job in response.get("jobs", []):
                if isinstance(job, Mapping) and isinstance(job.get("jobId"), str):
                    described[str(job["jobId"])] = job
            for job_id in chunk:
                if job_id not in described:
                    errors.setdefault(job_id, []).append(
                        "Batch did not return the requested job"
                    )
        return described, errors

    def _wait_for_terminal_jobs(
        self, job_ids: Sequence[str], timeout_seconds: float
    ) -> tuple[list[Mapping[str, Any]], dict[str, list[str]]]:
        pending = set(job_ids)
        terminal: dict[str, Mapping[str, Any]] = {}
        last_seen: dict[str, Mapping[str, Any]] = {}
        evidence_errors: dict[str, list[str]] = {}
        deadline = self._monotonic() + timeout_seconds
        while pending:
            described, errors = self._describe_job_chunks(sorted(pending))
            for job_id, messages in errors.items():
                evidence_errors.setdefault(job_id, []).extend(messages)
            for job_id, job in described.items():
                last_seen[job_id] = job
                status = job.get("status")
                if status in _TERMINAL_JOB_STATES:
                    terminal[job_id] = job
                    pending.discard(job_id)
            if pending:
                if self._monotonic() >= deadline:
                    for job_id in pending:
                        evidence_errors.setdefault(job_id, []).append(
                            f"wait timeout after {timeout_seconds:g} seconds"
                        )
                        terminal[job_id] = last_seen.get(
                            job_id,
                            {"jobId": job_id, "status": "UNKNOWN"},
                        )
                    pending.clear()
                    break
                self._sleep(self._poll_interval_seconds)
        return [terminal[job_id] for job_id in job_ids], evidence_errors

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

    def _collect_job_evidence(
        self,
        job: Mapping[str, Any],
        initial_errors: Sequence[str],
    ) -> JobEvidence:
        errors = list(initial_errors)
        job_id = str(job["jobId"])
        status = str(job.get("status", "UNKNOWN"))
        latest = job
        identity = " ".join(
            str(latest.get(field, "")) for field in ("jobName", "jobDefinition")
        )
        is_gpu = "trainer-heavy-test-g6x-" in identity
        container = latest.get("container")
        if not isinstance(container, Mapping):
            container = {}
        stream = container.get("logStreamName")
        if not isinstance(stream, str):
            stream = None

        if status in _TERMINAL_JOB_STATES and stream is None:
            last_error: Exception | None = None
            for attempt in range(self._evidence_max_attempts):
                try:
                    response = self._batch.describe_jobs(jobs=[job_id])
                    refreshed = response.get("jobs", [])
                    if refreshed and isinstance(refreshed[0], Mapping):
                        latest = refreshed[0]
                        refreshed_container = latest.get("container")
                        if isinstance(refreshed_container, Mapping):
                            container = refreshed_container
                            candidate = container.get("logStreamName")
                            if isinstance(candidate, str) and candidate:
                                stream = candidate
                                break
                except Exception as error:
                    last_error = error
                if attempt + 1 < self._evidence_max_attempts:
                    self._sleep(self._retry_delay_seconds)
            if stream is None:
                detail = f": {last_error}" if last_error is not None else ""
                errors.append(f"CloudWatch log stream unavailable{detail}")

        log_lines: tuple[str, ...] = ()
        if stream is not None:
            last_error = None
            for attempt in range(self._evidence_max_attempts):
                try:
                    candidate_lines = self._read_log_lines(stream)
                    log_lines = candidate_lines
                    has_rss = any(
                        "Maximum resident set size (kbytes):" in line
                        for line in candidate_lines
                    )
                    has_jax_gpu = any(
                        "CudaDevice(" in line for line in candidate_lines
                    )
                    has_l4 = any(
                        line.strip().startswith("NVIDIA L4")
                        for line in candidate_lines
                    )
                    if candidate_lines and (
                        status != "SUCCEEDED"
                        or (has_rss and (not is_gpu or (has_jax_gpu and has_l4)))
                    ):
                        break
                except Exception as error:
                    last_error = error
                if attempt + 1 < self._evidence_max_attempts:
                    self._sleep(self._retry_delay_seconds)
            if last_error is not None and not log_lines:
                errors.append(f"CloudWatch logs unavailable: {last_error}")
            elif not log_lines:
                errors.append("CloudWatch logs unavailable: no events returned")

        maximum_rss_lines = tuple(
            line
            for line in log_lines
            if "Maximum resident set size (kbytes):" in line
        )
        gpu_lines = tuple(
            line
            for line in log_lines
            if line.strip().startswith("NVIDIA L4")
        )
        jax_gpu_lines = tuple(line for line in log_lines if "CudaDevice(" in line)
        if status == "SUCCEEDED" and not maximum_rss_lines:
            errors.append("SUCCEEDED job is missing maximum RSS evidence")
        if status == "SUCCEEDED" and is_gpu and not jax_gpu_lines:
            errors.append("SUCCEEDED g6x job is missing JAX GPU evidence")
        if status == "SUCCEEDED" and is_gpu and not gpu_lines:
            errors.append("SUCCEEDED g6x job is missing NVIDIA L4 evidence")

        return JobEvidence(
            job_id=job_id,
            status=status,
            log_stream_name=stream,
            maximum_rss_lines=maximum_rss_lines,
            gpu_lines=gpu_lines,
            log_lines=log_lines,
            status_reason=latest.get("statusReason")
            if isinstance(latest.get("statusReason"), str)
            else None,
            container_reason=container.get("reason")
            if isinstance(container.get("reason"), str)
            else None,
            exit_code=container.get("exitCode")
            if isinstance(container.get("exitCode"), int)
            else None,
            jax_gpu_lines=jax_gpu_lines,
            evidence_errors=tuple(errors),
        )

    def wait(
        self,
        job_ids: Sequence[str],
        *,
        timeout_seconds: float | None = None,
    ) -> tuple[JobEvidence, ...]:
        if not job_ids:
            raise ValueError("at least one job ID is required")
        effective_timeout = (
            self._wait_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        if effective_timeout <= 0:
            raise ValueError("wait timeout must be positive")
        jobs, initial_errors = self._wait_for_terminal_jobs(
            tuple(job_ids), effective_timeout
        )
        evidence = [
            self._collect_job_evidence(job, initial_errors.get(str(job["jobId"]), ()))
            for job in jobs
        ]
        result = tuple(evidence)
        if any(
            item.status != "SUCCEEDED" or item.evidence_errors for item in result
        ):
            raise AggregateJobFailure(result)
        return result
