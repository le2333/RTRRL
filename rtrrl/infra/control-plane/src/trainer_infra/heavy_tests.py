from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import itertools
import math
from pathlib import Path, PurePosixPath
import re
import shlex
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

from trainer_infra.batch_topology import (
    ACCOUNT_ID,
    REGION,
    BatchTopologyValidator,
    ExecutionPurpose,
    ResourceProfile,
    expected_topology,
    queue_for,
)


@dataclass(frozen=True)
class ResourceRequirement:
    type: str
    value: str


@dataclass(frozen=True)
class SubmittedTestJob:
    job_id: str
    test_file: str
    purpose: ExecutionPurpose
    kind: str
    name_prefix: str
    profile: str
    queue_name: str
    queue_arn: str
    image: str
    job_definition_arn: str
    job_definition_revision: int
    resource_requirements: tuple[ResourceRequirement, ...]
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
    purpose: ExecutionPurpose | None = None
    kind: str | None = None
    name_prefix: str | None = None
    profile: str | None = None
    queue_name: str | None = None
    queue_arn: str | None = None
    job_definition_arn: str | None = None
    job_definition_revision: int | None = None
    image: str | None = None
    resource_requirements: tuple[ResourceRequirement, ...] = ()


@dataclass(frozen=True)
class _JobIdentity:
    purpose: ExecutionPurpose
    kind: str
    name_prefix: str
    profile: str
    queue_name: str
    queue_arn: str
    job_definition_arn: str
    job_definition_revision: int
    image: str
    resource_requirements: tuple[ResourceRequirement, ...]


@dataclass(frozen=True)
class _ParsedJobIdentity:
    purpose: ExecutionPurpose
    kind: str
    name_prefix: str
    profile: str
    queue_name: str
    queue_arn: str
    definition_reference: str
    definition_name: str
    definition_revision: int
    image_digest: str
    image: str
    resource_requirements: tuple[ResourceRequirement, ...]


class _DefinitionNotReady(RuntimeError):
    """Raised only when definition evidence may still become visible."""


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


_IMAGE_COMPONENT = r"[a-z0-9]+(?:[._-][a-z0-9]+)*"
_REGISTRY_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_REGISTRY = rf"(?:localhost|{_REGISTRY_LABEL}(?:\.{_REGISTRY_LABEL})*)(?::[0-9]+)?"
_DIGEST_IMAGE_RE = re.compile(
    rf"(?:{_REGISTRY}/)?{_IMAGE_COMPONENT}(?:/{_IMAGE_COMPONENT})*"
    r"@sha256:[0-9a-f]{64}"
)
_ALLOWED_NAME_PREFIXES = {
    "trainer-heavy-test": "heavy-test",
    "trainer-smoke": "smoke",
}
_AWS_BATCH_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,128}")
_JOB_NAME_RE = re.compile(
    r"(trainer-(heavy-test|smoke))-(dev|run)-(c7am|c7al|c7ax|g6x)-"
    r"[A-Za-z0-9_-]+-[0-9a-f]{12}"
)
_JOB_DEFINITION_ARN_RE = re.compile(
    rf"arn:aws:batch:{REGION}:({ACCOUNT_ID}):job-definition/"
    r"((trainer-(heavy-test|smoke))-(dev|run)-(c7am|c7al|c7ax|g6x)-"
    r"([0-9a-f]{64})):([1-9][0-9]*)"
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


def _get_profile(name: str) -> ResourceProfile:
    try:
        return expected_topology().profiles[name]
    except KeyError as error:
        expected = ", ".join(expected_topology().profiles)
        raise ValueError(
            f"unknown test profile {name!r}; expected one of: {expected}"
        ) from error


def _validate_digest_image(image: str) -> None:
    if _DIGEST_IMAGE_RE.fullmatch(image) is None:
        raise ValueError("image must be an exact lowercase sha256 digest reference")


def _validate_aws_batch_name(name: str, *, field: str) -> None:
    if _AWS_BATCH_NAME_RE.fullmatch(name) is None:
        raise ValueError(
            f"{field} must contain only AWS Batch name characters and be at most "
            "128 characters"
        )


def _validate_name_prefix(name_prefix: str) -> str:
    if type(name_prefix) is not str or name_prefix not in _ALLOWED_NAME_PREFIXES:
        expected = ", ".join(_ALLOWED_NAME_PREFIXES)
        raise ValueError(
            f"name_prefix must be exactly one of: {expected}; got {name_prefix!r}"
        )
    _validate_aws_batch_name(name_prefix, field="name_prefix")
    return _ALLOWED_NAME_PREFIXES[name_prefix]


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


def _resource_requirements(profile: ResourceProfile) -> list[dict[str, str]]:
    return [
        {"type": requirement_type, "value": value}
        for requirement_type, value in profile.resource_requirements
    ]


def _typed_resource_requirements(
    profile: ResourceProfile,
) -> tuple[ResourceRequirement, ...]:
    return _normalize_resource_requirements(_resource_requirements(profile))


def _normalize_resource_requirements(
    value: object,
) -> tuple[ResourceRequirement, ...]:
    if not isinstance(value, list):
        raise RuntimeError(f"resourceRequirements must be a list, got {value!r}")
    normalized: list[ResourceRequirement] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise RuntimeError(f"invalid resource requirement: {item!r}")
        requirement_type = item.get("type")
        requirement_value = item.get("value")
        if not isinstance(requirement_type, str) or not isinstance(
            requirement_value, str
        ):
            raise RuntimeError(f"invalid resource requirement: {item!r}")
        normalized.append(
            ResourceRequirement(type=requirement_type, value=requirement_value)
        )
    if len({item.type for item in normalized}) != len(normalized):
        raise RuntimeError("duplicate resource requirement types")
    return tuple(sorted(normalized, key=lambda item: item.type))


def _container_properties(
    profile: ResourceProfile, image: str
) -> dict[str, object]:
    return {
        "image": image,
        "command": list(_JOB_DEFINITION_COMMAND),
        "resourceRequirements": _resource_requirements(profile),
        "logConfiguration": {"logDriver": "awslogs"},
    }


def _definition_name(
    name_prefix: str,
    purpose: ExecutionPurpose,
    profile_name: str,
    image: str,
) -> str:
    digest = image.rsplit("@sha256:", 1)[1]
    name = f"{name_prefix}-{purpose.value}-{profile_name}-{digest}"
    _validate_aws_batch_name(name, field="job definition name")
    return name


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
        sts: Any,
        *,
        repository_root: Path | None = None,
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
        self._topology_validator = BatchTopologyValidator(batch, sts)
        root = _DEFAULT_REPOSITORY_ROOT if repository_root is None else repository_root
        self._repository_root = root.resolve(strict=True)
        self._definition_lock_dir = definition_lock_dir
        self._sleep = sleep
        self._monotonic = monotonic
        self._poll_interval_seconds = poll_interval_seconds
        self._wait_timeout_seconds = self._validate_timeout(wait_timeout_seconds)
        self._evidence_max_attempts = evidence_max_attempts
        self._retry_delay_seconds = retry_delay_seconds
        self._job_name_sequence = itertools.count()
        if evidence_max_attempts < 1:
            raise ValueError("evidence attempts must be at least one")

    @staticmethod
    def _validate_timeout(timeout_seconds: float) -> float:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("wait timeout must be finite and positive")
        return timeout_seconds

    def _get_or_register_definition(
        self,
        name_prefix: str,
        purpose: ExecutionPurpose,
        profile_name: str,
        profile: ResourceProfile,
        image: str,
    ) -> tuple[str, int]:
        name = _definition_name(name_prefix, purpose, profile_name, image)
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
        self,
        *,
        profile: str,
        image: str,
        tests: Sequence[str],
        purpose: ExecutionPurpose = ExecutionPurpose.DEV,
        name_prefix: str = "trainer-heavy-test",
    ) -> tuple[SubmittedTestJob, ...]:
        kind = _validate_name_prefix(name_prefix)
        purpose = ExecutionPurpose(purpose)
        _validate_digest_image(image)
        test_files = tuple(
            _validate_test_path(test_file, self._repository_root) for test_file in tests
        )
        if not test_files:
            raise ValueError("at least one memo/tests file is required")
        profile_spec = _get_profile(profile)
        queue_spec = queue_for(purpose, profile)
        topology = self._topology_validator.validate()
        queue_arn = topology.queue_arns[f"{purpose.value}-{profile}"]
        definition_arn, definition_revision = self._get_or_register_definition(
            name_prefix, purpose, profile, profile_spec, image
        )

        submitted = []
        for test_file in test_files:
            command = _command_text(profile, test_file)
            stem = re.sub(r"[^A-Za-z0-9_-]+", "-", PurePosixPath(test_file).stem)
            unique = hashlib.sha256(
                (
                    f"{time.time_ns()}:{next(self._job_name_sequence)}:"
                    f"{test_file}"
                ).encode()
            ).hexdigest()[:12]
            prefix = f"{name_prefix}-{purpose.value}-{profile}-"
            suffix = f"-{unique}"
            stem = stem[: 128 - len(prefix) - len(suffix)]
            job_name = f"{prefix}{stem}{suffix}"
            _validate_aws_batch_name(job_name, field="job name")
            try:
                response = self._batch.submit_job(
                    jobName=job_name,
                    jobQueue=queue_arn,
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
                    purpose=purpose,
                    kind=kind,
                    name_prefix=name_prefix,
                    profile=profile,
                    queue_name=queue_spec.name,
                    queue_arn=queue_arn,
                    image=image,
                    job_definition_arn=definition_arn,
                    job_definition_revision=definition_revision,
                    resource_requirements=_typed_resource_requirements(
                        profile_spec
                    ),
                    command_text=command,
                )
            )
        return tuple(submitted)

    def _describe_job_chunks(
        self,
        job_ids: Sequence[str],
        *,
        deadline: float | None = None,
    ) -> tuple[dict[str, Mapping[str, Any]], dict[str, list[str]], bool]:
        described: dict[str, Mapping[str, Any]] = {}
        errors: dict[str, list[str]] = {}
        for offset in range(0, len(job_ids), 100):
            chunk = list(job_ids[offset : offset + 100])
            last_error: Exception | None = None
            response: Mapping[str, Any] | None = None
            for attempt in range(self._evidence_max_attempts):
                if deadline is not None and self._monotonic() >= deadline:
                    return described, errors, True
                try:
                    candidate = self._batch.describe_jobs(jobs=chunk)
                    if not isinstance(candidate, Mapping):
                        raise RuntimeError("Batch describe_jobs returned a non-mapping")
                    response = candidate
                    if deadline is not None and self._monotonic() >= deadline:
                        return described, errors, True
                    break
                except Exception as error:
                    last_error = error
                    if attempt + 1 < self._evidence_max_attempts:
                        retry_delay = self._retry_delay_seconds
                        if deadline is not None:
                            remaining = deadline - self._monotonic()
                            if remaining <= 0:
                                return described, errors, True
                            retry_delay = min(retry_delay, remaining)
                        self._sleep(retry_delay)
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
        return described, errors, False

    def _wait_for_terminal_jobs(
        self, job_ids: Sequence[str], timeout_seconds: float
    ) -> tuple[list[Mapping[str, Any]], dict[str, list[str]]]:
        pending = set(job_ids)
        terminal: dict[str, Mapping[str, Any]] = {}
        last_seen: dict[str, Mapping[str, Any]] = {}
        evidence_errors: dict[str, list[str]] = {}
        deadline = self._monotonic() + timeout_seconds

        def finish_timeout() -> None:
            for job_id in pending:
                evidence_errors.setdefault(job_id, []).append(
                    f"wait timeout after {timeout_seconds:g} seconds"
                )
                terminal[job_id] = last_seen.get(
                    job_id,
                    {"jobId": job_id, "status": "UNKNOWN"},
                )
            pending.clear()

        while pending:
            if self._monotonic() >= deadline:
                finish_timeout()
                break
            described, errors, deadline_reached = self._describe_job_chunks(
                sorted(pending), deadline=deadline
            )
            for job_id, messages in errors.items():
                evidence_errors.setdefault(job_id, []).extend(messages)
            for job_id, job in described.items():
                last_seen[job_id] = job
                status = job.get("status")
                if status in _TERMINAL_JOB_STATES:
                    terminal[job_id] = job
                    pending.discard(job_id)
            if deadline_reached and pending:
                finish_timeout()
                break
            if pending:
                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    finish_timeout()
                    break
                self._sleep(min(self._poll_interval_seconds, remaining))
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

    def _parse_job_identity(self, job: Mapping[str, Any]) -> _ParsedJobIdentity:
        job_name = job.get("jobName")
        if not isinstance(job_name, str):
            raise RuntimeError("jobName is missing")
        job_name_match = _JOB_NAME_RE.fullmatch(job_name)
        if job_name_match is None:
            raise RuntimeError(
                "jobName is not a trainer-heavy-test or trainer-smoke job: "
                f"{job_name!r}"
            )
        name_prefix, kind, purpose_text, profile_name = job_name_match.groups()
        purpose = ExecutionPurpose(purpose_text)
        if _ALLOWED_NAME_PREFIXES[name_prefix] != kind:
            raise RuntimeError("jobName kind does not match its approved prefix")
        profile = _get_profile(profile_name)
        queue = queue_for(purpose, profile_name)

        queue_reference = job.get("jobQueue")
        if not isinstance(queue_reference, str):
            raise RuntimeError("jobQueue is missing")
        queue_arn = (
            f"arn:aws:batch:{REGION}:{ACCOUNT_ID}:job-queue/{queue.name}"
        )
        if queue_reference not in {queue.name, queue_arn}:
            raise RuntimeError(
                f"jobQueue reference {queue_reference!r} does not match "
                f"{queue.name!r} / {queue_arn!r}"
            )

        definition_reference = job.get("jobDefinition")
        if not isinstance(definition_reference, str):
            raise RuntimeError("jobDefinition is missing")
        definition_match = _JOB_DEFINITION_ARN_RE.fullmatch(definition_reference)
        if definition_match is None:
            raise RuntimeError(
                f"jobDefinition ARN is not digest-bound: {definition_reference!r}"
            )
        (
            definition_account,
            definition_name,
            definition_prefix,
            definition_kind,
            definition_purpose,
            definition_profile,
            image_digest,
            revision_text,
        ) = definition_match.groups()
        if definition_account != ACCOUNT_ID:
            raise RuntimeError("jobDefinition AWS account does not match topology")
        if definition_prefix != name_prefix or definition_kind != kind:
            raise RuntimeError(
                "jobDefinition kind/prefix does not match the jobName kind/prefix"
            )
        if definition_purpose != purpose.value:
            raise RuntimeError(
                "jobDefinition purpose does not match the jobName purpose"
            )
        if definition_profile != profile_name:
            raise RuntimeError(
                "jobDefinition profile does not match the jobName profile"
            )
        definition_revision = int(revision_text)

        expected_resources = _typed_resource_requirements(profile)
        job_container = job.get("container")
        if not isinstance(job_container, Mapping):
            raise RuntimeError("job container details are missing")
        image = job_container.get("image")
        if not isinstance(image, str):
            raise RuntimeError("job container image is missing")
        try:
            _validate_digest_image(image)
        except ValueError as error:
            raise RuntimeError(f"job container image is invalid: {error}") from error
        if not image.endswith(f"@sha256:{image_digest}"):
            raise RuntimeError(
                "job container image does not match the jobDefinition digest"
            )
        job_resources = _normalize_resource_requirements(
            job_container.get("resourceRequirements")
        )
        if job_resources != expected_resources:
            raise RuntimeError(
                "job container resourceRequirements do not match the profile"
            )
        return _ParsedJobIdentity(
            purpose=purpose,
            kind=kind,
            name_prefix=name_prefix,
            profile=profile_name,
            queue_name=queue.name,
            queue_arn=queue_arn,
            definition_reference=definition_reference,
            definition_name=definition_name,
            definition_revision=definition_revision,
            image_digest=image_digest,
            image=image,
            resource_requirements=expected_resources,
        )

    def _load_job_definition(
        self, parsed: _ParsedJobIdentity
    ) -> _JobIdentity:
        definitions: list[Mapping[str, Any]] = []
        definition_token: str | None = None
        while True:
            definition_arguments: dict[str, object] = {
                "jobDefinitionName": parsed.definition_name,
            }
            if definition_token is not None:
                definition_arguments["nextToken"] = definition_token
            try:
                definition_response = self._batch.describe_job_definitions(
                    **definition_arguments
                )
            except Exception as error:
                raise _DefinitionNotReady(
                    f"jobDefinition query is not ready: {error}"
                ) from error
            definitions.extend(
                item
                for item in definition_response.get("jobDefinitions", [])
                if isinstance(item, Mapping)
            )
            next_definition_token = definition_response.get("nextToken")
            if (
                not isinstance(next_definition_token, str)
                or not next_definition_token
            ):
                break
            definition_token = next_definition_token
        matching_definitions = [
            item
            for item in definitions
            if item.get("jobDefinitionArn") == parsed.definition_reference
            and item.get("revision") == parsed.definition_revision
        ]
        if not matching_definitions:
            raise _DefinitionNotReady(
                "jobDefinition ARN/revision is not visible yet"
            )
        if len(matching_definitions) != 1:
            raise RuntimeError(
                "jobDefinition ARN/revision did not resolve to exactly one definition"
            )
        definition = matching_definitions[0]
        definition_container = definition.get("containerProperties")
        if not isinstance(definition_container, Mapping):
            raise RuntimeError("jobDefinition containerProperties are missing")
        image = definition_container.get("image")
        if not isinstance(image, str):
            raise RuntimeError("jobDefinition image is missing")
        try:
            _validate_digest_image(image)
        except ValueError as error:
            raise RuntimeError(f"jobDefinition image digest is invalid: {error}") from error
        if not image.endswith(f"@sha256:{parsed.image_digest}"):
            raise RuntimeError(
                "jobDefinition image digest does not match its definition name"
            )
        if image != parsed.image:
            raise RuntimeError(
                "job container image does not match the jobDefinition image"
            )
        profile = _get_profile(parsed.profile)
        expected_container = _container_properties(profile, image)
        if not _definition_matches(definition, expected_container):
            raise RuntimeError(
                "jobDefinition container image/resources do not match the profile"
            )
        return _JobIdentity(
            purpose=parsed.purpose,
            kind=parsed.kind,
            name_prefix=parsed.name_prefix,
            profile=parsed.profile,
            queue_name=parsed.queue_name,
            queue_arn=parsed.queue_arn,
            job_definition_arn=parsed.definition_reference,
            job_definition_revision=parsed.definition_revision,
            image=image,
            resource_requirements=parsed.resource_requirements,
        )

    def _resolve_job_identity(
        self, job: Mapping[str, Any]
    ) -> tuple[_JobIdentity | None, str | None]:
        try:
            parsed = self._parse_job_identity(job)
        except Exception as error:
            return None, f"job identity validation failed: {error}"
        last_error: Exception | None = None
        for attempt in range(self._evidence_max_attempts):
            try:
                return self._load_job_definition(parsed), None
            except _DefinitionNotReady as error:
                last_error = error
                if attempt + 1 < self._evidence_max_attempts:
                    self._sleep(self._retry_delay_seconds)
            except Exception as error:
                return None, f"job identity validation failed: {error}"
        return None, f"job identity validation failed: {last_error}"

    def _collect_job_evidence(
        self,
        job: Mapping[str, Any],
        initial_errors: Sequence[str],
    ) -> JobEvidence:
        errors = list(initial_errors)
        job_id = str(job["jobId"])
        status = str(job.get("status", "UNKNOWN"))
        latest = job
        identity, identity_error = self._resolve_job_identity(latest)
        if identity_error is not None:
            errors.append(identity_error)
        evidence_attempts = (
            self._evidence_max_attempts if identity is not None else 1
        )
        is_gpu = identity is not None and identity.profile == "g6x"
        container = latest.get("container")
        if not isinstance(container, Mapping):
            container = {}
        stream = container.get("logStreamName")
        if not isinstance(stream, str):
            stream = None

        if status in _TERMINAL_JOB_STATES and stream is None:
            last_error: Exception | None = None
            for attempt in range(evidence_attempts):
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
                if attempt + 1 < evidence_attempts:
                    self._sleep(self._retry_delay_seconds)
            if stream is None:
                detail = f": {last_error}" if last_error is not None else ""
                errors.append(f"CloudWatch log stream unavailable{detail}")

        log_lines: tuple[str, ...] = ()
        if stream is not None:
            last_error = None
            for attempt in range(evidence_attempts):
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
                if attempt + 1 < evidence_attempts:
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
            purpose=identity.purpose if identity is not None else None,
            kind=identity.kind if identity is not None else None,
            name_prefix=identity.name_prefix if identity is not None else None,
            profile=identity.profile if identity is not None else None,
            queue_name=identity.queue_name if identity is not None else None,
            queue_arn=identity.queue_arn if identity is not None else None,
            job_definition_arn=identity.job_definition_arn
            if identity is not None
            else None,
            job_definition_revision=identity.job_definition_revision
            if identity is not None
            else None,
            image=identity.image if identity is not None else None,
            resource_requirements=identity.resource_requirements
            if identity is not None
            else (),
        )

    def wait(
        self,
        job_ids: Sequence[str],
        *,
        timeout_seconds: float | None = None,
    ) -> tuple[JobEvidence, ...]:
        if not job_ids:
            raise ValueError("at least one job ID is required")
        self._topology_validator.validate()
        effective_timeout = (
            self._wait_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        effective_timeout = self._validate_timeout(effective_timeout)
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
