from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import secrets
import tempfile
import time
from typing import Any

from trainer_infra.batch_topology import (
    ACCOUNT_ID,
    REGION,
    BatchTopologyValidator,
    ExecutionPurpose,
    expected_topology,
    queue_for,
)
from trainer_infra.heavy_tests import (
    AggregateJobFailure,
    HeavyTestRunner,
    JobEvidence,
    PartialSubmissionError,
    RegisteredJobDefinition,
    ResourceRequirement,
    SubmittedTestJob,
)

_DIGEST_IMAGE_RE = re.compile(
    r"(?:[a-z0-9]+(?:[._-][a-z0-9]+)*/)*"
    r"[a-z0-9]+(?:[._-][a-z0-9]+)*@sha256:[0-9a-f]{64}"
)
_QUEUE_ARN_RE = re.compile(
    rf"arn:aws:batch:{REGION}:{ACCOUNT_ID}:job-queue/([A-Za-z0-9_-]+)"
)
_DEFINITION_ARN_RE = re.compile(
    rf"arn:aws:batch:{REGION}:{ACCOUNT_ID}:job-definition/"
    r"(trainer-smoke-(s[a-z0-9]{12})-(c7am|c7al|c7ax|g6x)-"
    r"([0-9a-f]{64})):([1-9][0-9]*)"
)
_SMOKE_TEST = "memo/tests/test_logging_compat.py"
_REPORT_PATH = Path(".trainer/smoke/trainer-smoke-shared-queues.json")
_MATRIX = (
    (ExecutionPurpose.DEV, "c7am"),
    (ExecutionPurpose.DEV, "c7al"),
    (ExecutionPurpose.DEV, "c7ax"),
    (ExecutionPurpose.RUN, "c7am"),
    (ExecutionPurpose.RUN, "c7al"),
    (ExecutionPurpose.RUN, "c7ax"),
    (ExecutionPurpose.DEV, "g6x"),
    (ExecutionPurpose.RUN, "g6x"),
)


class SmokeDeadlineExceeded(RuntimeError):
    """The one smoke execution deadline was exhausted."""


@dataclass(frozen=True)
class SmokeServices:
    batch: Any
    logs: Any
    sts: Any
    ecs: Any
    ec2: Any


@dataclass(frozen=True)
class SmokeCase:
    purpose: ExecutionPurpose
    profile: str
    smoke_name: str
    queue_name: str
    image: str
    resource_requirements: tuple[tuple[str, str], ...]
    expected_instance_type: str


@dataclass(frozen=True)
class SmokeEvidence:
    purpose: ExecutionPurpose
    profile: str
    smoke_name: str
    expected_queue_name: str
    expected_image: str
    expected_resource_requirements: tuple[tuple[str, str], ...]
    expected_instance_type: str
    job_id: str | None = None
    status: str = "PLANNED"
    queue_name: str | None = None
    queue_arn: str | None = None
    image: str | None = None
    resource_requirements: tuple[tuple[str, str], ...] = ()
    job_definition_arn: str | None = None
    job_definition_revision: int | None = None
    definition_owned: bool = False
    log_stream_name: str | None = None
    log_lines: tuple[str, ...] = ()
    profile_marker_lines: tuple[str, ...] = ()
    purpose_marker_lines: tuple[str, ...] = ()
    container_instance_arn: str | None = None
    ec2_instance_id: str | None = None
    instance_type: str | None = None
    exit_code: int | None = None
    maximum_rss_lines: tuple[str, ...] = ()
    gpu_lines: tuple[str, ...] = ()
    jax_gpu_lines: tuple[str, ...] = ()
    evidence_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class SmokeReport:
    smoke_id: str
    definition_scope: str
    account_id: str
    region: str
    captured_at: datetime
    queue_deployment_observed_at: datetime | None
    execute: bool
    passed: bool
    failure_text: str
    cases: tuple[SmokeEvidence, ...]
    job_definition_arns: tuple[str, ...]
    owned_job_definition_arns: tuple[str, ...]
    temporary_image_tags: tuple[tuple[str, str], ...]
    log_stream_names: tuple[str, ...]


@dataclass
class _ExecutionState:
    definition_scope: str
    owned_definitions: dict[str, RegisteredJobDefinition]


class _Deadline:
    def __init__(
        self,
        *,
        timeout_seconds: float,
        monotonic: Callable[[], float],
        sleep: Callable[[float], None],
    ) -> None:
        self.monotonic = monotonic
        self._sleep = sleep
        self.deadline = monotonic() + timeout_seconds
        self.timeout_seconds = timeout_seconds

    def remaining(self, stage: str) -> float:
        remaining = self.deadline - self.monotonic()
        if remaining <= 0:
            raise SmokeDeadlineExceeded(
                f"smoke deadline exhausted during {stage} after "
                f"{self.timeout_seconds:g} seconds"
            )
        return remaining

    def check(self, stage: str) -> None:
        self.remaining(stage)

    def sleep(self, seconds: float) -> None:
        self._sleep(min(seconds, self.remaining("retry sleep")))
        self.check("retry sleep")


class _DeadlineClient:
    def __init__(
        self,
        client: Any,
        *,
        service: str,
        deadline: _Deadline,
        state: _ExecutionState,
    ) -> None:
        self._client = client
        self._service = service
        self._deadline = deadline
        self._state = state

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._client, name)
        if not callable(value):
            return value

        def call(*args: object, **kwargs: object) -> object:
            stage = f"{self._service}.{name}"
            self._deadline.check(stage)
            response = value(*args, **kwargs)
            if self._service == "batch" and name == "register_job_definition":
                self._record_owned_definition(response, kwargs)
            self._deadline.check(stage)
            return response

        return call

    def _record_owned_definition(
        self, response: object, arguments: Mapping[str, object]
    ) -> None:
        if not isinstance(response, Mapping):
            return
        name = arguments.get("jobDefinitionName")
        response_name = response.get("jobDefinitionName")
        arn = response.get("jobDefinitionArn")
        revision = response.get("revision")
        if (
            type(name) is not str
            or response_name != name
            or type(arn) is not str
            or type(revision) is not int
            or revision < 1
        ):
            return
        expected = (
            f"arn:aws:batch:{REGION}:{ACCOUNT_ID}:job-definition/"
            f"{name}:{revision}"
        )
        if arn != expected:
            return
        self._state.owned_definitions[arn] = RegisteredJobDefinition(
            name=name,
            arn=arn,
            revision=revision,
            owned=True,
            scope=self._state.definition_scope,
        )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _validate_timeout(timeout_seconds: float) -> float:
    if (
        type(timeout_seconds) not in (int, float)
        or isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be finite and positive")
    return float(timeout_seconds)


def _validate_digest_image(image: str) -> None:
    if type(image) is not str or _DIGEST_IMAGE_RE.fullmatch(image) is None:
        raise ValueError("image must be an exact lowercase sha256 digest reference")


def smoke_plan(cpu_image: str, gpu_image: str) -> tuple[SmokeCase, ...]:
    _validate_digest_image(cpu_image)
    _validate_digest_image(gpu_image)
    topology = expected_topology()
    return tuple(
        SmokeCase(
            purpose=purpose,
            profile=profile,
            smoke_name=f"trainer-smoke-{purpose.value}-{profile}",
            queue_name=queue_for(purpose, profile).name,
            image=gpu_image if profile == "g6x" else cpu_image,
            resource_requirements=topology.profiles[profile].resource_requirements,
            expected_instance_type=topology.compute_environments[
                topology.profiles[profile].compute_environment
            ].instance_type,
        )
        for purpose, profile in _MATRIX
    )


def _planned_evidence(case: SmokeCase) -> SmokeEvidence:
    return SmokeEvidence(
        purpose=case.purpose,
        profile=case.profile,
        smoke_name=case.smoke_name,
        expected_queue_name=case.queue_name,
        expected_image=case.image,
        expected_resource_requirements=case.resource_requirements,
        expected_instance_type=case.expected_instance_type,
    )


def _strict_mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{path} must be a mapping")
    return value


def _strict_list(value: object, path: str) -> list[Any]:
    if type(value) is not list:
        raise RuntimeError(f"{path} must be a list")
    return value


def _normalize_resources(value: object) -> tuple[tuple[str, str], ...]:
    values = _strict_list(value, "job.container.resourceRequirements")
    normalized: list[tuple[str, str]] = []
    for index, item_value in enumerate(values):
        item = _strict_mapping(
            item_value, f"job.container.resourceRequirements[{index}]"
        )
        kind = item.get("type")
        amount = item.get("value")
        if type(kind) is not str or type(amount) is not str:
            raise RuntimeError("resource requirement type/value must be strings")
        normalized.append((kind, amount))
    if len({kind for kind, _ in normalized}) != len(normalized):
        raise RuntimeError("duplicate resource requirement types")
    return tuple(sorted(normalized))


def _describe_exact_job(batch: Any, job_id: str) -> Mapping[str, Any]:
    response = _strict_mapping(batch.describe_jobs(jobs=[job_id]), "describe_jobs")
    jobs = _strict_list(response.get("jobs"), "describe_jobs.jobs")
    if len(jobs) != 1:
        raise RuntimeError(
            f"describe_jobs must return exactly one requested job {job_id!r}"
        )
    job = _strict_mapping(jobs[0], "describe_jobs.jobs[0]")
    if job.get("jobId") != job_id:
        raise RuntimeError("describe_jobs returned an unexpected job")
    return job


def _successful_attempt(job: Mapping[str, Any]) -> Mapping[str, Any]:
    attempts = _strict_list(job.get("attempts"), "job.attempts")
    successful: list[Mapping[str, Any]] = []
    for index, attempt_value in enumerate(attempts):
        attempt = _strict_mapping(attempt_value, f"job.attempts[{index}]")
        container = _strict_mapping(
            attempt.get("container"), f"job.attempts[{index}].container"
        )
        if type(container.get("exitCode")) is int and container["exitCode"] == 0:
            successful.append(container)
    if len(successful) != 1:
        raise RuntimeError(
            f"job must have exactly one successful attempt; got {len(successful)}"
        )
    return successful[0]


def _actual_queue(reference: object) -> tuple[str | None, str | None]:
    if type(reference) is not str or not reference:
        raise RuntimeError("jobQueue must be a non-empty string")
    match = _QUEUE_ARN_RE.fullmatch(reference)
    if match is not None:
        return match.group(1), reference
    if re.fullmatch(r"[A-Za-z0-9_-]+", reference) is None:
        raise RuntimeError(f"jobQueue reference is malformed: {reference!r}")
    return reference, None


def _actual_definition(reference: object) -> tuple[str, int]:
    if type(reference) is not str:
        raise RuntimeError("jobDefinition must be a string")
    match = _DEFINITION_ARN_RE.fullmatch(reference)
    if match is None:
        raise RuntimeError("jobDefinition is not a scoped smoke definition ARN")
    return reference, int(match.group(5))


def _compute_environment_clusters(batch: Any) -> tuple[str, ...]:
    clusters: list[str] = []
    for environment in expected_topology().compute_environments.values():
        matches: list[Mapping[str, Any]] = []
        token: str | None = None
        seen_tokens: set[str] = set()
        while True:
            arguments: dict[str, object] = {
                "computeEnvironments": [environment.name]
            }
            if token is not None:
                arguments["nextToken"] = token
            response = _strict_mapping(
                batch.describe_compute_environments(**arguments),
                "describe_compute_environments",
            )
            values = _strict_list(
                response.get("computeEnvironments"),
                "describe_compute_environments.computeEnvironments",
            )
            matches.extend(
                _strict_mapping(value, "compute environment") for value in values
            )
            next_token = response.get("nextToken")
            if next_token is None:
                break
            if type(next_token) is not str or not next_token:
                raise RuntimeError("compute environment nextToken is malformed")
            if next_token in seen_tokens:
                raise RuntimeError("compute environment nextToken cycle")
            seen_tokens.add(next_token)
            token = next_token
        if len(matches) != 1:
            raise RuntimeError(
                f"expected exactly one compute environment {environment.name!r}"
            )
        item = matches[0]
        if item.get("computeEnvironmentName") != environment.name:
            raise RuntimeError("unexpected compute environment")
        cluster = item.get("ecsClusterArn")
        if type(cluster) is not str or not cluster:
            raise RuntimeError("compute environment has no ecsClusterArn")
        clusters.append(cluster)
    if len(set(clusters)) != len(clusters):
        raise RuntimeError("compute environments must have unique ECS clusters")
    return tuple(clusters)


def _resolve_ec2_instance(
    services: SmokeServices,
    container_instance_arn: str,
) -> tuple[str, str]:
    matches: list[Mapping[str, Any]] = []
    for cluster in _compute_environment_clusters(services.batch):
        response = _strict_mapping(
            services.ecs.describe_container_instances(
                cluster=cluster,
                containerInstances=[container_instance_arn],
            ),
            "describe_container_instances",
        )
        values = _strict_list(
            response.get("containerInstances"),
            "describe_container_instances.containerInstances",
        )
        _strict_list(response.get("failures"), "describe_container_instances.failures")
        for value in values:
            item = _strict_mapping(value, "ECS container instance")
            if item.get("containerInstanceArn") != container_instance_arn:
                raise RuntimeError(
                    "describe_container_instances returned an unexpected "
                    "container instance"
                )
            matches.append(item)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one ECS container instance match, got {len(matches)}"
        )
    instance_id = matches[0].get("ec2InstanceId")
    if type(instance_id) is not str or not instance_id:
        raise RuntimeError("ECS container instance has no ec2InstanceId")
    instances: list[Mapping[str, Any]] = []
    token: str | None = None
    seen_tokens: set[str] = set()
    while True:
        arguments: dict[str, object] = {"InstanceIds": [instance_id]}
        if token is not None:
            arguments["NextToken"] = token
        response = _strict_mapping(
            services.ec2.describe_instances(**arguments),
            "describe_instances",
        )
        reservations = _strict_list(
            response.get("Reservations"), "describe_instances.Reservations"
        )
        for reservation_value in reservations:
            reservation = _strict_mapping(reservation_value, "EC2 reservation")
            for instance_value in _strict_list(
                reservation.get("Instances"), "EC2 reservation instances"
            ):
                instance = _strict_mapping(instance_value, "EC2 instance")
                if instance.get("InstanceId") != instance_id:
                    raise RuntimeError(
                        "describe_instances returned an unexpected instance"
                    )
                instances.append(instance)
        next_token = response.get("NextToken")
        if next_token is None:
            break
        if type(next_token) is not str or not next_token:
            raise RuntimeError("describe_instances NextToken is malformed")
        if next_token in seen_tokens:
            raise RuntimeError("describe_instances NextToken cycle")
        seen_tokens.add(next_token)
        token = next_token
    if len(instances) != 1:
        raise RuntimeError(f"expected exactly one EC2 instance {instance_id!r}")
    instance_type = instances[0].get("InstanceType")
    if type(instance_type) is not str or not instance_type:
        raise RuntimeError("EC2 instance has no InstanceType")
    return instance_id, instance_type


def _resource_pairs(
    requirements: Sequence[ResourceRequirement],
) -> tuple[tuple[str, str], ...]:
    return tuple((item.type, item.value) for item in requirements)


def _observe_job(
    services: SmokeServices,
    case: SmokeCase,
    current: SmokeEvidence,
    job_evidence: JobEvidence,
) -> SmokeEvidence:
    errors = list(job_evidence.evidence_errors)
    queue_name: str | None = None
    queue_arn: str | None = None
    image: str | None = None
    resources: tuple[tuple[str, str], ...] = ()
    definition_arn = current.job_definition_arn
    definition_revision = current.job_definition_revision
    submitted_definition = (definition_arn, definition_revision)
    stream: str | None = None
    exit_code: int | None = None
    container_instance_arn: str | None = None
    ec2_instance_id: str | None = None
    instance_type: str | None = None
    try:
        job = _describe_exact_job(services.batch, job_evidence.job_id)
        queue_name, queue_arn = _actual_queue(job.get("jobQueue"))
        definition_arn, definition_revision = _actual_definition(
            job.get("jobDefinition")
        )
        container = _strict_mapping(job.get("container"), "job.container")
        actual_image = container.get("image")
        if type(actual_image) is not str:
            raise RuntimeError("job container image must be a string")
        image = actual_image
        resources = _normalize_resources(container.get("resourceRequirements"))
        attempt = _successful_attempt(job)
        exit_code_value = attempt.get("exitCode")
        stream_value = attempt.get("logStreamName")
        container_arn_value = attempt.get("containerInstanceArn")
        if type(exit_code_value) is not int:
            raise RuntimeError("successful attempt exitCode must be an integer")
        if type(stream_value) is not str or not stream_value:
            raise RuntimeError("successful attempt has no logStreamName")
        if type(container_arn_value) is not str or not container_arn_value:
            raise RuntimeError("successful attempt has no containerInstanceArn")
        exit_code = exit_code_value
        stream = stream_value
        container_instance_arn = container_arn_value
        ec2_instance_id, instance_type = _resolve_ec2_instance(
            services, container_instance_arn
        )
    except Exception as error:
        errors.append(f"observed job evidence failed: {type(error).__name__}: {error}")

    if stream is not None and job_evidence.log_stream_name != stream:
        errors.append(
            "log stream mismatch between successful attempt and CloudWatch evidence"
        )
    log_lines = (
        job_evidence.log_lines
        if stream is not None and job_evidence.log_stream_name == stream
        else ()
    )
    rss = tuple(
        line
        for line in log_lines
        if "Maximum resident set size (kbytes):" in line
    )
    gpu = tuple(line for line in log_lines if line.strip().startswith("NVIDIA L4"))
    jax = tuple(line for line in log_lines if "CudaDevice(" in line)
    profile_markers = tuple(
        line
        for line in log_lines
        if line.strip().startswith("trainer_smoke_profile=")
    )
    purpose_markers = tuple(
        line
        for line in log_lines
        if line.strip().startswith("trainer_smoke_purpose=")
    )

    if job_evidence.status != "SUCCEEDED":
        errors.append(f"status must be SUCCEEDED, got {job_evidence.status!r}")
    if (definition_arn, definition_revision) != submitted_definition:
        errors.append(
            "definition ARN/revision mismatch: expected "
            f"{submitted_definition!r}, got "
            f"{(definition_arn, definition_revision)!r}"
        )
    if queue_name != case.queue_name:
        errors.append(
            f"queue mismatch: expected {case.queue_name!r}, got {queue_name!r}"
        )
    expected_queue_arn = (
        f"arn:aws:batch:{REGION}:{ACCOUNT_ID}:job-queue/{case.queue_name}"
    )
    if queue_arn != expected_queue_arn:
        errors.append(
            f"queue ARN mismatch: expected {expected_queue_arn!r}, "
            f"got {queue_arn!r}"
        )
    if image != case.image:
        errors.append(f"image mismatch: expected {case.image!r}, got {image!r}")
    if resources != tuple(sorted(case.resource_requirements)):
        errors.append("resource requirements mismatch")
    if exit_code != 0:
        errors.append(f"exit code must be zero, got {exit_code!r}")
    if not rss:
        errors.append("maximum RSS evidence is missing")
    if profile_markers != (f"trainer_smoke_profile={case.profile}",):
        errors.append("profile marker evidence is not exact")
    if purpose_markers != (
        f"trainer_smoke_purpose={case.purpose.value}",
    ):
        errors.append("purpose marker evidence is not exact")
    if instance_type != case.expected_instance_type:
        errors.append(
            "instance type mismatch: expected "
            f"{case.expected_instance_type!r}, got {instance_type!r}"
        )
    if case.profile == "g6x" and not gpu:
        errors.append("NVIDIA L4 evidence is missing")
    if case.profile == "g6x" and not jax:
        errors.append("JAX CUDA device evidence is missing")

    return replace(
        current,
        status=job_evidence.status,
        queue_name=queue_name,
        queue_arn=queue_arn,
        image=image,
        resource_requirements=resources,
        job_definition_arn=definition_arn,
        job_definition_revision=definition_revision,
        log_stream_name=stream,
        log_lines=log_lines,
        profile_marker_lines=profile_markers,
        purpose_marker_lines=purpose_markers,
        container_instance_arn=container_instance_arn,
        ec2_instance_id=ec2_instance_id,
        instance_type=instance_type,
        exit_code=exit_code,
        maximum_rss_lines=rss,
        gpu_lines=gpu,
        jax_gpu_lines=jax,
        evidence_errors=tuple(dict.fromkeys(errors)),
    )


def _definition_reuse_errors(cases: Sequence[SmokeEvidence]) -> dict[int, str]:
    errors: dict[int, str] = {}
    for profile in expected_topology().profiles:
        indexes = [index for index, item in enumerate(cases) if item.profile == profile]
        identities = {
            (cases[index].job_definition_arn, cases[index].job_definition_revision)
            for index in indexes
        }
        if len(identities) != 1 or next(iter(identities))[0] is None:
            for index in indexes:
                errors[index] = (
                    f"{profile} dev/run did not reuse one definition ARN/revision"
                )
    return errors


def _make_report(
    *,
    scope: str,
    captured_at: datetime,
    topology_observed_at: datetime | None,
    execute: bool,
    cases: Sequence[SmokeEvidence],
    owned_definitions: Mapping[str, RegisteredJobDefinition],
) -> SmokeReport:
    finalized = list(cases)
    if execute:
        for index, message in _definition_reuse_errors(finalized).items():
            finalized[index] = replace(
                finalized[index],
                evidence_errors=(*finalized[index].evidence_errors, message),
            )
    failures = [
        f"{case.smoke_name}: {message}"
        for case in finalized
        for message in case.evidence_errors
    ]
    if execute:
        failures.extend(
            f"{case.smoke_name}: no job was submitted"
            for case in finalized
            if case.job_id is None and not case.evidence_errors
        )
    definitions = tuple(
        dict.fromkeys(
            case.job_definition_arn
            for case in finalized
            if case.job_definition_arn is not None
        )
    )
    streams = tuple(
        dict.fromkeys(
            case.log_stream_name
            for case in finalized
            if case.log_stream_name is not None
        )
    )
    return SmokeReport(
        smoke_id=f"trainer-smoke-{scope}",
        definition_scope=scope,
        account_id=ACCOUNT_ID,
        region=REGION,
        captured_at=captured_at,
        queue_deployment_observed_at=topology_observed_at,
        execute=execute,
        passed=execute and not failures and len(finalized) == len(_MATRIX),
        failure_text="\n".join(failures),
        cases=tuple(finalized),
        job_definition_arns=definitions,
        owned_job_definition_arns=tuple(owned_definitions),
        temporary_image_tags=(),
        log_stream_names=streams,
    )


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _write_report_atomically(report: SmokeReport) -> None:
    path = _REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        asdict(report),
        allow_nan=False,
        default=_json_default,
        ensure_ascii=False,
        sort_keys=True,
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _scoped_services(
    services: SmokeServices,
    deadline: _Deadline,
    state: _ExecutionState,
) -> SmokeServices:
    return SmokeServices(
        batch=_DeadlineClient(
            services.batch, service="batch", deadline=deadline, state=state
        ),
        logs=_DeadlineClient(
            services.logs, service="logs", deadline=deadline, state=state
        ),
        sts=_DeadlineClient(
            services.sts, service="sts", deadline=deadline, state=state
        ),
        ecs=_DeadlineClient(
            services.ecs, service="ecs", deadline=deadline, state=state
        ),
        ec2=_DeadlineClient(
            services.ec2, service="ec2", deadline=deadline, state=state
        ),
    )


def _retain_submission(
    current: SmokeEvidence,
    submitted: SubmittedTestJob,
) -> SmokeEvidence:
    return replace(
        current,
        job_id=submitted.job_id,
        status="SUBMITTED",
        job_definition_arn=submitted.job_definition_arn,
        job_definition_revision=submitted.job_definition_revision,
        definition_owned=submitted.definition_owned,
    )


def run_smoke(
    services: SmokeServices,
    *,
    cpu_image: str,
    gpu_image: str,
    execute: bool = False,
    timeout_seconds: float = 3600.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> SmokeReport:
    timeout = _validate_timeout(timeout_seconds)
    cases = smoke_plan(cpu_image, gpu_image)
    captured_at = _utc_now()
    scope = f"s{secrets.token_hex(6)}"
    evidence = [_planned_evidence(case) for case in cases]
    if not execute:
        return _make_report(
            scope=scope,
            captured_at=captured_at,
            topology_observed_at=None,
            execute=False,
            cases=evidence,
            owned_definitions={},
        )

    state = _ExecutionState(scope, {})
    deadline = _Deadline(
        timeout_seconds=timeout,
        monotonic=monotonic,
        sleep=sleep,
    )
    scoped = _scoped_services(services, deadline, state)
    topology_observed_at: datetime | None = None
    submitted_indexes: list[int] = []
    try:
        BatchTopologyValidator(scoped.batch, scoped.sts).validate()
        deadline.check("topology validation")
        topology_observed_at = _utc_now()
        runner = HeavyTestRunner(
            scoped.batch,
            scoped.logs,
            scoped.sts,
            sleep=deadline.sleep,
            monotonic=monotonic,
            wait_timeout_seconds=timeout,
        )
        for index, case in enumerate(cases):
            try:
                jobs = runner.submit(
                    profile=case.profile,
                    image=case.image,
                    tests=[_SMOKE_TEST],
                    purpose=case.purpose,
                    name_prefix="trainer-smoke",
                    definition_scope=scope,
                )
                if len(jobs) != 1:
                    raise RuntimeError(
                        f"expected exactly one submitted job, got {len(jobs)}"
                    )
                evidence[index] = _retain_submission(evidence[index], jobs[0])
                submitted_indexes.append(index)
            except PartialSubmissionError as error:
                retained_definitions = list(error.registered_definitions)
                if not retained_definitions:
                    retained_definitions = [
                        definition
                        for definition in state.owned_definitions.values()
                        if f"-{case.profile}-" in definition.name
                    ]
                for definition in retained_definitions:
                    if definition.owned:
                        state.owned_definitions[definition.arn] = definition
                    evidence[index] = replace(
                        evidence[index],
                        job_definition_arn=definition.arn,
                        job_definition_revision=definition.revision,
                        definition_owned=definition.owned,
                    )
                evidence[index] = replace(
                    evidence[index],
                    evidence_errors=(
                        f"submission failed: {error}; retained job IDs: "
                        f"{[item.job_id for item in error.submitted]}",
                    ),
                )
                break
            except Exception as error:
                evidence[index] = replace(
                    evidence[index],
                    evidence_errors=(
                        f"submission failed: {type(error).__name__}: {error}",
                    ),
                )
                break
        for index in range(len(submitted_indexes) + 1, len(evidence)):
            if evidence[index].job_id is None and not evidence[index].evidence_errors:
                evidence[index] = replace(
                    evidence[index],
                    evidence_errors=(
                        "not submitted after an earlier submission failure",
                    ),
                )

        if submitted_indexes:
            job_ids = [
                evidence[index].job_id
                for index in submitted_indexes
                if evidence[index].job_id is not None
            ]
            try:
                terminal = runner.wait(
                    job_ids,
                    timeout_seconds=deadline.remaining("Batch terminal wait"),
                )
            except AggregateJobFailure as error:
                terminal = error.evidence
            terminal_by_id = {item.job_id: item for item in terminal}
            if len(terminal_by_id) != len(terminal):
                raise RuntimeError("wait returned duplicate job evidence")
            for index in submitted_indexes:
                job_id = evidence[index].job_id
                item = terminal_by_id.get(str(job_id))
                if item is None:
                    evidence[index] = replace(
                        evidence[index],
                        evidence_errors=(
                            *evidence[index].evidence_errors,
                            "wait returned no evidence for submitted job",
                        ),
                    )
                else:
                    evidence[index] = _observe_job(
                        scoped, cases[index], evidence[index], item
                    )
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        targets = submitted_indexes or [0]
        for index in targets:
            evidence[index] = replace(
                evidence[index],
                evidence_errors=(
                    *evidence[index].evidence_errors,
                    message,
                ),
            )

    report = _make_report(
        scope=scope,
        captured_at=captured_at,
        topology_observed_at=topology_observed_at,
        execute=True,
        cases=evidence,
        owned_definitions=state.owned_definitions,
    )
    _write_report_atomically(report)
    return report
