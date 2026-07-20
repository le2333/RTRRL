from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from trainer_infra.batch_topology import (
    ACCOUNT_ID,
    REGION,
    ExecutionPurpose,
    expected_topology,
    queue_for,
)
from trainer_infra.heavy_tests import (
    AggregateJobFailure,
    HeavyTestRunner,
    JobEvidence,
    PartialSubmissionError,
    ResourceRequirement,
)

_DIGEST_IMAGE_RE = re.compile(
    r"(?:[a-z0-9]+(?:[._-][a-z0-9]+)*/)*"
    r"[a-z0-9]+(?:[._-][a-z0-9]+)*@sha256:[0-9a-f]{64}"
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
    queue_name: str
    image: str
    resource_requirements: tuple[tuple[str, str], ...]
    expected_instance_type: str
    job_id: str | None = None
    status: str = "PLANNED"
    job_definition_arn: str | None = None
    job_definition_revision: int | None = None
    log_stream_name: str | None = None
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
    account_id: str
    region: str
    captured_at: datetime
    queue_deployment_observed_at: datetime
    execute: bool
    passed: bool
    failure_text: str
    cases: tuple[SmokeEvidence, ...]
    job_definition_arns: tuple[str, ...]
    temporary_image_tags: tuple[tuple[str, str], ...]
    log_stream_names: tuple[str, ...]


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
        queue_name=case.queue_name,
        image=case.image,
        resource_requirements=case.resource_requirements,
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


def _next_token(response: Mapping[str, Any], *, path: str) -> str | None:
    token = response.get("nextToken")
    if token is None:
        return None
    if type(token) is not str or not token:
        raise RuntimeError(f"{path}.nextToken must be a non-empty string")
    return token


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
            for index, value in enumerate(values):
                item = _strict_mapping(
                    value,
                    f"describe_compute_environments.computeEnvironments[{index}]",
                )
                if item.get("computeEnvironmentName") != environment.name:
                    raise RuntimeError(
                        "describe_compute_environments returned an unexpected "
                        "compute environment"
                    )
                matches.append(item)
            token = _next_token(response, path="describe_compute_environments")
            if token is None:
                break
            if token in seen_tokens:
                raise RuntimeError("describe_compute_environments repeated nextToken")
            seen_tokens.add(token)
        if len(matches) != 1:
            raise RuntimeError(
                f"expected exactly one compute environment {environment.name!r}, "
                f"got {len(matches)}"
            )
        cluster = matches[0].get("ecsClusterArn")
        if type(cluster) is not str or not cluster:
            raise RuntimeError(
                f"compute environment {environment.name!r} has no ecsClusterArn"
            )
        clusters.append(cluster)
    if len(set(clusters)) != len(clusters):
        raise RuntimeError("compute environments must have unique ECS clusters")
    return tuple(clusters)


def _describe_exact_job(batch: Any, job_id: str) -> Mapping[str, Any]:
    response = _strict_mapping(
        batch.describe_jobs(jobs=[job_id]),
        "describe_jobs",
    )
    jobs = _strict_list(response.get("jobs"), "describe_jobs.jobs")
    matches = [
        _strict_mapping(job, f"describe_jobs.jobs[{index}]")
        for index, job in enumerate(jobs)
        if isinstance(job, Mapping) and job.get("jobId") == job_id
    ]
    if len(matches) != 1 or len(jobs) != 1:
        raise RuntimeError(
            f"describe_jobs must return exactly one requested job {job_id!r}"
        )
    return matches[0]


def _successful_container_instance(job: Mapping[str, Any]) -> str:
    attempts = _strict_list(job.get("attempts"), "job.attempts")
    matches: list[str] = []
    for index, attempt_value in enumerate(attempts):
        attempt = _strict_mapping(attempt_value, f"job.attempts[{index}]")
        container = _strict_mapping(
            attempt.get("container"), f"job.attempts[{index}].container"
        )
        exit_code = container.get("exitCode")
        arn = container.get("containerInstanceArn")
        if type(exit_code) is int and exit_code == 0 and type(arn) is str and arn:
            matches.append(arn)
    if len(matches) != 1:
        raise RuntimeError(
            "successful job must have exactly one zero-exit attempt with "
            "containerInstanceArn"
        )
    return matches[0]


def _resolve_ec2_instance(
    services: SmokeServices,
    *,
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
        _strict_list(
            response.get("failures"),
            "describe_container_instances.failures",
        )
        for index, value in enumerate(values):
            item = _strict_mapping(
                value,
                f"describe_container_instances.containerInstances[{index}]",
            )
            if item.get("containerInstanceArn") != container_instance_arn:
                raise RuntimeError(
                    "describe_container_instances returned an unexpected "
                    "container instance"
                )
            matches.append(item)
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one ECS container instance match, "
            f"got {len(matches)}"
        )
    ec2_instance_id = matches[0].get("ec2InstanceId")
    if type(ec2_instance_id) is not str or not ec2_instance_id:
        raise RuntimeError("ECS container instance has no ec2InstanceId")

    instances: list[Mapping[str, Any]] = []
    token: str | None = None
    seen_tokens: set[str] = set()
    while True:
        arguments: dict[str, object] = {"InstanceIds": [ec2_instance_id]}
        if token is not None:
            arguments["NextToken"] = token
        response = _strict_mapping(
            services.ec2.describe_instances(**arguments),
            "describe_instances",
        )
        reservations = _strict_list(
            response.get("Reservations"), "describe_instances.Reservations"
        )
        for reservation_index, reservation_value in enumerate(reservations):
            reservation = _strict_mapping(
                reservation_value,
                f"describe_instances.Reservations[{reservation_index}]",
            )
            values = _strict_list(
                reservation.get("Instances"),
                f"describe_instances.Reservations[{reservation_index}].Instances",
            )
            for instance_index, instance_value in enumerate(values):
                instance = _strict_mapping(
                    instance_value,
                    "describe_instances.Reservations"
                    f"[{reservation_index}].Instances[{instance_index}]",
                )
                if instance.get("InstanceId") != ec2_instance_id:
                    raise RuntimeError(
                        "describe_instances returned an unexpected EC2 instance"
                    )
                instances.append(instance)
        raw_token = response.get("NextToken")
        if raw_token is None:
            break
        if type(raw_token) is not str or not raw_token:
            raise RuntimeError(
                "describe_instances.NextToken must be a non-empty string"
            )
        if raw_token in seen_tokens:
            raise RuntimeError("describe_instances repeated NextToken")
        seen_tokens.add(raw_token)
        token = raw_token
    if len(instances) != 1:
        raise RuntimeError(
            f"expected exactly one EC2 instance {ec2_instance_id!r}, "
            f"got {len(instances)}"
        )
    instance_type = instances[0].get("InstanceType")
    if type(instance_type) is not str or not instance_type:
        raise RuntimeError("EC2 instance has no InstanceType")
    return ec2_instance_id, instance_type


def _resource_pairs(
    requirements: Sequence[ResourceRequirement],
) -> tuple[tuple[str, str], ...]:
    return tuple((item.type, item.value) for item in requirements)


def _validate_job_evidence(
    case: SmokeCase,
    evidence: JobEvidence,
) -> list[str]:
    errors = list(evidence.evidence_errors)
    if evidence.status != "SUCCEEDED":
        errors.append(f"status must be SUCCEEDED, got {evidence.status!r}")
    if evidence.purpose is not case.purpose:
        errors.append(
            f"purpose mismatch: expected {case.purpose.value!r}, "
            f"got {evidence.purpose!r}"
        )
    if evidence.profile != case.profile:
        errors.append(
            f"profile mismatch: expected {case.profile!r}, got {evidence.profile!r}"
        )
    if evidence.queue_name != case.queue_name:
        errors.append(
            f"queue mismatch: expected {case.queue_name!r}, "
            f"got {evidence.queue_name!r}"
        )
    if evidence.image != case.image:
        errors.append(
            f"digest image mismatch: expected {case.image!r}, got {evidence.image!r}"
        )
    if _resource_pairs(evidence.resource_requirements) != tuple(
        sorted(case.resource_requirements)
    ):
        errors.append("resource requirements do not match the smoke profile")
    if evidence.exit_code != 0:
        errors.append(f"exit code must be zero, got {evidence.exit_code!r}")
    if not evidence.maximum_rss_lines:
        errors.append("maximum RSS evidence is missing")
    if not any(
        line.strip() == f"trainer_smoke_profile={case.profile}"
        for line in evidence.log_lines
    ):
        errors.append("smoke profile log evidence is missing")
    if not any(
        line.strip() == f"trainer_smoke_purpose={case.purpose.value}"
        for line in evidence.log_lines
    ):
        errors.append("smoke purpose log evidence is missing")
    if case.profile == "g6x":
        if not evidence.jax_gpu_lines:
            errors.append("JAX CUDA device evidence is missing")
        if not evidence.gpu_lines:
            errors.append("NVIDIA L4 evidence is missing")
    return errors


def _merge_terminal_evidence(
    services: SmokeServices,
    case: SmokeCase,
    current: SmokeEvidence,
    job: JobEvidence,
) -> SmokeEvidence:
    errors = _validate_job_evidence(case, job)
    container_instance_arn: str | None = None
    ec2_instance_id: str | None = None
    instance_type: str | None = None
    try:
        described = _describe_exact_job(services.batch, job.job_id)
        container_instance_arn = _successful_container_instance(described)
        ec2_instance_id, instance_type = _resolve_ec2_instance(
            services,
            container_instance_arn=container_instance_arn,
        )
        if instance_type != case.expected_instance_type:
            errors.append(
                "instance type mismatch: expected "
                f"{case.expected_instance_type!r}, got {instance_type!r}"
            )
    except Exception as error:
        errors.append(f"instance evidence failed: {type(error).__name__}: {error}")
    return replace(
        current,
        status=job.status,
        job_definition_arn=job.job_definition_arn,
        job_definition_revision=job.job_definition_revision,
        log_stream_name=job.log_stream_name,
        container_instance_arn=container_instance_arn,
        ec2_instance_id=ec2_instance_id,
        instance_type=instance_type,
        exit_code=job.exit_code,
        maximum_rss_lines=job.maximum_rss_lines,
        gpu_lines=job.gpu_lines,
        jax_gpu_lines=job.jax_gpu_lines,
        evidence_errors=tuple(errors),
    )


def _definition_reuse_errors(
    cases: Sequence[SmokeEvidence],
) -> dict[int, str]:
    errors: dict[int, str] = {}
    for profile in expected_topology().profiles:
        indexes = [
            index for index, case in enumerate(cases) if case.profile == profile
        ]
        identities = {
            (cases[index].job_definition_arn, cases[index].job_definition_revision)
            for index in indexes
        }
        if len(indexes) == 2 and (
            len(identities) != 1 or next(iter(identities))[0] is None
        ):
            message = (
                f"{profile} dev/run did not reuse the same job definition ARN/revision"
            )
            for index in indexes:
                errors[index] = message
    return errors


def _report(
    *,
    captured_at: datetime,
    execute: bool,
    cases: Sequence[SmokeEvidence],
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
        smoke_id="trainer-smoke-shared-queues",
        account_id=ACCOUNT_ID,
        region=REGION,
        captured_at=captured_at,
        queue_deployment_observed_at=captured_at,
        execute=execute,
        passed=execute and not failures and len(finalized) == len(_MATRIX),
        failure_text="\n".join(failures),
        cases=tuple(finalized),
        job_definition_arns=definitions,
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


def run_smoke(
    services: SmokeServices,
    *,
    cpu_image: str,
    gpu_image: str,
    execute: bool = False,
) -> SmokeReport:
    cases = smoke_plan(cpu_image, gpu_image)
    captured_at = datetime.now(timezone.utc)
    evidence = [_planned_evidence(case) for case in cases]
    if not execute:
        return _report(captured_at=captured_at, execute=False, cases=evidence)

    runner = HeavyTestRunner(
        services.batch,
        services.logs,
        services.sts,
    )
    submitted_indexes: list[int] = []
    submission_failed = False
    for index, case in enumerate(cases):
        try:
            submitted = runner.submit(
                profile=case.profile,
                image=case.image,
                tests=[_SMOKE_TEST],
                purpose=case.purpose,
                name_prefix="trainer-smoke",
            )
            if len(submitted) != 1:
                raise RuntimeError(
                    f"expected exactly one submitted job, got {len(submitted)}"
                )
            job = submitted[0]
            evidence[index] = replace(
                evidence[index],
                job_id=job.job_id,
                status="SUBMITTED",
                job_definition_arn=job.job_definition_arn,
                job_definition_revision=job.job_definition_revision,
            )
            submitted_indexes.append(index)
        except PartialSubmissionError as error:
            evidence[index] = replace(
                evidence[index],
                evidence_errors=(
                    f"submission failed: {error}; retained job IDs: "
                    f"{[item.job_id for item in error.submitted]}",
                ),
            )
            submission_failed = True
            break
        except Exception as error:
            evidence[index] = replace(
                evidence[index],
                evidence_errors=(
                    f"submission failed: {type(error).__name__}: {error}",
                ),
            )
            submission_failed = True
            break

    if submitted_indexes:
        job_ids = [
            evidence[index].job_id
            for index in submitted_indexes
            if evidence[index].job_id is not None
        ]
        terminal: tuple[JobEvidence, ...]
        try:
            terminal = runner.wait(job_ids)
        except AggregateJobFailure as error:
            terminal = error.evidence
        except Exception as error:
            terminal = ()
            message = f"wait failed: {type(error).__name__}: {error}"
            for index in submitted_indexes:
                evidence[index] = replace(
                    evidence[index],
                    evidence_errors=(*evidence[index].evidence_errors, message),
                )
        terminal_by_id: dict[str, JobEvidence] = {}
        duplicate_ids: set[str] = set()
        for item in terminal:
            if item.job_id in terminal_by_id:
                duplicate_ids.add(item.job_id)
            terminal_by_id[item.job_id] = item
        for index in submitted_indexes:
            job_id = evidence[index].job_id
            if job_id is None:
                continue
            if job_id in duplicate_ids:
                evidence[index] = replace(
                    evidence[index],
                    evidence_errors=(
                        *evidence[index].evidence_errors,
                        "wait returned duplicate job evidence",
                    ),
                )
                continue
            item = terminal_by_id.get(job_id)
            if item is None:
                evidence[index] = replace(
                    evidence[index],
                    evidence_errors=(
                        *evidence[index].evidence_errors,
                        "wait returned no evidence for submitted job",
                    ),
                )
                continue
            evidence[index] = _merge_terminal_evidence(
                services, cases[index], evidence[index], item
            )

    if submission_failed:
        for index in range(len(submitted_indexes) + 1, len(evidence)):
            evidence[index] = replace(
                evidence[index],
                evidence_errors=("not submitted after an earlier submission failure",),
            )
    report = _report(captured_at=captured_at, execute=True, cases=evidence)
    _write_report_atomically(report)
    return report
