from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from pydantic import model_validator

from trainer_infra.aws_profiles import PROFILES, AwsProfile
from trainer_infra.execution import JobBundle, JobQuery
from trainer_infra.image_catalog import resolve_image
from trainer_infra.models import ContractModel, ResourceProfileName

_DEFINITION_ARN = re.compile(
    r"arn:aws:batch:[^:]+:[0-9]{12}:job-definition/"
    r"(trainer-(c7am|c7al|c7ax|g6x)-([0-9a-f]{64})):([1-9][0-9]*)\Z"
)


class ValidatedJobDefinition(ContractModel):
    arn: str
    image_digest: str
    resource_profile: ResourceProfileName

    @model_validator(mode="after")
    def require_digest_bound_identity(self) -> ValidatedJobDefinition:
        match = _DEFINITION_ARN.fullmatch(self.arn)
        if match is None or match.group(2) != self.resource_profile:
            raise ValueError("job definition ARN is not digest-bound for its profile")
        resolved = resolve_image(self.image_digest)
        if resolved.reference != self.image_digest:
            raise ValueError("job definition image must be a canonical digest reference")
        if resolved.digest.removeprefix("sha256:") != match.group(3):
            raise ValueError("job definition ARN and image digest do not match")
        return self


@dataclass(frozen=True)
class SubmittedJob:
    job_id: str
    bundle_id: str


def _resources(profile: AwsProfile) -> list[dict[str, str]]:
    result = [
        {"type": "VCPU", "value": str(profile.vcpus)},
        {"type": "MEMORY", "value": str(profile.memory_mib)},
    ]
    if profile.gpus:
        result.append({"type": "GPU", "value": str(profile.gpus)})
    return result


def _one(items: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], Mapping):
        raise ValueError(f"{context}: expected exactly one active AWS resource")
    return items[0]


class AwsBatchPreflight:
    """Read-only validation of formal queues, compute environments, and definitions."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def _definition(self, arn: str, profile: AwsProfile) -> ValidatedJobDefinition:
        match = _DEFINITION_ARN.fullmatch(arn)
        if match is None or match.group(2) != profile.name:
            raise ValueError(f"job definition is not digest-bound for profile {profile.name!r}")
        name, _, name_digest, _ = match.groups()
        definitions: list[Mapping[str, Any]] = []
        token: str | None = None
        while True:
            arguments: dict[str, Any] = {
                "jobDefinitionName": name,
                "status": "ACTIVE",
            }
            if token is not None:
                arguments["nextToken"] = token
            response = self._client.describe_job_definitions(**arguments)
            definitions.extend(
                item
                for item in response.get("jobDefinitions", [])
                if isinstance(item, Mapping) and item.get("jobDefinitionArn") == arn
            )
            next_token = response.get("nextToken")
            if not isinstance(next_token, str) or not next_token:
                break
            token = next_token
        definition = _one(definitions, context=f"job definition {arn!r}")
        if definition.get("status") != "ACTIVE":
            raise ValueError(f"job definition {arn!r} is not ACTIVE")
        container = definition.get("containerProperties")
        if not isinstance(container, Mapping):
            raise ValueError(f"job definition {arn!r} has no container properties")
        image = container.get("image")
        if not isinstance(image, str):
            raise ValueError(f"job definition {arn!r} has no image")
        resolved = resolve_image(image)
        if resolved.digest.removeprefix("sha256:") != name_digest:
            raise ValueError(f"job definition {arn!r} image digest does not match its name")
        if container.get("resourceRequirements") != _resources(profile):
            raise ValueError(f"job definition {arn!r} resources do not match profile")
        return ValidatedJobDefinition(
            arn=arn,
            image_digest=resolved.reference,
            resource_profile=profile.name,
        )

    def validate(
        self,
        job_definitions: Mapping[ResourceProfileName, str],
    ) -> tuple[ValidatedJobDefinition, ...]:
        if set(job_definitions) != set(PROFILES):
            raise ValueError("preflight requires job definitions for all four profiles")
        result = []
        for name, profile in PROFILES.items():
            environment_response = self._client.describe_compute_environments(
                computeEnvironments=[profile.compute_environment]
            )
            environment = _one(
                environment_response.get("computeEnvironments", []),
                context=f"compute environment {profile.compute_environment!r}",
            )
            if (
                environment.get("computeEnvironmentName") != profile.compute_environment
                or environment.get("state") != "ENABLED"
                or environment.get("status") != "VALID"
            ):
                raise ValueError(f"compute environment {profile.compute_environment!r} is invalid")
            environment_arn = environment.get("computeEnvironmentArn")
            if not isinstance(environment_arn, str) or not environment_arn:
                raise ValueError("compute environment ARN is missing")

            queue_response = self._client.describe_job_queues(jobQueues=[profile.run_queue])
            queue = _one(
                queue_response.get("jobQueues", []),
                context=f"job queue {profile.run_queue!r}",
            )
            if (
                queue.get("jobQueueName") != profile.run_queue
                or queue.get("state") != "ENABLED"
                or queue.get("status") != "VALID"
                or queue.get("computeEnvironmentOrder")
                != [{"order": 1, "computeEnvironment": environment_arn}]
            ):
                raise ValueError(f"job queue {profile.run_queue!r} is invalid")
            result.append(self._definition(job_definitions[name], profile))
        return tuple(result)


class AwsBatchAdapter:
    """Runtime-only Batch adapter: submit once and query."""

    def __init__(self, client: Any, experiment_s3_prefix: str) -> None:
        if (
            not experiment_s3_prefix.startswith("s3://")
            or "/experiments/" not in experiment_s3_prefix
            or not experiment_s3_prefix.endswith("/")
        ):
            raise ValueError("experiment_s3_prefix must be an S3 experiment prefix")
        self._client = client
        self._experiment_s3_prefix = experiment_s3_prefix

    def submit(
        self,
        job_bundle: JobBundle,
        profile: AwsProfile,
        job_definition: ValidatedJobDefinition,
    ) -> SubmittedJob:
        if job_bundle.resource_profile != profile.name:
            raise ValueError("job bundle resource profile does not match submission profile")
        if job_definition.resource_profile != profile.name:
            raise ValueError("job definition resource profile does not match submission profile")
        if job_definition.image_digest != job_bundle.image_digest:
            raise ValueError("job definition image does not match job bundle image")
        bundle_uri = (
            f"{self._experiment_s3_prefix}jobs/{job_bundle.job_id}/bundle.json"
        )
        job_name = re.sub(r"[^A-Za-z0-9_-]", "-", f"trainer-{job_bundle.job_id}")[:128]
        response = self._client.submit_job(
            jobName=job_name,
            jobQueue=profile.run_queue,
            jobDefinition=job_definition.arn,
            containerOverrides={
                "command": [
                    "python",
                    "/opt/trainer/worker.py",
                    "--bundle-s3-uri",
                    bundle_uri,
                ],
                "resourceRequirements": _resources(profile),
            },
            retryStrategy={"attempts": 1},
        )
        job_id = response.get("jobId")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("Batch submit_job returned no jobId")
        return SubmittedJob(job_id=job_id, bundle_id=job_bundle.job_id)

    def query(self, job_ids: Sequence[str]) -> tuple[JobQuery, ...]:
        jobs: dict[str, Mapping[str, Any]] = {}
        for offset in range(0, len(job_ids), 100):
            chunk = list(job_ids[offset : offset + 100])
            response = self._client.describe_jobs(jobs=chunk)
            for job in response.get("jobs", []):
                if isinstance(job, Mapping) and isinstance(job.get("jobId"), str):
                    jobs[str(job["jobId"])] = job
        result = []
        for job_id in job_ids:
            if job_id not in jobs:
                raise ValueError(f"Batch did not return requested job {job_id!r}")
            job = jobs[job_id]
            result.append(
                JobQuery(
                    job_id=job_id,
                    status=job.get("status"),
                    status_reason=job.get("statusReason"),
                )
            )
        return tuple(result)
