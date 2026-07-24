from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from pydantic import model_validator
from training_sdk.storage import ExperimentS3Namespace

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


class JobDefinitionExpectation(ContractModel):
    arn: str
    job_role_arn: str
    execution_role_arn: str
    worker_protocol_version: str
    log_configuration: dict[str, Any]


class AwsInfrastructurePreflightContract(ContractModel):
    subnets: tuple[str, ...]
    security_group_ids: tuple[str, ...]
    instance_role: str


class AwsBatchPreflightContract(AwsInfrastructurePreflightContract):
    job_definitions: dict[ResourceProfileName, JobDefinitionExpectation]


@dataclass(frozen=True)
class ValidatedAwsProfile:
    profile: AwsProfile
    compute_environment_arn: str


_COMPUTE_EXPECTATIONS = {
    "c7am": ("c7a.medium", 16, "ECS_AL2023"),
    "c7al": ("c7a.large", 32, "ECS_AL2023"),
    "c7ax": ("c7a.xlarge", 16, "ECS_AL2023"),
    "g6x": ("g6.xlarge", 32, "ECS_AL2023_NVIDIA"),
}


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


def _normalize_log_configuration(
    value: object,
) -> tuple[str, tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    if not isinstance(value, Mapping):
        raise ValueError("logConfiguration must be a mapping")
    known_keys = {"logDriver", "options", "secretOptions"}
    if "logDriver" not in value or set(value) - known_keys:
        raise ValueError("logConfiguration does not match the known AWS schema")
    log_driver = value["logDriver"]
    if not isinstance(log_driver, str) or log_driver != "awslogs":
        raise ValueError("logConfiguration logDriver must be exactly 'awslogs'")

    options = value.get("options", {})
    if not isinstance(options, Mapping):
        raise ValueError("logConfiguration options must be a mapping")
    normalized_options: list[tuple[str, str]] = []
    for key, option_value in options.items():
        if not isinstance(key, str) or not isinstance(option_value, str):
            raise ValueError("logConfiguration options must map strings to strings")
        normalized_options.append((key, option_value))

    secret_options = value.get("secretOptions", [])
    if not isinstance(secret_options, list):
        raise ValueError("logConfiguration secretOptions must be a list")
    normalized_secrets: list[tuple[str, str]] = []
    secret_names: set[str] = set()
    for secret in secret_options:
        if not isinstance(secret, Mapping) or set(secret) != {"name", "valueFrom"}:
            raise ValueError(
                "logConfiguration secretOptions entries must contain exactly name and valueFrom"
            )
        name = secret["name"]
        value_from = secret["valueFrom"]
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(value_from, str)
            or not value_from
        ):
            raise ValueError(
                "logConfiguration secretOptions name and valueFrom must be nonempty strings"
            )
        if name in secret_names:
            raise ValueError("logConfiguration secretOptions names must be unique")
        secret_names.add(name)
        normalized_secrets.append((name, value_from))

    return (
        log_driver,
        tuple(sorted(normalized_options)),
        tuple(sorted(normalized_secrets)),
    )


class AwsBatchPreflight:
    """Read-only validation of formal queues, compute environments, and definitions."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def _definition(
        self,
        expected: JobDefinitionExpectation,
        profile: AwsProfile,
    ) -> ValidatedJobDefinition:
        arn = expected.arn
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
        if (
            definition.get("type") != "container"
            or definition.get("platformCapabilities") != ["EC2"]
        ):
            raise ValueError(f"job definition {arn!r} is not an EC2 container definition")
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
        expected_container_fields = {
            "command": ["python", "/opt/trainer/worker.py"],
            "environment": [
                {
                    "name": "TRAINER_WORKER_PROTOCOL_VERSION",
                    "value": expected.worker_protocol_version,
                }
            ],
            "jobRoleArn": expected.job_role_arn,
            "executionRoleArn": expected.execution_role_arn,
        }
        for field, wanted in expected_container_fields.items():
            if container.get(field) != wanted:
                raise ValueError(
                    f"job definition {arn!r} {field} does not match expected contract"
                )
        try:
            actual_log_configuration = _normalize_log_configuration(
                container.get("logConfiguration")
            )
        except ValueError as error:
            raise ValueError(
                f"job definition {arn!r} actual logConfiguration schema error: {error}"
            ) from error
        try:
            expected_log_configuration = _normalize_log_configuration(
                expected.log_configuration
            )
        except ValueError as error:
            raise ValueError(
                f"job definition {arn!r} expected logConfiguration schema error: {error}"
            ) from error
        if actual_log_configuration != expected_log_configuration:
            raise ValueError(
                f"job definition {arn!r} logConfiguration does not match expected contract"
            )
        return ValidatedJobDefinition(
            arn=arn,
            image_digest=resolved.reference,
            resource_profile=profile.name,
        )

    def validate_profiles(
        self,
        contract: AwsInfrastructurePreflightContract,
    ) -> tuple[ValidatedAwsProfile, ...]:
        result: list[ValidatedAwsProfile] = []
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
                or environment.get("type") != "MANAGED"
                or environment.get("state") != "ENABLED"
                or environment.get("status") != "VALID"
            ):
                raise ValueError(f"compute environment {profile.compute_environment!r} is invalid")
            environment_arn = environment.get("computeEnvironmentArn")
            if not isinstance(environment_arn, str) or not environment_arn:
                raise ValueError("compute environment ARN is missing")
            resources = environment.get("computeResources")
            if not isinstance(resources, Mapping):
                raise ValueError("compute environment resources are missing")
            instance_type, max_vcpus, image_type = _COMPUTE_EXPECTATIONS[name]
            expected_resource_fields = {
                "type": "EC2",
                "instanceTypes": [instance_type],
                "minvCpus": 0,
                "maxvCpus": max_vcpus,
                "subnets": list(contract.subnets),
                "securityGroupIds": list(contract.security_group_ids),
                "instanceRole": contract.instance_role,
            }
            for field, wanted in expected_resource_fields.items():
                if resources.get(field) != wanted:
                    raise ValueError(
                        f"compute environment {profile.compute_environment!r} "
                        f"{field} does not match expected contract"
                    )
            image_configurations = resources.get("ec2Configuration")
            if (
                not isinstance(image_configurations, list)
                or len(image_configurations) != 1
                or not isinstance(image_configurations[0], Mapping)
                or image_configurations[0].get("imageType") != image_type
                or image_configurations[0].get("imageIdOverride") not in (None, "")
            ):
                raise ValueError(
                    f"compute environment {profile.compute_environment!r} "
                    "ec2Configuration does not match expected contract"
                )

            for queue_name, priority in (
                (profile.dev_queue, 10),
                (profile.run_queue, 100),
            ):
                queue_response = self._client.describe_job_queues(
                    jobQueues=[queue_name]
                )
                queue = _one(
                    queue_response.get("jobQueues", []),
                    context=f"job queue {queue_name!r}",
                )
                if (
                    queue.get("jobQueueName") != queue_name
                    or queue.get("state") != "ENABLED"
                    or queue.get("status") != "VALID"
                    or queue.get("priority") != priority
                    or queue.get("computeEnvironmentOrder")
                    != [{"order": 1, "computeEnvironment": environment_arn}]
                ):
                    raise ValueError(f"job queue {queue_name!r} is invalid")
            result.append(
                ValidatedAwsProfile(
                    profile=profile,
                    compute_environment_arn=environment_arn,
                )
            )
        return tuple(result)

    def validate(
        self,
        contract: AwsBatchPreflightContract,
    ) -> tuple[ValidatedJobDefinition, ...]:
        if set(contract.job_definitions) != set(PROFILES):
            raise ValueError("preflight requires job definitions for all four profiles")
        profiles = self.validate_profiles(contract)
        return tuple(
            self._definition(contract.job_definitions[item.profile.name], item.profile)
            for item in profiles
        )


class AwsBatchAdapter:
    """Runtime-only Batch adapter: submit once and query."""

    def __init__(self, client: Any, experiment_s3_prefix: str) -> None:
        self._client = client
        self._namespace = ExperimentS3Namespace.from_prefix(experiment_s3_prefix)

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
        bundle_uri = self._namespace.uri(f"jobs/{job_bundle.job_id}/bundle.json")
        namespace, parsed_job_id = ExperimentS3Namespace.from_bundle_uri(bundle_uri)
        if namespace != self._namespace or parsed_job_id != job_bundle.job_id:
            raise ValueError("job ID does not form one canonical S3 key segment")
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
