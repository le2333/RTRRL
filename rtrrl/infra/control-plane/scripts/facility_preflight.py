from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError


ACCOUNT_ID = "007122174918"
REGION = "eu-north-1"
BUCKET = "rtrrl-artifacts-007122174918"
PREFIX = "experiments/"
ECR_REPOSITORY = "rtrrl"
FORMAL_IMAGE_TAGS = (
    "memorax-rtrl-facility-cpu",
    "memorax-rtrl-facility-gpu",
)
COMPUTE_ENVIRONMENTS = (
    {"profile": "c7am", "name": "rtrrl-cpu-c7am-ce", "instance_type": "c7a.medium"},
    {"profile": "c7al", "name": "rtrrl-cpu-c7al-ce", "instance_type": "c7a.large"},
    {"profile": "c7ax", "name": "rtrrl-cpu-c7ax-ce", "instance_type": "c7a.xlarge"},
    {"profile": "g6x", "name": "rtrrl-gpu-g6x-ce", "instance_type": "g6.xlarge"},
)
QUEUES = (
    {
        "profile": "c7am",
        "kind": "dev",
        "name": "dev-cpu-c7am-queue",
        "priority": 10,
        "environment": "rtrrl-cpu-c7am-ce",
    },
    {
        "profile": "c7am",
        "kind": "run",
        "name": "run-cpu-c7am-queue",
        "priority": 100,
        "environment": "rtrrl-cpu-c7am-ce",
    },
    {
        "profile": "c7al",
        "kind": "dev",
        "name": "dev-cpu-c7al-queue",
        "priority": 10,
        "environment": "rtrrl-cpu-c7al-ce",
    },
    {
        "profile": "c7al",
        "kind": "run",
        "name": "run-cpu-c7al-queue",
        "priority": 100,
        "environment": "rtrrl-cpu-c7al-ce",
    },
    {
        "profile": "c7ax",
        "kind": "dev",
        "name": "dev-cpu-c7ax-queue",
        "priority": 10,
        "environment": "rtrrl-cpu-c7ax-ce",
    },
    {
        "profile": "c7ax",
        "kind": "run",
        "name": "run-cpu-c7ax-queue",
        "priority": 100,
        "environment": "rtrrl-cpu-c7ax-ce",
    },
    {
        "profile": "g6x",
        "kind": "dev",
        "name": "dev-gpu-queue",
        "priority": 10,
        "environment": "rtrrl-gpu-g6x-ce",
    },
    {
        "profile": "g6x",
        "kind": "run",
        "name": "run-gpu-queue",
        "priority": 100,
        "environment": "rtrrl-gpu-g6x-ce",
    },
)

# These are the exact write/read permissions needed by the later, separately
# authorized Task 7 phases. This script only asks IAM to evaluate them.
REQUIRED_ACTIONS = (
    "batch:DescribeComputeEnvironments",
    "batch:DescribeJobDefinitions",
    "batch:DescribeJobQueues",
    "batch:DescribeJobs",
    "batch:RegisterJobDefinition",
    "batch:SubmitJob",
    "ecr:BatchCheckLayerAvailability",
    "ecr:BatchGetImage",
    "ecr:CompleteLayerUpload",
    "ecr:DescribeImages",
    "ecr:DescribeRepositories",
    "ecr:GetAuthorizationToken",
    "ecr:GetDownloadUrlForLayer",
    "ecr:InitiateLayerUpload",
    "ecr:PutImage",
    "ecr:UploadLayerPart",
    "iam:PassRole",
    "logs:DescribeLogStreams",
    "logs:GetLogEvents",
    "s3:GetBucketLocation",
    "s3:GetObject",
    "s3:ListBucket",
    "s3:PutObject",
)


class PreflightConfig:
    def __init__(
        self,
        *,
        aim_repo: Path,
        main_repo: Path,
        aim_endpoint: str,
        account_id: str = ACCOUNT_ID,
        region: str = REGION,
        bucket: str = BUCKET,
        prefix: str = PREFIX,
        ecr_repository: str = ECR_REPOSITORY,
    ) -> None:
        self.aim_repo = Path(aim_repo)
        self.main_repo = Path(main_repo)
        self.aim_endpoint = aim_endpoint
        self.account_id = account_id
        self.region = region
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/"
        self.ecr_repository = ecr_repository


def stable_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _permission_denied(error: BaseException) -> bool:
    if isinstance(error, PermissionError):
        return True
    if isinstance(error, ClientError):
        code = str(error.response.get("Error", {}).get("Code", ""))
        return code in {
            "AccessDenied",
            "AccessDeniedException",
            "AuthorizationError",
            "UnauthorizedOperation",
        }
    return False


def _error(error: BaseException) -> dict[str, str]:
    return {
        "error": str(error),
        "status": "blocked" if _permission_denied(error) else "failed",
        "type": type(error).__name__,
    }


def _one(items: object, context: str) -> Mapping[str, Any]:
    if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], Mapping):
        raise ValueError(f"{context}: expected exactly one resource")
    return items[0]


def _principal_arn(identity_arn: str) -> str:
    marker = ":assumed-role/"
    if marker not in identity_arn:
        return identity_arn
    prefix, role_and_session = identity_arn.split(marker, 1)
    role_name, separator, _session = role_and_session.rpartition("/")
    if not separator or not role_name:
        raise ValueError("STS assumed-role ARN is malformed")
    account = prefix.rsplit(":", 1)[-1]
    return f"arn:aws:iam::{account}:role/{role_name}"


def _check_iam(iam: Any, identity_arn: str) -> dict[str, Any]:
    principal = _principal_arn(identity_arn)
    evaluations: dict[str, str] = {}
    marker: str | None = None
    try:
        while True:
            arguments: dict[str, Any] = {
                "PolicySourceArn": principal,
                "ActionNames": list(REQUIRED_ACTIONS),
            }
            if marker is not None:
                arguments["Marker"] = marker
            response = iam.simulate_principal_policy(**arguments)
            for item in response.get("EvaluationResults", []):
                action = item.get("EvalActionName")
                decision = item.get("EvalDecision")
                if isinstance(action, str) and isinstance(decision, str):
                    evaluations[action] = decision
            if not response.get("IsTruncated"):
                break
            marker_value = response.get("Marker")
            if not isinstance(marker_value, str) or not marker_value:
                raise ValueError("IAM simulation was truncated without a marker")
            marker = marker_value
    except BaseException as error:
        return {
            "actions": list(REQUIRED_ACTIONS),
            "error": str(error),
            "method": "SimulatePrincipalPolicy",
            "principal": principal,
            "status": "blocked" if _permission_denied(error) else "failed",
            "type": type(error).__name__,
        }
    missing = [action for action in REQUIRED_ACTIONS if action not in evaluations]
    denied = [
        action
        for action in REQUIRED_ACTIONS
        if evaluations.get(action) not in {"allowed", "allowedByOrganizations"}
    ]
    status = "allowed" if not denied and not missing else "denied"
    return {
        "decisions": {action: evaluations.get(action, "missing") for action in REQUIRED_ACTIONS},
        "denied": denied,
        "method": "SimulatePrincipalPolicy",
        "missing": missing,
        "principal": principal,
        "status": status,
    }


def _check_profiles(batch: Any) -> list[dict[str, Any]]:
    result = []
    for expected_environment in COMPUTE_ENVIRONMENTS:
        profile = expected_environment["profile"]
        environment = _one(
            batch.describe_compute_environments(
                computeEnvironments=[expected_environment["name"]]
            ).get("computeEnvironments"),
            expected_environment["name"],
        )
        environment_arn = environment.get("computeEnvironmentArn")
        resources = environment.get("computeResources")
        if (
            environment.get("computeEnvironmentName") != expected_environment["name"]
            or environment.get("state") != "ENABLED"
            or environment.get("status") != "VALID"
            or not isinstance(environment_arn, str)
            or not isinstance(resources, Mapping)
            or resources.get("instanceTypes") != [expected_environment["instance_type"]]
        ):
            raise ValueError(f"compute environment contract mismatch for {profile}")

        queue_reports: dict[str, dict[str, Any]] = {}
        for expected_queue in (item for item in QUEUES if item["profile"] == profile):
            queue = _one(
                batch.describe_job_queues(jobQueues=[expected_queue["name"]]).get(
                    "jobQueues"
                ),
                expected_queue["name"],
            )
            if (
                queue.get("jobQueueName") != expected_queue["name"]
                or queue.get("state") != "ENABLED"
                or queue.get("status") != "VALID"
                or queue.get("priority") != expected_queue["priority"]
                or queue.get("computeEnvironmentOrder")
                != [{"order": 1, "computeEnvironment": environment_arn}]
            ):
                raise ValueError(f"job queue contract mismatch for {expected_queue['name']}")
            queue_reports[f"{expected_queue['kind']}_queue"] = {
                "name": expected_queue["name"],
                "priority": expected_queue["priority"],
                "status": "visible",
            }
        result.append(
            {
                "compute_environment": {
                    "instance_type": expected_environment["instance_type"],
                    "name": expected_environment["name"],
                    "status": "visible",
                },
                "profile": profile,
                **queue_reports,
                "status": "ready",
            }
        )
    return result


def _check_s3(s3: Any, config: PreflightConfig) -> dict[str, Any]:
    try:
        s3.head_bucket(Bucket=config.bucket)
        location = s3.get_bucket_location(Bucket=config.bucket).get("LocationConstraint")
        actual_region = location or "us-east-1"
        if actual_region != config.region:
            raise ValueError(
                f"S3 bucket region mismatch: expected {config.region}, got {actual_region}"
            )
        listing = s3.list_objects_v2(
            Bucket=config.bucket,
            Prefix=config.prefix,
            MaxKeys=1,
        )
        return {
            "bucket": config.bucket,
            "prefix": config.prefix,
            "region": actual_region,
            "sample_count": int(listing.get("KeyCount", 0)),
            "status": "visible",
            "writes_performed": False,
        }
    except BaseException as error:
        return {"bucket": config.bucket, "prefix": config.prefix, **_error(error)}


def _check_ecr(ecr: Any, config: PreflightConfig) -> dict[str, Any]:
    try:
        repository = _one(
            ecr.describe_repositories(
                registryId=config.account_id,
                repositoryNames=[config.ecr_repository],
            ).get("repositories"),
            config.ecr_repository,
        )
    except BaseException as error:
        return {
            "images": {},
            "repository": config.ecr_repository,
            **_error(error),
        }
    expected_uri = f"{config.account_id}.dkr.ecr.{config.region}.amazonaws.com/{config.ecr_repository}"
    if (
        repository.get("repositoryName") != config.ecr_repository
        or repository.get("repositoryUri") != expected_uri
    ):
        return {
            "images": {},
            "repository": config.ecr_repository,
            "status": "failed",
            "error": "ECR repository identity does not match account and region",
        }
    images: dict[str, dict[str, str]] = {}
    for tag in FORMAL_IMAGE_TAGS:
        try:
            response = ecr.batch_get_image(
                registryId=config.account_id,
                repositoryName=config.ecr_repository,
                imageIds=[{"imageTag": tag}],
            )
            failures = response.get("failures", [])
            if failures:
                if (
                    len(failures) == 1
                    and failures[0].get("failureCode") == "ImageNotFound"
                ):
                    images[tag] = {"status": "missing"}
                    continue
                raise ValueError(f"ECR returned failures for {tag!r}: {failures!r}")
            image = _one(response.get("images"), tag)
            image_id = image.get("imageId")
            digest = image_id.get("imageDigest") if isinstance(image_id, Mapping) else None
            if not isinstance(digest, str) or not digest.startswith("sha256:"):
                raise ValueError(f"ECR image {tag!r} has no digest")
            images[tag] = {"digest": digest, "status": "visible"}
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            if code == "ImageNotFoundException":
                images[tag] = {"status": "missing"}
            else:
                images[tag] = _error(error)
        except BaseException as error:
            images[tag] = _error(error)
    statuses = {item["status"] for item in images.values()}
    status = "visible" if statuses == {"visible"} else "pending"
    if "blocked" in statuses:
        status = "blocked"
    elif "failed" in statuses:
        status = "failed"
    return {
        "images": images,
        "repository": config.ecr_repository,
        "repository_uri": expected_uri,
        "status": status,
        "writes_performed": False,
    }


def _find_aim_pid(port: int) -> int | None:
    proc = Path("/proc")
    for item in proc.iterdir():
        if not item.name.isdigit():
            continue
        try:
            command = (item / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (OSError, UnicodeDecodeError):
            continue
        if "aim" in command and "server" in command and str(port) in command:
            return int(item.name)
    return None


def probe_aim(endpoint: str) -> dict[str, Any]:
    parsed = urlparse(endpoint)
    if parsed.scheme != "aim" or parsed.hostname is None or parsed.port is None:
        raise ValueError("Aim endpoint must be aim://HOST:PORT")
    with socket.create_connection((parsed.hostname, parsed.port), timeout=2):
        pass
    return {"healthy": True, "pid": _find_aim_pid(parsed.port)}


def _check_aim(
    config: PreflightConfig,
    aim_probe: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    scratch = config.aim_repo.resolve()
    main = config.main_repo.resolve()
    isolated = scratch != main and main not in scratch.parents and scratch not in main.parents
    try:
        if not isolated:
            raise ValueError("Aim scratch repository is not isolated from the main repository")
        if not (scratch / ".aim").is_dir():
            raise ValueError(f"Aim repository metadata is missing under {scratch}")
        health = aim_probe(config.aim_endpoint)
        if health.get("healthy") is not True:
            raise ValueError("Aim scratch endpoint is unhealthy")
        return {
            "endpoint": config.aim_endpoint,
            "healthy": True,
            "isolated": True,
            "main_repo": str(main),
            "pid": health.get("pid"),
            "repo": str(scratch),
            "status": "ready",
        }
    except BaseException as error:
        return {
            "endpoint": config.aim_endpoint,
            "isolated": isolated,
            "main_repo": str(main),
            "repo": str(scratch),
            **_error(error),
        }


def run_preflight(
    session: Any,
    config: PreflightConfig,
    *,
    aim_probe: Callable[[str], dict[str, Any]] = probe_aim,
) -> dict[str, Any]:
    sts = session.client("sts")
    identity = sts.get_caller_identity()
    actual_account = identity.get("Account")
    account = {
        "actual": actual_account,
        "expected": config.account_id,
        "status": "match" if actual_account == config.account_id else "mismatch",
    }
    actual_region = getattr(session, "region_name", None)
    region = {
        "actual": actual_region,
        "expected": config.region,
        "status": "match" if actual_region == config.region else "mismatch",
    }

    try:
        profiles: object = _check_profiles(session.client("batch"))
    except BaseException as error:
        profiles = _error(error)
    s3 = _check_s3(session.client("s3"), config)
    ecr = _check_ecr(session.client("ecr"), config)
    identity_arn = identity.get("Arn")
    iam = (
        _check_iam(session.client("iam"), identity_arn)
        if isinstance(identity_arn, str)
        else {"status": "failed", "error": "STS identity ARN is missing"}
    )
    aim = _check_aim(config, aim_probe)

    statuses = [
        account["status"],
        region["status"],
        profiles.get("status") if isinstance(profiles, dict) else "ready",
        s3["status"],
        ecr["status"],
        iam["status"],
        aim["status"],
    ]
    if "blocked" in statuses:
        status = "blocked"
    elif any(item in {"failed", "mismatch"} for item in statuses):
        status = "failed"
    elif any(item in {"pending", "denied"} for item in statuses):
        status = "needs_context"
    else:
        status = "ready"
    return {
        "account": account,
        "aim": aim,
        "ecr": ecr,
        "iam": iam,
        "profiles": profiles,
        "region": region,
        "s3": s3,
        "schema_version": 1,
        "status": status,
        "writes_performed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Task 7 AWS/Aim preflight.")
    parser.add_argument("--aim-repo", required=True, type=Path)
    parser.add_argument("--main-repo", required=True, type=Path)
    parser.add_argument("--aim-endpoint", default="aim://127.0.0.1:53801")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = PreflightConfig(
        aim_repo=arguments.aim_repo,
        main_repo=arguments.main_repo,
        aim_endpoint=arguments.aim_endpoint,
    )
    report = run_preflight(boto3.Session(region_name=config.region), config)
    rendered = stable_json(report)
    print(rendered)
    if arguments.output is not None:
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] in {"ready", "needs_context"} else 1


if __name__ == "__main__":
    sys.exit(main())
