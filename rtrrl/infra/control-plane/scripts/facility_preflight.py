from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Mapping

import boto3
from botocore.exceptions import ClientError

from trainer_infra.adapters.aws_batch import (
    AwsBatchPreflight,
    AwsInfrastructurePreflightContract,
)
from trainer_infra.aim_scratch import validate_aim_scratch
from trainer_infra.facility_control import FacilityControl, load_facility_control


REQUIRED_ACTIONS = (
    "batch:DescribeJobs",
    "batch:RegisterJobDefinition",
    "batch:SubmitJob",
    "ecr:BatchCheckLayerAvailability",
    "ecr:BatchGetImage",
    "ecr:CompleteLayerUpload",
    "ecr:GetAuthorizationToken",
    "ecr:GetDownloadUrlForLayer",
    "ecr:InitiateLayerUpload",
    "ecr:PutImage",
    "ecr:UploadLayerPart",
    "iam:PassRole",
    "logs:DescribeLogStreams",
    "logs:GetLogEvents",
    "s3:GetObject",
    "s3:ListBucket",
    "s3:PutObject",
)
CANONICAL_REPORT = Path("/tmp/complete-facility-task7-phase-a-preflight.json")


def stable_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def write_canonical_report(
    report: object,
    *,
    path: Path = CANONICAL_REPORT,
) -> Path:
    rendered = stable_json(report) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path


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
            "availability": "unavailable",
            "actions": list(REQUIRED_ACTIONS),
            "blocking": False,
            "error": str(error),
            "method": "SimulatePrincipalPolicy",
            "principal": principal,
            "status": "unknown",
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
        "availability": "available",
        "blocking": False,
    }


def _check_profiles(batch: Any, control: FacilityControl) -> list[dict[str, Any]]:
    contract = AwsInfrastructurePreflightContract(
        subnets=control.subnets,
        security_group_ids=control.security_group_ids,
        instance_role=control.instance_role,
    )
    validated = AwsBatchPreflight(batch).validate_profiles(contract)
    return [
        {
            "compute_environment": item.profile.compute_environment,
            "compute_environment_arn": item.compute_environment_arn,
            "dev_queue": {
                "name": item.profile.dev_queue,
                "priority": 10,
            },
            "gpus": item.profile.gpus,
            "memory_mib": item.profile.memory_mib,
            "profile": item.profile.name,
            "run_queue": {
                "name": item.profile.run_queue,
                "priority": 100,
            },
            "status": "ready",
            "vcpus": item.profile.vcpus,
        }
        for item in validated
    ]


def _check_s3(s3: Any, config: FacilityControl) -> dict[str, Any]:
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
            Prefix=config.prefix + "/",
            MaxKeys=1,
        )
        return {
            "bucket": config.bucket,
            "prefix": config.prefix + "/",
            "region": actual_region,
            "sample_count": int(listing.get("KeyCount", 0)),
            "status": "visible",
            "writes_performed": False,
        }
    except BaseException as error:
        return {"bucket": config.bucket, "prefix": config.prefix + "/", **_error(error)}


def _check_ecr(ecr: Any, config: FacilityControl) -> dict[str, Any]:
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
    for tag in (config.cpu_image_tag, config.gpu_image_tag):
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


def _check_aim(
    control: FacilityControl,
    aim_validator: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    try:
        return aim_validator(control.aim)
    except BaseException as error:
        return {
            "endpoint": control.aim.endpoint,
            "repo": str(control.aim.repo),
            **_error(error),
        }


def run_preflight(
    session: Any,
    config: FacilityControl,
    *,
    aim_validator: Callable[[Any], dict[str, Any]] = validate_aim_scratch,
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
        profiles: object = _check_profiles(session.client("batch"), config)
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
    aim = _check_aim(config, aim_validator)

    statuses = [
        account["status"],
        region["status"],
        profiles.get("status") if isinstance(profiles, dict) else "ready",
        s3["status"],
        ecr["status"],
        iam["status"],
        aim["status"],
    ]
    blocking_statuses = [item for item in statuses if item != "unknown"]
    if any(item in {"blocked", "failed", "mismatch"} for item in blocking_statuses):
        status = "blocked"
    else:
        status = "pass"
    return {
        "account": account,
        "aim": aim,
        "caller": {
            "account": actual_account,
            "arn": identity_arn,
            "region": actual_region,
        },
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
    parser.add_argument("--control", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = load_facility_control(arguments.control)
    report = run_preflight(boto3.Session(region_name=config.region), config)
    rendered = stable_json(report)
    print(rendered)
    write_canonical_report(report)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
