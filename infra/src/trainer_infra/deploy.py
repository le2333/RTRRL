"""Bind an immutable trainer image to the existing AWS Batch facility."""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

import boto3
from botocore.exceptions import ClientError

from trainer_infra.batch import (
    ACCOUNT_ID,
    EXECUTION_ROLE_ARN,
    JOB_LOG_GROUP,
    JOB_ROLE_ARN,
    PROFILES,
    REGION,
    BatchProfile,
)

REGISTRY = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com"
REPOSITORY = "rtrrl"
WORKER_COMMAND = ["python", "-m", "worker"]
LOG_RETENTION_DAYS = 30
_IMAGE = re.compile(
    rf"{re.escape(REGISTRY)}/{REPOSITORY}@sha256:(?P<digest>[0-9a-f]{{64}})\Z"
)


def validated_image(image: str | None, variant: str) -> str:
    if image is None or _IMAGE.fullmatch(image) is None:
        raise ValueError(
            f"{variant} image must be {REGISTRY}/{REPOSITORY}@sha256:<64 lowercase hex>"
        )
    return image


def definition_name(profile: BatchProfile, image: str) -> str:
    digest = image.rsplit(":", 1)[1]
    return f"trainer-{profile.profile}-{digest}"


def resource_requirements(profile: BatchProfile) -> list[dict[str, str]]:
    requirements = [
        {"type": "VCPU", "value": str(profile.vcpus)},
        {"type": "MEMORY", "value": str(profile.memory_mib)},
    ]
    if profile.gpus:
        requirements.append({"type": "GPU", "value": str(profile.gpus)})
    return requirements


def ensure_log_group(session: Any) -> None:
    logs = session.client("logs")
    try:
        logs.create_log_group(logGroupName=JOB_LOG_GROUP)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") != "ResourceAlreadyExistsException":
            raise
    logs.put_retention_policy(
        logGroupName=JOB_LOG_GROUP,
        retentionInDays=LOG_RETENTION_DAYS,
    )


def register_definitions(
    session: Any, *, cpu_image: str, gpu_image: str | None
) -> list[str]:
    batch = session.client("batch")
    definitions = []
    for profile in PROFILES.values():
        image = gpu_image if profile.gpus else cpu_image
        if image is None:
            continue
        response = batch.register_job_definition(
            jobDefinitionName=definition_name(profile, image),
            type="container",
            platformCapabilities=["EC2"],
            retryStrategy={"attempts": 1},
            containerProperties={
                "command": WORKER_COMMAND,
                "executionRoleArn": EXECUTION_ROLE_ARN,
                "image": image,
                "jobRoleArn": JOB_ROLE_ARN,
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": JOB_LOG_GROUP,
                        "awslogs-region": REGION,
                    },
                },
                "resourceRequirements": resource_requirements(profile),
            },
        )
        definitions.append(response["jobDefinitionArn"])
    return definitions


def deploy(
    *,
    register: bool = False,
    confirm_account: str | None = None,
    cpu_image: str | None = None,
    gpu_image: str | None = None,
    session: Any | None = None,
) -> dict[str, Any]:
    if not register:
        return {
            "mode": "dry-run",
            "profiles": sorted(PROFILES),
            "worker_command": WORKER_COMMAND,
        }
    if confirm_account != ACCOUNT_ID:
        raise ValueError(f"registering requires --confirm-account {ACCOUNT_ID}")
    cpu = validated_image(cpu_image, "CPU")
    gpu = validated_image(gpu_image, "GPU") if gpu_image is not None else None
    if gpu == cpu:
        raise ValueError("CPU and GPU images must use different digests")

    active = session or boto3.Session(region_name=REGION)
    account = active.client("sts").get_caller_identity().get("Account")
    if account != ACCOUNT_ID:
        raise ValueError(f"credentials are for account {account!r}, not {ACCOUNT_ID}")
    ensure_log_group(active)
    definitions = register_definitions(active, cpu_image=cpu, gpu_image=gpu)
    return {
        "mode": "register",
        "images": {"cpu": cpu, **({"gpu": gpu} if gpu else {})},
        "job_definitions": definitions,
        "worker_command": WORKER_COMMAND,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--confirm-account")
    parser.add_argument("--cpu-image")
    parser.add_argument("--gpu-image")
    arguments = parser.parse_args(argv)
    print(
        json.dumps(
            deploy(
                register=arguments.register,
                confirm_account=arguments.confirm_account,
                cpu_image=arguments.cpu_image,
                gpu_image=arguments.gpu_image,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
