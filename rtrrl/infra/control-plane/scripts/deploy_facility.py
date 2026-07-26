"""Register the Batch job definitions a launch needs, and the log group they write to.

Queues and compute environments are not created here: they already exist and are
described by `trainer_infra.queues`. This script only binds an image digest to a
job definition per queue profile, which is the one thing that changes when a new
image is pushed.

Dry run by default. Registering requires `--register` and the account id, because
it is the first step in this repository that mutates AWS.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

import boto3
from botocore.exceptions import ClientError

from trainer_infra.queues import (
    ACCOUNT_ID,
    EXECUTION_ROLE_ARN,
    JOB_LOG_GROUP,
    JOB_ROLE_ARN,
    QUEUES,
    REGION,
    QueueBinding,
    job_definition_name,
)

REGISTRY = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com"
REPOSITORY = "rtrrl"
WORKER_COMMAND = ["python", "-m", "training_sdk.worker"]
LOG_RETENTION_DAYS = 30

_DIGEST = re.compile(
    rf"{re.escape(REGISTRY)}/{re.escape(REPOSITORY)}@sha256:(?P<hex>[0-9a-f]{{64}})\Z"
)


def _stable_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def validated_digest(image: str | None, kind: str) -> str:
    if not image:
        raise ValueError(f"--{kind}-digest is required to register job definitions")
    if _DIGEST.fullmatch(image) is None:
        raise ValueError(
            f"{kind} image must be {REGISTRY}/{REPOSITORY}@sha256:<64 lowercase hex>, "
            f"not {image!r}; a tag is not accepted because it can move"
        )
    return image


def _resource_requirements(binding: QueueBinding) -> list[dict[str, str]]:
    requirements = [
        {"type": "VCPU", "value": str(binding.vcpus_per_job)},
        {"type": "MEMORY", "value": str(binding.memory_mib)},
    ]
    if binding.gpus_per_job:
        requirements.append({"type": "GPU", "value": str(binding.gpus_per_job)})
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


def register_definitions(session: Any, *, cpu_image: str, gpu_image: str) -> list[str]:
    batch = session.client("batch")
    arns: list[str] = []
    for binding in QUEUES.values():
        image = gpu_image if binding.gpus_per_job else cpu_image
        digest = image.rsplit("@", 1)[1]
        response = batch.register_job_definition(
            jobDefinitionName=job_definition_name(binding, digest),
            type="container",
            platformCapabilities=["EC2"],
            # A failed run ends the launch, so a second attempt would only spend
            # money reproducing a failure the operator has to fix anyway.
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
                "resourceRequirements": _resource_requirements(binding),
            },
        )
        arn = response.get("jobDefinitionArn")
        if not isinstance(arn, str) or not arn:
            raise ValueError(f"Batch returned no ARN for profile {binding.profile!r}")
        arns.append(arn)
    return arns


def deploy(
    *,
    register: bool = False,
    confirm_account: str | None = None,
    cpu_digest: str | None = None,
    gpu_digest: str | None = None,
    session: Any | None = None,
) -> dict[str, Any]:
    if not register:
        return {
            "mode": "dry-run",
            "planned_definitions": [
                job_definition_name(binding, "sha256:<digest>") for binding in QUEUES.values()
            ],
            "log_group": JOB_LOG_GROUP,
            "worker_command": WORKER_COMMAND,
        }

    if confirm_account != ACCOUNT_ID:
        raise ValueError(f"registering requires --confirm-account {ACCOUNT_ID}")
    cpu_image = validated_digest(cpu_digest, "cpu")
    gpu_image = validated_digest(gpu_digest, "gpu")
    if cpu_image == gpu_image:
        raise ValueError("the CPU and GPU images must be different digests")

    active = session or boto3.Session(region_name=REGION)
    identity = active.client("sts").get_caller_identity()
    if identity.get("Account") != ACCOUNT_ID:
        raise ValueError(
            f"these credentials are for account {identity.get('Account')!r}, not {ACCOUNT_ID}"
        )

    ensure_log_group(active)
    definitions = register_definitions(active, cpu_image=cpu_image, gpu_image=gpu_image)
    return {
        "mode": "register",
        "images": {"cpu": cpu_image, "gpu": gpu_image},
        "job_definitions": definitions,
        "log_group": JOB_LOG_GROUP,
        "worker_command": WORKER_COMMAND,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--confirm-account")
    parser.add_argument("--cpu-digest")
    parser.add_argument("--gpu-digest")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    print(
        _stable_json(
            deploy(
                register=arguments.register,
                confirm_account=arguments.confirm_account,
                cpu_digest=arguments.cpu_digest,
                gpu_digest=arguments.gpu_digest,
            )
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
