from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import boto3

from trainer_infra.image_catalog import encode_catalog_file


ACCOUNT_ID = "007122174918"
REGION = "eu-north-1"
ECR_REPOSITORY = "rtrrl"
ROOT = Path(__file__).resolve().parents[4]
REGISTRY = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com"
TAGS = {
    "cpu": "memorax-rtrl-facility-cpu",
    "gpu": "memorax-rtrl-facility-gpu",
}
DOCKERFILES = {
    "cpu": ROOT / "memo" / "infra" / "docker" / "Dockerfile.facility",
    "gpu": ROOT / "memo" / "infra" / "docker" / "Dockerfile.facility.gpu",
}
PROFILE_RESOURCES = {
    "c7am": (1, 1600, 0, "cpu"),
    "c7al": (2, 3200, 0, "cpu"),
    "c7ax": (4, 7168, 0, "cpu"),
    "g6x": (4, 12000, 1, "gpu"),
}
_DIGEST_IMAGE = re.compile(
    rf"{ACCOUNT_ID}\.dkr\.ecr\.{REGION}\.amazonaws\.com/"
    rf"{ECR_REPOSITORY}@sha256:([0-9a-f]{{64}})\Z"
)


class DeployConfig:
    def __init__(
        self,
        *,
        build: bool = False,
        push: bool = False,
        register: bool = False,
        cpu_digest: str | None = None,
        gpu_digest: str | None = None,
        job_role_arn: str | None = None,
        execution_role_arn: str | None = None,
    ) -> None:
        self.build = build
        self.push = push
        self.register = register
        self.cpu_digest = cpu_digest
        self.gpu_digest = gpu_digest
        self.job_role_arn = job_role_arn
        self.execution_role_arn = execution_role_arn


def _stable_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _tagged_image(kind: str) -> str:
    return f"{REGISTRY}/{ECR_REPOSITORY}:{TAGS[kind]}"


def _run(command: list[str], *, input_text: str | None = None) -> None:
    subprocess.run(
        command,
        check=True,
        input=input_text,
        text=True,
    )


def _build_images() -> list[str]:
    catalog = encode_catalog_file(ROOT / "memo" / "infra" / "scripts" / "index.yaml")
    images = []
    for kind in ("cpu", "gpu"):
        image = _tagged_image(kind)
        _run(
            [
                "docker",
                "build",
                "--platform",
                "linux/amd64",
                "--build-arg",
                f"TRAINER_SCRIPT_CATALOG={catalog}",
                "--tag",
                image,
                "--file",
                str(DOCKERFILES[kind]),
                str(ROOT),
            ]
        )
        images.append(image)
    return images


def _push_images(session: Any) -> dict[str, str]:
    ecr = session.client("ecr")
    authorization = ecr.get_authorization_token()["authorizationData"][0]
    user_password = base64.b64decode(authorization["authorizationToken"]).decode()
    username, password = user_password.split(":", 1)
    _run(
        ["docker", "login", "--username", username, "--password-stdin", REGISTRY],
        input_text=password,
    )
    resolved = {}
    for kind in ("cpu", "gpu"):
        _run(["docker", "push", _tagged_image(kind)])
        response = ecr.describe_images(
            registryId=ACCOUNT_ID,
            repositoryName=ECR_REPOSITORY,
            imageIds=[{"imageTag": TAGS[kind]}],
        )
        details = response.get("imageDetails", [])
        if len(details) != 1 or not isinstance(details[0].get("imageDigest"), str):
            raise ValueError(f"ECR did not return one digest for {TAGS[kind]}")
        resolved[kind] = f"{REGISTRY}/{ECR_REPOSITORY}@{details[0]['imageDigest']}"
    return resolved


def _validated_digest(image: str | None, kind: str) -> tuple[str, str]:
    if image is None:
        raise ValueError(f"{kind} digest image is required for registration")
    match = _DIGEST_IMAGE.fullmatch(image)
    if match is None:
        raise ValueError(
            f"{kind} digest must be {REGISTRY}/{ECR_REPOSITORY}@sha256:<64 lowercase hex>"
        )
    return image, match.group(1)


def _resource_requirements(vcpus: int, memory: int, gpus: int) -> list[dict[str, str]]:
    result = [
        {"type": "VCPU", "value": str(vcpus)},
        {"type": "MEMORY", "value": str(memory)},
    ]
    if gpus:
        result.append({"type": "GPU", "value": str(gpus)})
    return result


def _register_definitions(
    session: Any,
    *,
    cpu_digest: str | None,
    gpu_digest: str | None,
    job_role_arn: str | None,
    execution_role_arn: str | None,
) -> list[str]:
    cpu_image, cpu_hex = _validated_digest(cpu_digest, "CPU")
    gpu_image, gpu_hex = _validated_digest(gpu_digest, "GPU")
    if not job_role_arn or not execution_role_arn:
        raise ValueError("job and execution role ARNs are required for registration")
    images = {"cpu": (cpu_image, cpu_hex), "gpu": (gpu_image, gpu_hex)}
    batch = session.client("batch")
    arns = []
    for profile, (vcpus, memory, gpus, kind) in PROFILE_RESOURCES.items():
        image, digest_hex = images[kind]
        response = batch.register_job_definition(
            jobDefinitionName=f"trainer-{profile}-{digest_hex}",
            type="container",
            platformCapabilities=["EC2"],
            retryStrategy={"attempts": 1},
            containerProperties={
                "command": ["python", "/opt/trainer/worker.py"],
                "environment": [
                    {"name": "TRAINER_WORKER_PROTOCOL_VERSION", "value": "1"}
                ],
                "executionRoleArn": execution_role_arn,
                "image": image,
                "jobRoleArn": job_role_arn,
                "logConfiguration": {"logDriver": "awslogs"},
                "resourceRequirements": _resource_requirements(vcpus, memory, gpus),
            },
        )
        arn = response.get("jobDefinitionArn")
        if not isinstance(arn, str) or not arn:
            raise ValueError(f"Batch returned no ARN for profile {profile}")
        arns.append(arn)
    return arns


def deploy(config: DeployConfig, *, session: Any | None = None) -> dict[str, Any]:
    if config.push and not config.build:
        raise ValueError("--push requires --build in the same explicit invocation")
    if config.register and not config.push:
        _validated_digest(config.cpu_digest, "CPU")
        _validated_digest(config.gpu_digest, "GPU")

    requested = {
        "build": config.build,
        "push": config.push,
        "register": config.register,
    }
    if not any(requested.values()):
        return {
            "mode": "dry-run",
            "planned_images": [_tagged_image("cpu"), _tagged_image("gpu")],
            "planned_profiles": list(PROFILE_RESOURCES),
            "requested": requested,
            "retry_attempts": 1,
            "submission_supported": False,
        }

    active_session = session
    built: list[str] = []
    digests = {"cpu": config.cpu_digest, "gpu": config.gpu_digest}
    if config.build:
        built = _build_images()
    if config.push:
        if active_session is None:
            active_session = boto3.Session(region_name=REGION)
        digests.update(_push_images(active_session))
    definitions: list[str] = []
    if config.register:
        if active_session is None:
            active_session = boto3.Session(region_name=REGION)
        definitions = _register_definitions(
            active_session,
            cpu_digest=digests["cpu"],
            gpu_digest=digests["gpu"],
            job_role_arn=config.job_role_arn,
            execution_role_arn=config.execution_role_arn,
        )
    return {
        "built": built,
        "digests": digests,
        "job_definitions": definitions,
        "mode": "execute",
        "requested": requested,
        "retry_attempts": 1,
        "submission_supported": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Explicit facility image/job-definition deployment. With no phase flags, "
            "prints a dry-run plan."
        )
    )
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--cpu-digest")
    parser.add_argument("--gpu-digest")
    parser.add_argument("--job-role-arn")
    parser.add_argument("--execution-role-arn")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = DeployConfig(
        build=arguments.build,
        push=arguments.push,
        register=arguments.register,
        cpu_digest=arguments.cpu_digest,
        gpu_digest=arguments.gpu_digest,
        job_role_arn=arguments.job_role_arn,
        execution_role_arn=arguments.execution_role_arn,
    )
    print(_stable_json(deploy(config)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
