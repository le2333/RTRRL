from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import boto3

from trainer_infra.aws_profiles import PROFILES
from trainer_infra.ecr import BotoEcrCatalogReader
from trainer_infra.facility_control import FacilityControl, load_facility_control
from trainer_infra.image_catalog import load_catalog_index
from trainer_infra.models import ScriptCatalog


ROOT = Path(__file__).resolve().parents[4]
EXPECTED_CATALOG = ROOT / "rtrrl" / "infra" / "mock-trainer" / "scripts" / "index.yaml"


class DeployRequest:
    def __init__(
        self,
        *,
        register: bool = False,
        confirm_account: str | None = None,
        cpu_digest: str | None = None,
        gpu_digest: str | None = None,
    ) -> None:
        self.register = register
        self.confirm_account = confirm_account
        self.cpu_digest = cpu_digest
        self.gpu_digest = gpu_digest


def _stable_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _registry(control: FacilityControl) -> str:
    return f"{control.account_id}.dkr.ecr.{control.region}.amazonaws.com"


def _tagged_image(control: FacilityControl, kind: str) -> str:
    tag = control.cpu_image_tag if kind == "cpu" else control.gpu_image_tag
    return f"{_registry(control)}/{control.ecr_repository}:{tag}"


def _validated_digest(
    image: str | None,
    kind: str,
    control: FacilityControl,
) -> tuple[str, str]:
    if image is None:
        raise ValueError(f"{kind} digest image is required for registration")
    pattern = re.compile(
        rf"{re.escape(_registry(control))}/"
        rf"{re.escape(control.ecr_repository)}@sha256:([0-9a-f]{{64}})\Z"
    )
    match = pattern.fullmatch(image)
    if match is None:
        raise ValueError(
            f"{kind} digest must be {_registry(control)}/"
            f"{control.ecr_repository}@sha256:<64 lowercase hex>"
        )
    return image, match.group(1)


def _verify_digest_catalogs(
    session: Any,
    control: FacilityControl,
    images: dict[str, tuple[str, str]],
) -> None:
    expected = load_catalog_index(EXPECTED_CATALOG)
    if expected.protocol_version != "1" or set(expected.scripts) != {
        "brax_ppo_acceptance"
    }:
        raise ValueError("local expected catalog identity is invalid")
    descriptor = expected.scripts["brax_ppo_acceptance"]
    if (
        descriptor.sdk_protocol_version != "1"
        or descriptor.objective.metric != "eval/episode_return"
        or descriptor.environments != ("inverted_pendulum",)
        or set(descriptor.fields)
        != {
            "seed",
            "learning_rate",
            "num_envs",
            "episode_length",
            "failure_mode",
        }
    ):
        raise ValueError("local expected catalog contract is invalid")
    expected_bytes = _catalog_bytes(expected)
    reader = BotoEcrCatalogReader(
        session.client("ecr"),
        account_id=control.account_id,
        region=control.region,
    )
    for image, _digest_hex in images.values():
        verified = reader.resolve_and_fetch(image)
        actual = verified.catalog
        if (
            verified.reference != image
            or verified.repository != f"{_registry(control)}/{control.ecr_repository}"
            or verified.digest != image.rsplit("@", 1)[1]
            or type(actual) is not type(expected)
            or actual != expected
            or _catalog_bytes(actual) != expected_bytes
        ):
            raise ValueError(f"ECR digest catalog verification failed for {image}")


def _catalog_bytes(catalog: ScriptCatalog) -> bytes:
    return catalog.model_dump_json(exclude_none=True).encode("utf-8")


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
    control: FacilityControl,
    *,
    images: dict[str, tuple[str, str]],
) -> list[str]:
    batch = session.client("batch")
    arns = []
    for name, profile in PROFILES.items():
        kind = "gpu" if profile.gpus else "cpu"
        image, digest_hex = images[kind]
        response = batch.register_job_definition(
            jobDefinitionName=f"trainer-{name}-{digest_hex}",
            type="container",
            platformCapabilities=["EC2"],
            retryStrategy={"attempts": 1},
            containerProperties={
                "command": ["python", "/opt/trainer/worker.py"],
                "environment": [
                    {"name": "TRAINER_WORKER_PROTOCOL_VERSION", "value": "1"}
                ],
                "executionRoleArn": control.execution_role_arn,
                "image": image,
                "jobRoleArn": control.job_role_arn,
                "logConfiguration": {"logDriver": "awslogs"},
                "resourceRequirements": _resource_requirements(
                    profile.vcpus,
                    profile.memory_mib,
                    profile.gpus,
                ),
            },
        )
        arn = response.get("jobDefinitionArn")
        if not isinstance(arn, str) or not arn:
            raise ValueError(f"Batch returned no ARN for profile {name}")
        arns.append(arn)
    return arns


def deploy(
    config: DeployRequest,
    *,
    control: FacilityControl,
    session: Any | None = None,
) -> dict[str, Any]:
    requested = {"register": config.register}
    if not config.register:
        return {
            "mode": "dry-run",
            "planned_images": [
                _tagged_image(control, "cpu"),
                _tagged_image(control, "gpu"),
            ],
            "planned_profiles": list(PROFILES),
            "requested": requested,
            "retry_attempts": 1,
            "submission_supported": False,
        }

    if config.confirm_account != control.account_id:
        raise ValueError(
            f"mutating phases require --confirm-account {control.account_id}"
        )
    active_session = session or boto3.Session(region_name=control.region)
    identity = active_session.client("sts").get_caller_identity()
    if identity.get("Account") != control.account_id:
        raise ValueError("AWS account does not match facility control")
    if active_session.region_name != control.region:
        raise ValueError("AWS region does not match facility control")

    images = {
        "cpu": _validated_digest(config.cpu_digest, "CPU", control),
        "gpu": _validated_digest(config.gpu_digest, "GPU", control),
    }
    if images["cpu"][1] == images["gpu"][1]:
        raise ValueError("CPU and GPU image digests must be distinct")
    _verify_digest_catalogs(active_session, control, images)
    definitions = _register_definitions(active_session, control, images=images)
    return {
        "digests": {kind: image for kind, (image, _digest) in images.items()},
        "job_definitions": definitions,
        "mode": "execute",
        "requested": requested,
        "retry_attempts": 1,
        "submission_supported": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan fixed facility images or explicitly register digest-bound definitions."
        )
    )
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--confirm-account")
    parser.add_argument("--cpu-digest")
    parser.add_argument("--gpu-digest")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = DeployRequest(
        register=arguments.register,
        confirm_account=arguments.confirm_account,
        cpu_digest=arguments.cpu_digest,
        gpu_digest=arguments.gpu_digest,
    )
    control = load_facility_control(arguments.control)
    print(_stable_json(deploy(config, control=control)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
