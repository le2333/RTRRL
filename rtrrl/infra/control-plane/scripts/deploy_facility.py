from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

import boto3

from trainer_infra.aws_profiles import PROFILES
from trainer_infra.ecr import BotoEcrCatalogReader
from trainer_infra.facility_control import FacilityControl, load_facility_control
from trainer_infra.image_catalog import LABEL, decode_catalog, encode_catalog_file


ROOT = Path(__file__).resolve().parents[4]
DOCKERFILES = {
    "cpu": ROOT / "memo" / "infra" / "docker" / "Dockerfile.facility",
    "gpu": ROOT / "memo" / "infra" / "docker" / "Dockerfile.facility.gpu",
}
_PUSH_DIGEST = re.compile(r"(?:^|\s)digest:\s*(sha256:[0-9a-f]{64})(?:\s|$)")


class DeployRequest:
    def __init__(
        self,
        *,
        build: bool = False,
        push: bool = False,
        register: bool = False,
        confirm_account: str | None = None,
        cpu_digest: str | None = None,
        gpu_digest: str | None = None,
    ) -> None:
        self.build = build
        self.push = push
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


def _run_capture(
    command: list[str],
    *,
    input_text: str | None = None,
    env: dict[str, str] | None = None,
) -> str:
    result = subprocess.run(
        command,
        check=True,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return result.stdout


def _verify_image(
    kind: str,
    image: str,
    *,
    run_capture: Any = _run_capture,
) -> None:
    labels_raw = run_capture(
        ["docker", "image", "inspect", "--format", "{{json .Config.Labels}}", image]
    )
    labels = json.loads(labels_raw)
    catalog_label = labels.get(LABEL)
    if not isinstance(catalog_label, str) or not catalog_label:
        raise ValueError(f"{image} is missing catalog label {LABEL}")
    catalog = decode_catalog(catalog_label)
    if set(catalog.scripts) != {"memo_stream_ac", "memo_rtrrl"}:
        raise ValueError(f"{image} has an unexpected facility catalog")
    runtime_check = r"""
import importlib
import importlib.util
import json
from pathlib import Path

importlib.import_module("training_sdk")
importlib.import_module("experiments.memo_stream_ac.run")
importlib.import_module("experiments.memo_rtrrl.run")
has_cuda_plugin = importlib.util.find_spec("jax_cuda12_plugin") is not None
print(json.dumps({
    "jax_variant": "gpu" if has_cuda_plugin else "cpu",
    "launchers": True,
    "training_sdk": True,
    "worker": Path("/opt/trainer/worker.py").is_file(),
}))
"""
    result = json.loads(
        run_capture(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "/opt/venv/bin/python",
                image,
                "-c",
                runtime_check,
            ]
        )
    )
    expected = {
        "jax_variant": kind,
        "launchers": True,
        "training_sdk": True,
        "worker": True,
    }
    if result != expected:
        raise ValueError(f"{image} runtime contract mismatch: {result!r}")


def _build_images(control: FacilityControl) -> list[str]:
    catalog = encode_catalog_file(ROOT / "memo" / "infra" / "scripts" / "index.yaml")
    images = []
    for kind in ("cpu", "gpu"):
        image = _tagged_image(control, kind)
        _run_capture(
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
        _verify_image(kind, image)
        images.append(image)
    return images


def _push_images(
    session: Any,
    control: FacilityControl,
    *,
    run_capture: Any = _run_capture,
    reader_factory: Any = BotoEcrCatalogReader,
) -> dict[str, str]:
    ecr = session.client("ecr")
    authorization = ecr.get_authorization_token()["authorizationData"][0]
    user_password = base64.b64decode(authorization["authorizationToken"]).decode()
    username, password = user_password.split(":", 1)
    resolved: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="trainer-docker-config-") as temporary:
        environment = {**os.environ, "DOCKER_CONFIG": temporary}
        run_capture(
            [
                "docker",
                "login",
                "--username",
                username,
                "--password-stdin",
                _registry(control),
            ],
            input_text=password,
            env=environment,
        )
        reader = reader_factory(
            ecr,
            account_id=control.account_id,
            region=control.region,
        )
        for kind in ("cpu", "gpu"):
            tagged = _tagged_image(control, kind)
            output = run_capture(["docker", "push", tagged], env=environment)
            matches = _PUSH_DIGEST.findall(output)
            if len(set(matches)) != 1:
                raise ValueError(f"docker push returned no unique digest for {tagged}")
            reference = (
                f"{_registry(control)}/{control.ecr_repository}@{matches[0]}"
            )
            verified = reader.resolve_and_fetch(reference)
            if verified.reference != reference or verified.catalog is None:
                raise ValueError(f"ECR digest verification failed for {reference}")
            resolved[kind] = reference
    return resolved


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
    cpu_digest: str | None,
    gpu_digest: str | None,
) -> list[str]:
    cpu_image, cpu_hex = _validated_digest(cpu_digest, "CPU", control)
    gpu_image, gpu_hex = _validated_digest(gpu_digest, "GPU", control)
    images = {"cpu": (cpu_image, cpu_hex), "gpu": (gpu_image, gpu_hex)}
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
    requested = {
        "build": config.build,
        "push": config.push,
        "register": config.register,
    }
    if not any(requested.values()):
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

    active_session = session
    if config.push or config.register:
        if config.confirm_account != control.account_id:
            raise ValueError(
                f"mutating phases require --confirm-account {control.account_id}"
            )
        if active_session is None:
            active_session = boto3.Session(region_name=control.region)
        identity = active_session.client("sts").get_caller_identity()
        if identity.get("Account") != control.account_id:
            raise ValueError("AWS account does not match facility control")
        if active_session.region_name != control.region:
            raise ValueError("AWS region does not match facility control")

    built: list[str] = []
    digests = {"cpu": config.cpu_digest, "gpu": config.gpu_digest}
    if config.build:
        built = _build_images(control)
    if config.push:
        assert active_session is not None
        digests.update(_push_images(active_session, control))
    definitions: list[str] = []
    if config.register:
        assert active_session is not None
        definitions = _register_definitions(
            active_session,
            control,
            cpu_digest=digests["cpu"],
            gpu_digest=digests["gpu"],
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
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--confirm-account")
    parser.add_argument("--cpu-digest")
    parser.add_argument("--gpu-digest")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = DeployRequest(
        build=arguments.build,
        push=arguments.push,
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
