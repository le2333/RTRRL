"""AWS Batch bootstrap and artifact uploader for S1 DiagSSM experiments."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
import venv
import zipfile
from pathlib import Path

import boto3
import yaml


REGION = "eu-north-1"
BUCKET = "rtrrl-artifacts-007122174918"
ASSET_PREFIX = "experiments/rtrrl-halfcheetah/platform/assets"
WHEELS = "rtrrl_wheels_cp313_linux.zip"
WHEELS_SHA256 = "3efef86ccab2a558511791fcc7bccd8197adb2706a782bb854e8172c1671c22d"
PACKAGES = (
    "jax==0.5.0",
    "jaxlib==0.5.0",
    "brax==0.10.5",
    "distrax==0.1.5",
    "flax==0.10.3",
    "optax==0.2.4",
    "numpy==2.2.2",
    "scipy==1.15.1",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_python(root: Path, s3) -> Path:
    archive = root / WHEELS
    s3.download_file(BUCKET, f"{ASSET_PREFIX}/{WHEELS}", str(archive))
    actual = sha256_file(archive)
    print(json.dumps({"event": "asset", "name": WHEELS, "sha256": actual}), flush=True)
    if actual != WHEELS_SHA256:
        raise RuntimeError(f"wheel archive checksum mismatch: {actual}")
    wheel_root = root / "wheelhouse"
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (wheel_root / member.filename).resolve()
            if not target.is_relative_to(wheel_root.resolve()):
                raise RuntimeError(f"unsafe wheel archive member: {member.filename}")
        bundle.extractall(wheel_root)
    wheelhouse = wheel_root / "wheels313"
    environment = root / "venv"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / "bin/python"
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-index", "--find-links", str(wheelhouse), *PACKAGES],
        check=True,
    )
    return python


def split_s3(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an S3 URI: {uri}")
    return tuple(uri.removeprefix("s3://").split("/", 1))  # type: ignore[return-value]


def upload_changed(s3, output: Path, destination: str, uploaded: dict[str, str]) -> None:
    if not output.exists():
        return
    bucket, prefix = split_s3(destination)
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output).as_posix()
        digest = sha256_file(path)
        if uploaded.get(relative) == digest:
            continue
        s3.upload_file(str(path), bucket, f"{prefix}/{relative}")
        uploaded[relative] = digest
        print(json.dumps({"event": "artifact_uploaded", "path": relative, "sha256": digest}), flush=True)


def command_for(config: dict, python: Path, output: Path) -> list[str]:
    mode = os.environ.get("S1_MODE", "s0")
    if mode == "s0":
        gate = config["gate"]
        return [
            str(python),
            "/opt/rtrrl/s1_diagssm_s0_gate.py",
            "--rtol",
            str(gate["relative_tolerance"]),
            "--atol",
            str(gate["absolute_tolerance"]),
            "--output",
            str(output / "s0_gate.json"),
        ]
    if mode != "train":
        raise ValueError(f"unknown S1_MODE: {mode}")
    training = config["training"]
    learner = "rtrl" if training["method"] == "exact-online-rtrl" else "bptt128"
    steps = int(os.environ.get("S1_STEPS", training["total_steps"]))
    command = [
        str(python),
        "/opt/rtrrl/s1_diagssm_runner.py",
        "--learner",
        learner,
        "--seed",
        str(config["seed"]),
        "--steps",
        str(steps),
        "--output",
        str(output),
    ]
    resume = os.environ.get("S1_RESUME")
    if resume:
        command.extend(("--resume", resume))
    return command


def main() -> int:
    config_path = Path(os.environ.get("S1_CONFIG", "/opt/rtrrl/s1_config.yml"))
    config = yaml.safe_load(config_path.read_text())
    destination = os.environ.get("S1_OUTPUT_PREFIX", config["artifacts"]["output_prefix"])
    root = Path(tempfile.mkdtemp(prefix="s1-diagssm-"))
    output = root / "output"
    output.mkdir()
    (output / "submitted_config.yml").write_text(config_path.read_text())
    s3 = boto3.client("s3", region_name=REGION)
    python = exact_python(root, s3)
    command = command_for(config, python, output)
    print(json.dumps({"event": "entry", "command": command, "destination": destination}), flush=True)
    process = subprocess.Popen(command)
    uploaded: dict[str, str] = {}
    while process.poll() is None:
        upload_changed(s3, output, destination, uploaded)
        time.sleep(5)
    upload_changed(s3, output, destination, uploaded)
    return int(process.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
