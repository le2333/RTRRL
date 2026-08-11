"""Reconstruct and run the archived HalfCheetah CPU reproduction gate."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path

BUCKET = "rtrrl-artifacts-007122174918"
PREFIX = "experiments/rtrrl-halfcheetah/platform/assets"
ASSETS = {
    "wheels": (
        "rtrrl_wheels_cp313_linux.zip",
        "3efef86ccab2a558511791fcc7bccd8197adb2706a782bb854e8172c1671c22d",
    ),
    "instability": (
        "rtrrl_halfcheetah_instability_probe_seed3_20260809.zip",
        "ed5ec2ce992e18307224b86a7fd71992636bb7539a601e8773cf1c4af68dfafa",
    ),
    "focused": (
        "rtrrl_hc_focused_mechanism_round_20260810.zip",
        "1f98c6bfd749521929a0dcb3531f16e329ba5da8c2f563be90fce81df70bc70e",
    ),
    "handoff": (
        "rtrrl_server_handoff_20260811.zip",
        "abc163294942147a5efc00dcb1ba2ab309961c39a073fb999c3255d845ad1e3b",
    ),
}


def download_and_extract(root: Path) -> dict[str, Path]:
    import boto3

    s3 = boto3.client("s3", region_name="eu-north-1")
    extracted = {}
    for name, (filename, expected) in ASSETS.items():
        archive = root / filename
        s3.download_file(BUCKET, f"{PREFIX}/{filename}", str(archive))
        actual = hashlib.sha256(archive.read_bytes()).hexdigest()
        print(json.dumps({"event": "asset", "name": name, "sha256": actual, "ok": actual == expected}))
        if actual != expected:
            raise RuntimeError(f"checksum mismatch for {filename}")
        destination = root / name
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                target = (destination / member.filename).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise RuntimeError(f"unsafe archive member in {filename}: {member.filename}")
            bundle.extractall(destination)
        extracted[name] = destination
    return extracted


def create_exact_environment(root: Path, wheels: Path) -> Path:
    """Create the numerical environment only from the archived wheel bundle."""
    wheelhouse = wheels / "wheels313"
    if not wheelhouse.is_dir():
        raise RuntimeError(f"archived wheelhouse is missing: {wheelhouse}")
    environment = root / ".venv-rtrrl-cpu"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / "bin/python"
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(wheelhouse),
            "jax==0.5.0",
            "jaxlib==0.5.0",
            "brax==0.10.5",
            "distrax==0.1.5",
            "flax==0.10.3",
            "optax==0.2.4",
            "numpy==2.2.2",
            "scipy==1.15.1",
        ],
        check=True,
    )
    return python


def prepare_workspace(root: Path, extracted: dict[str, Path]) -> tuple[Path, Path]:
    handoff = next(extracted["handoff"].glob("rtrrl_server_handoff_*"))
    instability = next(extracted["instability"].glob("rtrrl_halfcheetah_instability_probe_*"))
    focused = next(extracted["focused"].glob("rtrrl_hc_focused_mechanism_round_*"))
    round_root = root / "round" / "hc_bp_round_20260810"
    for relative in ("artifacts/hcbase", "work", "results", "scripts"):
        (round_root / relative).mkdir(parents=True, exist_ok=True)
    shutil.copytree(instability, round_root / "artifacts/hcbase" / instability.name)
    shutil.copytree(focused / "results", round_root / "results", dirs_exist_ok=True)
    shutil.copytree(focused / "scripts", round_root / "scripts", dirs_exist_ok=True)
    shutil.copy2(handoff / "scripts/teaching_runner_lib.py", round_root / "work/teaching_runner_lib.py")
    old = "Path('/mnt/data/hc_bp_round_20260810')"
    new = f"Path({str(round_root)!r})"
    for script in (round_root / "scripts").glob("*.py"):
        source = script.read_text()
        if old in source:
            script.write_text(source.replace(old, new))
    return handoff, round_root


def run_prepared(root: Path) -> int:
    handoffs = list((root / "handoff").glob("rtrrl_server_handoff_*"))
    if len(handoffs) != 1:
        raise RuntimeError(f"expected one handoff directory, found {len(handoffs)}")
    handoff = handoffs[0]
    round_root = root / "round" / "hc_bp_round_20260810"
    subprocess.run([sys.executable, str(handoff / "env/verify_env.py")], check=True)
    completed = subprocess.run(
        [
            sys.executable,
            str(handoff / "scripts/verify_bp_200_to_220.py"),
            "--round-root",
            str(round_root),
        ],
        check=False,
    )
    print(json.dumps({"event": "gate_complete", "returncode": completed.returncode}))
    return completed.returncode


def main() -> int:
    prepared_root = os.environ.get("RTRRL_HC_PREPARED_ROOT")
    if prepared_root:
        return run_prepared(Path(prepared_root))

    root = Path(tempfile.mkdtemp(prefix="rtrrl-hc-gate-"))
    extracted = download_and_extract(root)
    prepare_workspace(root, extracted)
    python = create_exact_environment(root, extracted["wheels"])
    environment = os.environ.copy()
    environment["RTRRL_HC_PREPARED_ROOT"] = str(root)
    completed = subprocess.run(
        [str(python), str(Path(__file__).resolve())], env=environment, check=False
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
