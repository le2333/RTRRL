"""Reconstruct and run the archived HalfCheetah CPU reproduction gate."""

from __future__ import annotations

import hashlib
import importlib.util
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
CARRY_NAMES = (
    "env_state",
    "W_R",
    "tau",
    "W_A",
    "W_C",
    "b_C",
    "B_A",
    "B_C",
    "h",
    "J_W",
    "J_tau",
    "z_A",
    "z_C_W",
    "z_C_b",
    "z_R_W",
    "z_R_tau",
    "action",
    "V_prev",
    "PRNG_key",
    "fixed_reset_key",
    "episodic_I",
    "episode_return",
    "episode_length",
)


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


def checkpoint_report(handoff: Path, round_root: Path) -> None:
    import jax

    runner_path = next(
        (round_root / "artifacts/hcbase").glob(
            "rtrrl_halfcheetah_instability_probe_*/rtrrl_halfcheetah_runner_v1_20260809.py"
        )
    )
    spec = importlib.util.spec_from_file_location("hc_archived_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import archived runner: {runner_path}")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    checkpoint_dir = round_root / "results/exp3_bp_fromscratch_3seed/batch_checkpoints"
    carry_200, payload_200 = runner.load_checkpoint(checkpoint_dir / "step_0200000.pkl")
    carry_220, payload_220 = runner.load_checkpoint(checkpoint_dir / "step_0220000.pkl")
    spans = []
    cursor = 0
    for name, component in zip(CARRY_NAMES, carry_200, strict=True):
        leaves = jax.tree_util.tree_leaves(component)
        spans.append({"block": name, "leaf_start": cursor, "leaf_end": cursor + len(leaves) - 1})
        cursor += len(leaves)
    report = {
        "event": "checkpoint_metadata",
        "handoff_sha256": hashlib.sha256(
            (handoff / "scripts/verify_bp_200_to_220.py").read_bytes()
        ).hexdigest(),
        "step_200k": {
            key: payload_200.get(key) for key in ("format", "step", "code_sha256", "config")
        },
        "step_220k": {
            key: payload_220.get(key) for key in ("format", "step", "code_sha256", "config")
        },
        "config_equal": payload_200.get("config") == payload_220.get("config"),
        "treedef_equal": payload_200.get("treedef") == payload_220.get("treedef"),
        "leaf_count_200k": len(jax.tree_util.tree_leaves(carry_200)),
        "leaf_count_220k": len(jax.tree_util.tree_leaves(carry_220)),
        "leaf_spans": spans,
    }
    print(json.dumps(report, default=str), flush=True)


def run_prepared(root: Path) -> int:
    handoffs = list((root / "handoff").glob("rtrrl_server_handoff_*"))
    if len(handoffs) != 1:
        raise RuntimeError(f"expected one handoff directory, found {len(handoffs)}")
    handoff = handoffs[0]
    round_root = root / "round" / "hc_bp_round_20260810"
    subprocess.run([sys.executable, str(handoff / "env/verify_env.py")], check=True)
    checkpoint_report(handoff, round_root)
    completed = subprocess.run(
        [
            sys.executable,
            str(handoff / "scripts/verify_bp_200_to_220.py"),
            "--round-root",
            str(round_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr, flush=True)
    if completed.returncode != 0:
        print(json.dumps({"event": "gate_complete", "returncode": completed.returncode}))
        return completed.returncode
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        print(json.dumps({"event": "gate_complete", "returncode": 2, "error": str(error)}))
        return 2
    gate_returncode = 0 if result.get("equal") and result.get("max_abs_diff") == 0.0 else 1
    print(json.dumps({"event": "gate_complete", "returncode": gate_returncode}))
    return gate_returncode


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
