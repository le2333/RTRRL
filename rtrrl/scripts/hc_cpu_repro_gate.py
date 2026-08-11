"""Reconstruct and run the archived HalfCheetah CPU reproduction gate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path

BUCKET = "rtrrl-artifacts-007122174918"
PREFIX = "experiments/rtrrl-halfcheetah/platform/assets"
OAI_METRICS_SHA256 = "cf972eee842f5e379a5f68f853c548ce92ca74ef914c8c10d134758fa92422bc"
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


def checkpoint_report(handoff: Path, round_root: Path) -> dict[int, dict[str, object]]:
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
    leaf_catalog = {}
    cursor = 0
    for name, component in zip(CARRY_NAMES, carry_200, strict=True):
        path_leaves, _ = jax.tree_util.tree_flatten_with_path(component)
        leaves = [leaf for _, leaf in path_leaves]
        spans.append({"block": name, "leaf_start": cursor, "leaf_end": cursor + len(leaves) - 1})
        for offset, (path, leaf) in enumerate(path_leaves):
            index = cursor + offset
            leaf_catalog[index] = {
                "block": name,
                "path": jax.tree_util.keystr(path),
                "shape": list(leaf.shape),
                "dtype": str(leaf.dtype),
            }
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
        "leaf_catalog": leaf_catalog,
    }
    print(json.dumps(report, default=str), flush=True)
    return leaf_catalog


def raw_sha256(value: object) -> str:
    import numpy as np

    return hashlib.sha256(np.asarray(value).tobytes(order="C")).hexdigest()


def one_step_report(round_root: Path, leaf_catalog: dict[int, dict[str, object]]) -> bool:
    import csv

    import jax
    import numpy as np

    runner_path = next(
        (round_root / "artifacts/hcbase").glob(
            "rtrrl_halfcheetah_instability_probe_*/rtrrl_halfcheetah_runner_v1_20260809.py"
        )
    )
    spec = importlib.util.spec_from_file_location("hc_one_step_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import archived runner: {runner_path}")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    sys.path.insert(0, str(round_root / "work"))
    from teaching_runner_lib import make_runner_class

    checkpoint = round_root / "results/exp3_bp_fromscratch_3seed/batch_checkpoints/step_0200000.pkl"
    carry_200, payload = runner.load_checkpoint(checkpoint)
    runner_class = make_runner_class(runner.Runner, "bp", "bp", 1.0, 1.0)
    replay = runner_class(seed=1, config=payload["config"])

    def batched_step(carry: object) -> tuple[object, object]:
        return jax.vmap(lambda item: replay.step(item, None))(carry)

    carry_201, step_output = batched_step(carry_200)
    metrics, finite = step_output
    jax.block_until_ready(carry_201)
    reference_path = Path(__file__).with_name("hc_bp_step200001_oai_reference.tsv")
    with reference_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    expected = {
        (row["carry"], int(row["leaf"])): row["sha256_raw_c_order_bytes"] for row in rows
    }
    mismatches = []
    for carry_name, carry in (("carry_200000", carry_200), ("carry_200001", carry_201)):
        for index, value in enumerate(jax.tree_util.tree_leaves(carry)):
            host = np.asarray(jax.device_get(value))
            actual = raw_sha256(host)
            wanted = expected[(carry_name, index)]
            if actual != wanted:
                detail = {
                    "carry": carry_name,
                    "index": index,
                    **leaf_catalog[index],
                    "expected_sha256": wanted,
                    "actual_sha256": actual,
                }
                if host.size <= 512:
                    detail["actual_values"] = host.tolist()
                mismatches.append(detail)
    metrics_sha256 = raw_sha256(jax.device_get(metrics))
    report = {
        "event": "one_step_comparison",
        "host": {"platform": platform.platform(), "processor": platform.processor()},
        "metrics_sha256": metrics_sha256,
        "expected_metrics_sha256": OAI_METRICS_SHA256,
        "metrics_equal": metrics_sha256 == OAI_METRICS_SHA256,
        "metrics": np.asarray(jax.device_get(metrics)).tolist(),
        "finite": np.asarray(jax.device_get(finite)).tolist(),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }
    print(json.dumps(report), flush=True)
    return not mismatches and metrics_sha256 == OAI_METRICS_SHA256


def run_prepared(root: Path) -> int:
    handoffs = list((root / "handoff").glob("rtrrl_server_handoff_*"))
    if len(handoffs) != 1:
        raise RuntimeError(f"expected one handoff directory, found {len(handoffs)}")
    handoff = handoffs[0]
    round_root = root / "round" / "hc_bp_round_20260810"
    subprocess.run([sys.executable, str(handoff / "env/verify_env.py")], check=True)
    leaf_catalog = checkpoint_report(handoff, round_root)
    gate_returncode = 0 if one_step_report(round_root, leaf_catalog) else 1
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
