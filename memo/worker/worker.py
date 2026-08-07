from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from worker import objects
from worker.contract import CONTRACT_VERSION, Catalog, RunConfig
from worker.reporter import METRICS_FILENAME
from worker.score import compute_score

CATALOG_PATH = Path("/opt/trainer/catalog.json")


class WorkerError(RuntimeError):
    """A run in this manifest did not complete."""


def load_catalog() -> Catalog:
    path = Path(os.environ.get("TRAINER_CATALOG", CATALOG_PATH))
    catalog = Catalog.model_validate(json.loads(path.read_text(encoding="utf-8")))
    if catalog.contract != CONTRACT_VERSION:
        raise WorkerError(
            f"image catalog declares contract {catalog.contract}; "
            f"this worker implements contract {CONTRACT_VERSION}"
        )
    return catalog


def run_manifest(manifest_uri: str, workspace: Path) -> None:
    """Execute every run in a manifest serially.

    Raises ``WorkerError`` when a run does not complete, or ``ScoreError`` when
    a run completes but its metrics do not yield a usable score; either stops
    the manifest at that run.
    """
    catalog = load_catalog()
    manifest = json.loads(objects.get_bytes(manifest_uri))
    for config_uri in manifest["runs"]:
        config = RunConfig.model_validate(json.loads(objects.get_bytes(config_uri)))
        if config.contract != CONTRACT_VERSION:
            raise WorkerError(
                f"run configuration declares contract {config.contract}; "
                f"this worker implements contract {CONTRACT_VERSION}"
            )
        scratch = Path(workspace) / config.run_id
        scratch.mkdir(parents=True, exist_ok=True)
        try:
            _execute(config, catalog, scratch)
            value = compute_score(scratch / METRICS_FILENAME, config.score)
            objects.put_bytes(
                config.score.s3,
                json.dumps(
                    {"run_id": config.run_id, "trial": config.trial, "value": value}
                ).encode(),
            )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)


def _execute(config: RunConfig, catalog: Catalog, scratch: Path) -> None:
    entry = catalog.entries.get(config.entry)
    if entry is None:
        raise WorkerError(f"image catalog does not declare entry {config.entry!r}")
    config_path = scratch / "config.json"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    environment = dict(os.environ)
    environment["TRAINER_RUN_CONFIG"] = str(config_path)
    environment["TRAINER_SCRATCH"] = str(scratch)

    # A run that hangs is bounded by the job timeout the experiment declares.
    # Second-guessing that here, by reading the pace of reports, only produces
    # ways to kill a healthy run.
    code = subprocess.call(list(entry.command), env=environment)
    if code != 0:
        raise WorkerError(f"run {config.run_id} exited with exit code {code}")


def main() -> int:
    try:
        manifest = os.environ["TRAINER_MANIFEST"]
    except KeyError:
        # The whole job is driven by this variable, and a Batch job that starts
        # without it is misconfigured upstream. Say so, rather than leaving a
        # bare KeyError traceback in CloudWatch for someone to decipher.
        print(
            "worker failed: TRAINER_MANIFEST is not set; "
            "the control plane must pass the manifest location",
            file=sys.stderr,
        )
        return 1
    workspace = Path(os.environ.get("TRAINER_WORKSPACE", "/tmp/trainer"))
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        run_manifest(manifest, workspace)
    except Exception as error:  # noqa: BLE001 - the exit code is the only signal
        print(f"worker failed: {error}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
