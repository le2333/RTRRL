from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from deployment.contract import Catalog
from worker import objects
from worker.envelope import WorkerEnvelope

CATALOG_PATH = Path("/opt/trainer/catalog.json")
ARTIFACTS_DIRECTORY = "artifacts"
RESULT_FILENAME = "result.json"


class WorkerError(RuntimeError):
    """A run in this manifest did not complete."""


def load_catalog() -> Catalog:
    path = Path(os.environ.get("TRAINER_CATALOG", CATALOG_PATH))
    return Catalog.model_validate(json.loads(path.read_text(encoding="utf-8")))


def run_manifest(manifest_uri: str, workspace: Path) -> None:
    """Execute every run serially and publish successful local artifacts."""
    catalog = load_catalog()
    manifest = json.loads(objects.get_bytes(manifest_uri))
    workspace.mkdir(parents=True, exist_ok=True)
    for config_uri in manifest["runs"]:
        config = WorkerEnvelope.model_validate(
            json.loads(objects.get_bytes(config_uri))
        )
        scratch = Path(tempfile.mkdtemp(prefix="run-", dir=workspace))
        _execute(config, catalog, scratch)
        artifacts = _upload_artifacts(config, scratch)
        _publish_result(config, artifacts)
        shutil.rmtree(scratch)


def _execute(config: WorkerEnvelope, catalog: Catalog, scratch: Path) -> None:
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
        raise WorkerError(f"run {config.identity.run_id} exited with exit code {code}")


def _upload_artifacts(config: WorkerEnvelope, scratch: Path) -> list[str]:
    directory = scratch / ARTIFACTS_DIRECTORY
    if not directory.is_dir():
        raise WorkerError(
            f"run {config.identity.run_id} produced no artifact directory"
        )
    paths = sorted(path for path in directory.rglob("*") if path.is_file())
    relative_names = [path.relative_to(directory).as_posix() for path in paths]
    root = config.artifacts.root.rstrip("/")
    for path, relative_name in zip(paths, relative_names, strict=True):
        objects.put_file(f"{root}/{relative_name}", path)
    return relative_names


def _publish_result(config: WorkerEnvelope, artifacts: list[str]) -> None:
    payload = {
        "contract": config.contract,
        "identity": config.identity.model_dump(mode="json"),
        "success": True,
        "artifacts": artifacts,
    }
    uri = f"{config.artifacts.root.rstrip('/')}/{RESULT_FILENAME}"
    objects.put_bytes(uri, json.dumps(payload, sort_keys=True).encode())


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
