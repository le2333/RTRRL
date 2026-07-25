from __future__ import annotations

import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from training_sdk import objects
from training_sdk.contract import CONTRACT_VERSION, Catalog, RunConfig
from training_sdk.reporter import METRICS_FILENAME
from training_sdk.score import compute_score

CATALOG_PATH = Path("/opt/trainer/catalog.json")
TERMINATE_GRACE_SECONDS = 10.0


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


def run_manifest(
    manifest_uri: str,
    workspace: Path,
    *,
    startup_seconds: float,
    stall_factor: int,
    poll_seconds: float = 5.0,
    minimum_stall_seconds: float = 60.0,
) -> None:
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
            _execute(
                config,
                catalog,
                scratch,
                startup_seconds,
                stall_factor,
                poll_seconds,
                minimum_stall_seconds,
            )
            value = compute_score(scratch / METRICS_FILENAME, config.score)
            objects.put_bytes(
                config.score.s3,
                json.dumps(
                    {"run_id": config.run_id, "trial": config.trial, "value": value}
                ).encode(),
            )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)


def _execute(
    config: RunConfig,
    catalog: Catalog,
    scratch: Path,
    startup_seconds: float,
    stall_factor: int,
    poll_seconds: float,
    minimum_stall_seconds: float,
) -> None:
    entry = catalog.entries.get(config.entry)
    if entry is None:
        raise WorkerError(f"image catalog does not declare entry {config.entry!r}")
    config_path = scratch / "config.json"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    environment = dict(os.environ)
    environment["TRAINER_RUN_CONFIG"] = str(config_path)
    environment["TRAINER_SCRATCH"] = str(scratch)
    heartbeat = scratch / METRICS_FILENAME

    process = subprocess.Popen(list(entry.command), env=environment)
    watcher = _Heartbeat(heartbeat, startup_seconds, stall_factor, minimum_stall_seconds)
    while True:
        code = process.poll()
        if code is not None:
            break
        if watcher.stalled():
            _kill(process)
            raise WorkerError(
                f"run {config.run_id} stalled: no report for "
                f"{watcher.silence():.0f}s (limit {watcher.limit():.0f}s)"
            )
        time.sleep(poll_seconds)
    if code != 0:
        raise WorkerError(f"run {config.run_id} exited with exit code {code}")


class _Heartbeat:
    def __init__(
        self, path: Path, startup_seconds: float, stall_factor: int, minimum: float
    ) -> None:
        self._path = path
        self._startup = startup_seconds
        self._factor = stall_factor
        self._minimum = minimum
        self._started = time.monotonic()
        self._intervals: list[float] = []
        self._last_mtime: float | None = None
        self._last_seen = self._started

    def _poll(self) -> None:
        try:
            mtime = self._path.stat().st_mtime
        except FileNotFoundError:
            return
        if self._last_mtime is None or mtime > self._last_mtime:
            now = time.monotonic()
            if self._last_mtime is not None:
                self._intervals.append(now - self._last_seen)
            self._last_mtime = mtime
            self._last_seen = now

    def limit(self) -> float:
        if self._last_mtime is None:
            return self._startup
        if not self._intervals:
            return self._minimum
        return max(statistics.median(self._intervals) * self._factor, self._minimum)

    def silence(self) -> float:
        return time.monotonic() - (self._last_seen if self._last_mtime else self._started)

    def stalled(self) -> bool:
        self._poll()
        return self.silence() > self.limit()


def _kill(process: subprocess.Popen[bytes]) -> None:
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main(argv: Sequence[str] | None = None) -> int:
    manifest = os.environ["TRAINER_MANIFEST"]
    workspace = Path(os.environ.get("TRAINER_WORKSPACE", "/tmp/trainer"))
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        run_manifest(
            manifest,
            workspace,
            startup_seconds=float(os.environ.get("TRAINER_STARTUP_SECONDS", "900")),
            stall_factor=int(os.environ.get("TRAINER_STALL_FACTOR", "10")),
        )
    except Exception as error:  # noqa: BLE001 - the exit code is the only signal
        print(f"worker failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
