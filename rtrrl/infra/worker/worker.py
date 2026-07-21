from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any
from urllib.parse import urlsplit

import boto3

from trainer_infra.adapters.protocols import ObjectStore
from trainer_infra.adapters.s3 import S3ObjectStore
from trainer_infra.execution import CompletionMarker, JobBundle, RunBundle
from trainer_infra.identities import canonical_json


def _bundle_location(bundle_uri: str) -> tuple[str, str]:
    parsed = urlsplit(bundle_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.startswith("/"):
        raise ValueError("bundle URI must be an s3://bucket/key URI")
    key = parsed.path[1:]
    marker = "/jobs/"
    if marker not in key:
        raise ValueError("bundle URI must be below an experiment jobs prefix")
    experiment_key = key.split(marker, 1)[0] + "/"
    return parsed.netloc, experiment_key


def _input_prefix_uri(bundle_uri: str, run: RunBundle) -> str:
    bucket, experiment_key = _bundle_location(bundle_uri)
    prefix = run.artifact_prefix
    if prefix.startswith("s3://"):
        parsed = urlsplit(prefix)
        if parsed.scheme != "s3" or parsed.netloc != bucket:
            raise ValueError("run input prefix must use the bundle bucket")
        key = parsed.path.removeprefix("/")
    else:
        key = prefix
    if not key.startswith(experiment_key) or not key.endswith("/input/"):
        raise ValueError("run artifact_prefix must be its exact experiment input prefix")
    return f"s3://{bucket}/{key}"


def _config_path(argv: Sequence[str]) -> Path:
    for option in ("--config", "--config_path"):
        if option in argv:
            index = argv.index(option)
            if index + 1 >= len(argv):
                break
            return Path(argv[index + 1])
    raise ValueError("run argv must contain --config or --config_path")


def _upload_artifacts(
    store: ObjectStore,
    run_root_uri: str,
    artifact_directory: Path,
) -> tuple[str, ...]:
    uploaded: list[str] = []
    paths: list[Path] = []
    for directory_name in ("aim-buffer", "rerun", "checkpoints"):
        directory = artifact_directory / directory_name
        if not directory.exists():
            continue
        if not directory.is_dir():
            raise ValueError(f"artifact path is not a directory: {directory}")
        paths.extend(item for item in directory.rglob("*") if item.is_file())
    for path in sorted(paths, key=lambda item: item.relative_to(artifact_directory).as_posix()):
        relative = path.relative_to(artifact_directory).as_posix()
        uri = f"{run_root_uri}{relative}"
        store.put_file(uri, path)
        uploaded.append(uri)
    return tuple(uploaded)


def execute_bundle(
    bundle_s3_uri: str,
    store: ObjectStore,
    *,
    run_command: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> int:
    raw_bundle = store.get_bytes(bundle_s3_uri)
    bundle = JobBundle.from_json(raw_bundle.decode("utf-8"))
    if raw_bundle != bundle.to_json().encode("utf-8"):
        raise ValueError("job bundle must use canonical JSON serialization")

    for run in bundle.runs:
        input_prefix = _input_prefix_uri(bundle_s3_uri, run)
        config = store.get_bytes(
            f"{input_prefix}config.yaml",
            expected_sha256=run.config_sha256,
        )
        context = store.get_bytes(
            f"{input_prefix}run-context.json",
            expected_sha256=run.run_context_sha256,
        )
        expected_context = canonical_json(run.run_context).encode("utf-8")
        if context != expected_context:
            raise ValueError("run-context input does not match canonical bundle context")

        config_path = _config_path(run.argv)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_bytes(config)
        artifact_directory = Path(str(run.run_context["artifact_directory"]))
        artifact_directory.mkdir(parents=True, exist_ok=True)
        run_root_uri = input_prefix.removesuffix("input/")

        started_at = now()
        with tempfile.TemporaryDirectory(prefix="trainer-run-context-") as temporary:
            context_path = Path(temporary) / "run-context.json"
            context_path.write_bytes(context)
            environment = dict(os.environ)
            environment["TRAINER_RUN_CONTEXT_PATH"] = str(context_path)
            completed = run_command(
                list(run.argv),
                env=environment,
                shell=False,
                check=False,
            )
        finished_at = now()
        artifacts = _upload_artifacts(
            store,
            run_root_uri,
            artifact_directory,
        )
        marker = CompletionMarker(
            run_id=run.run_id,
            attempt=0,
            exit_code=completed.returncode,
            started_at=started_at,
            finished_at=finished_at,
            artifacts=artifacts,
        )
        store.put_json(
            f"{run_root_uri}status/attempt-0.json",
            marker.model_dump(mode="json"),
        )
        if completed.returncode != 0:
            return completed.returncode
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute one immutable trainer job bundle.")
    parser.add_argument("--bundle-s3-uri", required=True)
    args = parser.parse_args(argv)
    bucket, experiment_key = _bundle_location(args.bundle_s3_uri)
    store = S3ObjectStore(
        boto3.client("s3"),
        f"s3://{bucket}/{experiment_key}",
    )
    return execute_bundle(args.bundle_s3_uri, store)


if __name__ == "__main__":
    raise SystemExit(main())
