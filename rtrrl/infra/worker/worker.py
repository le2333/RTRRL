from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from typing import Any, Protocol

import boto3

from training_sdk.execution import (
    CompletionMarker,
    JobBundle,
    RunBundle,
    canonical_json,
    thaw_json,
)
from training_sdk.storage import ExperimentS3Namespace


class ObjectStore(Protocol):
    def get_bytes(self, uri: str, *, expected_sha256: str | None = None) -> bytes: ...

    def put_json(self, uri: str, value: Any) -> str: ...

    def put_file(self, uri: str, path: Path) -> str: ...


class BotoS3ObjectStore:
    def __init__(self, client: Any, namespace: ExperimentS3Namespace) -> None:
        self._client = client
        self._namespace = namespace

    def _location(self, uri: str) -> tuple[str, str]:
        parsed = self._namespace.require_uri(uri)
        return parsed.bucket, parsed.key

    def get_bytes(self, uri: str, *, expected_sha256: str | None = None) -> bytes:
        bucket, key = self._location(uri)
        response = self._client.get_object(Bucket=bucket, Key=key)
        data = response["Body"].read()
        metadata = response.get("Metadata")
        stored = metadata.get("sha256") if isinstance(metadata, dict) else None
        actual = hashlib.sha256(data).hexdigest()
        if not isinstance(stored, str):
            raise ValueError(f"SHA-256 metadata is missing for {uri!r}")
        if actual != stored or (expected_sha256 is not None and actual != expected_sha256):
            raise ValueError(f"SHA-256 mismatch for {uri!r}")
        return data

    def _put_bytes(self, uri: str, data: bytes) -> str:
        bucket, key = self._location(uri)
        digest = hashlib.sha256(data).hexdigest()
        self._client.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            Metadata={"sha256": digest},
        )
        return digest

    def put_json(self, uri: str, value: Any) -> str:
        return self._put_bytes(uri, canonical_json(value).encode())

    def put_file(self, uri: str, path: Path) -> str:
        return self._put_bytes(uri, path.read_bytes())


def _input_prefix_uri(
    namespace: ExperimentS3Namespace,
    run: RunBundle,
) -> str:
    parsed = namespace.require_key(run.artifact_prefix)
    parts = parsed.key.split("/")
    if (
        len(parts) != 8
        or parts[0] != "experiments"
        or parts[1] != namespace.experiment_id
        or parts[2] != "groups"
        or parts[4] != "runs"
        or parts[6] != "input"
        or parts[7] != ""
    ):
        raise ValueError("run artifact_prefix must be its exact experiment input prefix")
    return f"s3://{parsed.bucket}/{parsed.key}"


def _safe_artifacts(
    store: ObjectStore,
    run_root_uri: str,
    artifact_root: Path,
) -> tuple[str, ...]:
    root = artifact_root.resolve(strict=True)
    candidates: list[Path] = []
    for name in ("aim-buffer", "rerun", "checkpoints"):
        directory = artifact_root / name
        if not directory.exists() and not directory.is_symlink():
            continue
        mode = directory.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise ValueError(f"artifact path is not a safe directory: {directory}")
        for current, directories, files in os.walk(directory, followlinks=False):
            current_path = Path(current)
            for entry_name in directories:
                entry = current_path / entry_name
                entry_mode = entry.lstat().st_mode
                if stat.S_ISLNK(entry_mode) or not stat.S_ISDIR(entry_mode):
                    raise ValueError(f"artifact entry is not a safe directory: {entry}")
                entry.resolve(strict=True).relative_to(root)
            for entry_name in files:
                entry = current_path / entry_name
                entry_mode = entry.lstat().st_mode
                if stat.S_ISLNK(entry_mode) or not stat.S_ISREG(entry_mode):
                    raise ValueError(f"artifact entry is not a regular file: {entry}")
                entry.resolve(strict=True).relative_to(root)
                candidates.append(entry)
    uploaded: list[str] = []
    for path in sorted(candidates, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        uri = f"{run_root_uri}{relative}"
        store.put_file(uri, path)
        uploaded.append(uri)
    return tuple(uploaded)


def _runtime_argv(run: RunBundle, config_path: Path) -> list[str]:
    return [str(config_path) if item == "{config_path}" else item for item in run.argv]


def execute_bundle(
    bundle_s3_uri: str,
    store: ObjectStore,
    *,
    run_command: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> int:
    namespace, bundle_id = ExperimentS3Namespace.from_bundle_uri(bundle_s3_uri)
    bundle = JobBundle.from_json(store.get_bytes(bundle_s3_uri).decode())
    if bundle.job_id != bundle_id:
        raise ValueError("bundle job_id does not match its S3 URI")

    for run in bundle.runs:
        input_prefix = _input_prefix_uri(namespace, run)
        config = store.get_bytes(
            f"{input_prefix}config.yaml",
            expected_sha256=run.config_sha256,
        )
        stored_context = store.get_bytes(
            f"{input_prefix}run-context.json",
            expected_sha256=run.run_context_sha256,
        )
        if stored_context != canonical_json(run.run_context).encode():
            raise ValueError("run-context input does not match canonical bundle context")
        run_root_uri = input_prefix.removesuffix("input/")
        started_at = now()

        with tempfile.TemporaryDirectory(prefix="trainer-run-") as temporary:
            temporary_root = Path(temporary)
            input_root = temporary_root / "input"
            artifact_root = temporary_root / "artifacts"
            input_root.mkdir()
            artifact_root.mkdir()
            config_path = input_root / "config.yaml"
            context_path = input_root / "run-context.json"
            config_path.write_bytes(config)
            runtime_context = thaw_json(run.run_context)
            runtime_context["artifact_directory"] = str(artifact_root)
            context_path.write_text(canonical_json(runtime_context))
            environment = dict(os.environ)
            environment["TRAINER_RUN_CONTEXT_PATH"] = str(context_path)
            completed = run_command(
                _runtime_argv(run, config_path),
                env=environment,
                shell=False,
                check=False,
            )

            artifacts: tuple[str, ...] = ()
            artifact_error: Exception | None = None
            try:
                artifacts = _safe_artifacts(store, run_root_uri, artifact_root)
            except Exception as error:
                artifact_error = error
            marker = CompletionMarker(
                run_id=run.run_id,
                attempt=0,
                exit_code=completed.returncode,
                started_at=started_at,
                finished_at=now(),
                artifacts=artifacts,
                error=(
                    f"artifact upload failed: {type(artifact_error).__name__}: "
                    f"{artifact_error}"
                    if artifact_error is not None
                    else None
                ),
            )
            store.put_json(
                f"{run_root_uri}status/attempt-0.json",
                marker.model_dump(mode="json"),
            )
            if artifact_error is not None:
                return completed.returncode if completed.returncode != 0 else 1
            if completed.returncode != 0:
                return completed.returncode
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute one immutable trainer job bundle.")
    parser.add_argument("--bundle-s3-uri", required=True)
    args = parser.parse_args(argv)
    namespace, _ = ExperimentS3Namespace.from_bundle_uri(args.bundle_s3_uri)
    store = BotoS3ObjectStore(boto3.client("s3"), namespace)
    return execute_bundle(args.bundle_s3_uri, store)


if __name__ == "__main__":
    raise SystemExit(main())
