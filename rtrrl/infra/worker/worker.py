from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
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
        bucket, key = self._location(uri)
        source_fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            if not stat.S_ISREG(os.fstat(source_fd).st_mode):
                raise ValueError(f"upload source is not a regular file: {path}")
            with os.fdopen(os.dup(source_fd), "rb") as source:
                digest = hashlib.sha256()
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                hexdigest = digest.hexdigest()
                source.seek(0)
                self._client.upload_fileobj(
                    source,
                    bucket,
                    key,
                    ExtraArgs={"Metadata": {"sha256": hexdigest}},
                )
        finally:
            os.close(source_fd)
        return hexdigest


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
    artifact_root_fd: int,
    staging_root: Path,
    uploaded: list[str],
) -> None:
    def stage_tree(directory_fd: int, relative_parts: tuple[str, ...]) -> None:
        for name in sorted(os.listdir(directory_fd)):
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            relative = (*relative_parts, name)
            if stat.S_ISDIR(info.st_mode):
                child_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
                try:
                    stage_tree(child_fd, relative)
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(
                    f"artifact entry is not a regular file: {'/'.join(relative)}"
                )
            source_fd = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory_fd,
            )
            try:
                if not stat.S_ISREG(os.fstat(source_fd).st_mode):
                    raise ValueError(
                        f"artifact entry changed type: {'/'.join(relative)}"
                    )
                staged = staging_root.joinpath(*relative)
                staged.parent.mkdir(parents=True, exist_ok=True)
                with os.fdopen(os.dup(source_fd), "rb") as source, staged.open("xb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
            finally:
                os.close(source_fd)
            uri = f"{run_root_uri}{'/'.join(relative)}"
            store.put_file(uri, staged)
            uploaded.append(uri)

    for name in sorted(("aim-buffer", "rerun", "checkpoints")):
        try:
            directory_fd = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=artifact_root_fd,
            )
        except FileNotFoundError:
            continue
        try:
            stage_tree(directory_fd, (name,))
        finally:
            os.close(directory_fd)


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

        with (
            tempfile.TemporaryDirectory(prefix="trainer-run-") as temporary,
            tempfile.TemporaryDirectory(prefix="trainer-stage-") as staging,
        ):
            temporary_root = Path(temporary)
            input_root = temporary_root / "input"
            artifact_root = temporary_root / "artifacts"
            input_root.mkdir()
            artifact_root.mkdir()
            artifact_root_fd = os.open(
                artifact_root,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
            config_path = input_root / "config.yaml"
            context_path = input_root / "run-context.json"
            config_path.write_bytes(config)
            runtime_context = thaw_json(run.run_context)
            runtime_context["artifact_directory"] = str(artifact_root)
            context_path.write_text(canonical_json(runtime_context))
            environment = dict(os.environ)
            environment["TRAINER_RUN_CONTEXT_PATH"] = str(context_path)
            try:
                completed = run_command(
                    _runtime_argv(run, config_path),
                    env=environment,
                    shell=False,
                    check=False,
                )
                artifacts: list[str] = []
                try:
                    _safe_artifacts(
                        store,
                        run_root_uri,
                        artifact_root_fd,
                        Path(staging),
                        artifacts,
                    )
                except Exception as artifact_error:
                    marker = CompletionMarker(
                        run_id=run.run_id,
                        attempt=0,
                        exit_code=completed.returncode,
                        started_at=started_at,
                        finished_at=now(),
                        artifacts=tuple(artifacts),
                        error=(
                            f"artifact upload failed: {type(artifact_error).__name__}: "
                            f"{artifact_error}"
                        ),
                    )
                    try:
                        store.put_json(
                            f"{run_root_uri}status/attempt-0.json",
                            marker.model_dump(mode="json"),
                        )
                    except Exception as marker_error:
                        artifact_error.add_note(
                            "completion marker write also failed: "
                            f"{type(marker_error).__name__}: {marker_error}"
                        )
                        raise artifact_error from marker_error
                    raise
            finally:
                os.close(artifact_root_fd)
            marker = CompletionMarker(
                run_id=run.run_id,
                attempt=0,
                exit_code=completed.returncode,
                started_at=started_at,
                finished_at=now(),
                artifacts=tuple(artifacts),
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
    namespace, _ = ExperimentS3Namespace.from_bundle_uri(args.bundle_s3_uri)
    store = BotoS3ObjectStore(boto3.client("s3"), namespace)
    return execute_bundle(args.bundle_s3_uri, store)


if __name__ == "__main__":
    raise SystemExit(main())
