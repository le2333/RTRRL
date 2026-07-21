from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any

from training_sdk.execution import canonical_json
from training_sdk.storage import ExperimentS3Namespace


class S3ObjectStore:
    def __init__(self, client: Any, experiment_prefix: str) -> None:
        self._client = client
        self._namespace = ExperimentS3Namespace.from_prefix(experiment_prefix)

    def _location(self, uri: str) -> tuple[str, str]:
        parsed = self._namespace.require_uri(uri)
        return parsed.bucket, parsed.key

    def put_bytes(self, uri: str, data: bytes) -> str:
        bucket, key = self._location(uri)
        digest = hashlib.sha256(data).hexdigest()
        self._client.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            Metadata={"sha256": digest},
        )
        return digest

    def get_bytes(self, uri: str, *, expected_sha256: str | None = None) -> bytes:
        bucket, key = self._location(uri)
        response = self._client.get_object(Bucket=bucket, Key=key)
        data = response["Body"].read()
        metadata = response.get("Metadata")
        stored_sha256 = metadata.get("sha256") if isinstance(metadata, dict) else None
        if not isinstance(stored_sha256, str):
            raise ValueError(f"SHA-256 metadata is missing for {uri!r}")
        actual = hashlib.sha256(data).hexdigest()
        if actual != stored_sha256 or (
            expected_sha256 is not None and actual != expected_sha256
        ):
            expected = expected_sha256 or stored_sha256
            raise ValueError(
                f"SHA-256 mismatch for {uri!r}: expected {expected}, got {actual}"
            )
        return data

    def put_json(self, uri: str, value: Any) -> str:
        return self.put_bytes(uri, canonical_json(value).encode("utf-8"))

    def get_json(self, uri: str, *, expected_sha256: str | None = None) -> Any:
        data = self.get_bytes(uri, expected_sha256=expected_sha256)
        try:
            value = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(f"object {uri!r} is not valid JSON") from error
        if data != canonical_json(value).encode("utf-8"):
            raise ValueError(f"object {uri!r} does not use canonical JSON")
        return value

    def put_file(self, uri: str, path: Path) -> str:
        bucket, key = self._location(uri)
        source_fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            if not stat.S_ISREG(os.fstat(source_fd).st_mode):
                raise ValueError(f"upload source is not a regular file: {path}")
            digest = hashlib.sha256()
            with (
                os.fdopen(os.dup(source_fd), "rb") as source,
                tempfile.TemporaryFile("w+b") as staged,
            ):
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    staged.write(chunk)
                hexdigest = digest.hexdigest()
                staged.flush()
                staged.seek(0)
                self._client.upload_fileobj(
                    staged,
                    bucket,
                    key,
                    ExtraArgs={"Metadata": {"sha256": hexdigest}},
                )
        finally:
            os.close(source_fd)
        return hexdigest
