from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from trainer_infra.identities import canonical_json


class S3ObjectStore:
    def __init__(self, client: Any, experiment_prefix: str) -> None:
        self._client = client
        bucket, key = self._parse_uri(experiment_prefix, require_object=False)
        if not key.startswith("experiments/") or not key.endswith("/"):
            raise ValueError("experiment prefix must be experiments/<experiment-id>/")
        parts = PurePosixPath(key).parts
        if len(parts) != 2 or not parts[1]:
            raise ValueError("experiment prefix must be experiments/<experiment-id>/")
        self._bucket = bucket
        self._prefix = key

    @staticmethod
    def _parse_uri(uri: str, *, require_object: bool = True) -> tuple[str, str]:
        parsed = urlsplit(uri)
        if (
            parsed.scheme != "s3"
            or not parsed.netloc
            or parsed.query
            or parsed.fragment
            or not parsed.path.startswith("/")
        ):
            raise ValueError(f"invalid S3 URI: {uri!r}")
        key = parsed.path[1:]
        parts = key.split("/")
        normalized_parts = [part for part in parts if part]
        if (
            not key
            or "." in parts
            or ".." in parts
            or (require_object and (not normalized_parts or "" in parts))
        ):
            raise ValueError(f"invalid S3 URI: {uri!r}")
        return parsed.netloc, key

    def _location(self, uri: str) -> tuple[str, str]:
        bucket, key = self._parse_uri(uri)
        if bucket != self._bucket or not key.startswith(self._prefix):
            raise ValueError(f"S3 URI is outside configured experiment prefix: {uri!r}")
        return bucket, key

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
        return json.loads(self.get_bytes(uri, expected_sha256=expected_sha256))

    def put_file(self, uri: str, path: Path) -> str:
        bucket, key = self._location(uri)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self._client.upload_file(
            str(path),
            bucket,
            key,
            ExtraArgs={"Metadata": {"sha256": digest}},
        )
        return digest
