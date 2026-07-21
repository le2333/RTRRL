from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from trainer_infra.adapters.s3 import S3ObjectStore


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.metadata: dict[tuple[str, str], dict[str, str]] = {}
        self.put_calls: list[dict[str, Any]] = []

    def put_object(self, **kwargs: Any) -> None:
        self.put_calls.append(kwargs)
        self.objects[(kwargs["Bucket"], kwargs["Key"])] = kwargs["Body"]
        self.metadata[(kwargs["Bucket"], kwargs["Key"])] = kwargs.get("Metadata", {})

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        data = self.objects[(kwargs["Bucket"], kwargs["Key"])]
        return {
            "Body": FakeBody(data),
            "Metadata": self.metadata[(kwargs["Bucket"], kwargs["Key"])],
        }

    def upload_file(
        self, filename: str, bucket: str, key: str, ExtraArgs: dict[str, Any]
    ) -> None:
        self.objects[(bucket, key)] = Path(filename).read_bytes()
        self.metadata[(bucket, key)] = ExtraArgs["Metadata"]


class FakeBody:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


def test_bytes_and_canonical_json_round_trip_with_hash_verification(tmp_path: Path) -> None:
    client = FakeS3()
    store = S3ObjectStore(client, "s3://bucket/experiments/exp-1/")
    bytes_uri = "s3://bucket/experiments/exp-1/input/data.bin"
    json_uri = "s3://bucket/experiments/exp-1/jobs/job-1/bundle.json"

    digest = store.put_bytes(bytes_uri, b"payload")
    assert digest == hashlib.sha256(b"payload").hexdigest()
    assert client.metadata[("bucket", "experiments/exp-1/input/data.bin")] == {
        "sha256": digest
    }
    assert store.get_bytes(bytes_uri, expected_sha256=digest) == b"payload"

    json_digest = store.put_json(json_uri, {"z": 1, "a": "é"})
    expected = b'{"a":"\xc3\xa9","z":1}'
    assert client.objects[("bucket", "experiments/exp-1/jobs/job-1/bundle.json")] == expected
    assert json_digest == hashlib.sha256(expected).hexdigest()
    assert store.get_json(json_uri, expected_sha256=json_digest) == {"a": "é", "z": 1}

    source = tmp_path / "artifact.rrd"
    source.write_bytes(b"artifact")
    artifact_uri = "s3://bucket/experiments/exp-1/groups/g/runs/r/rerun/eval.rrd"
    assert store.put_file(artifact_uri, source) == hashlib.sha256(b"artifact").hexdigest()
    assert client.objects[("bucket", artifact_uri.removeprefix("s3://bucket/"))] == b"artifact"


@pytest.mark.parametrize(
    "uri",
    [
        "https://bucket/experiments/exp-1/object",
        "s3:///experiments/exp-1/object",
        "s3://other/experiments/exp-1/object",
        "s3://bucket/experiments/exp-10/object",
        "s3://bucket/experiments/exp-1",
        "s3://bucket/experiments/exp-1/../other/object",
        "s3://bucket/experiments/exp-1//object",
        "s3://bucket/experiments/exp-1/object?versionId=x",
    ],
)
def test_every_operation_requires_exact_configured_experiment_prefix(
    uri: str, tmp_path: Path
) -> None:
    store = S3ObjectStore(FakeS3(), "s3://bucket/experiments/exp-1/")
    source = tmp_path / "artifact"
    source.write_bytes(b"x")

    for operation in (
        lambda: store.put_bytes(uri, b"x"),
        lambda: store.get_bytes(uri),
        lambda: store.put_json(uri, {}),
        lambda: store.get_json(uri),
        lambda: store.put_file(uri, source),
    ):
        with pytest.raises(ValueError, match="S3 URI|prefix"):
            operation()


def test_download_rejects_tampering_and_aws_errors_propagate_unchanged() -> None:
    client = FakeS3()
    store = S3ObjectStore(client, "s3://bucket/experiments/exp-1/")
    uri = "s3://bucket/experiments/exp-1/object"
    store.put_bytes(uri, b"changed")

    with pytest.raises(ValueError, match="SHA-256"):
        store.get_bytes(uri, expected_sha256="0" * 64)

    client.objects[("bucket", "experiments/exp-1/object")] = b"tampered"
    with pytest.raises(ValueError, match="SHA-256"):
        store.get_bytes(uri)

    error = RuntimeError("raw AWS error")

    def fail(**_kwargs: Any) -> Any:
        raise error

    client.get_object = fail  # type: ignore[method-assign]
    with pytest.raises(RuntimeError) as caught:
        store.get_bytes(uri)
    assert caught.value is error
