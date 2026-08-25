from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import boto3
from botocore.exceptions import ClientError


@lru_cache(maxsize=1)
def client():
    return boto3.client(
        "s3", endpoint_url=os.environ.get("TRAINER_S3_ENDPOINT") or None
    )


def split_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"not an s3 uri: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def file_path(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return None
    if parsed.netloc:
        raise ValueError(f"file uri must be local: {uri!r}")
    return Path(url2pathname(parsed.path))


def get_bytes(uri: str) -> bytes:
    path = file_path(uri)
    if path is not None:
        return path.read_bytes()
    bucket, key = split_uri(uri)
    return client().get_object(Bucket=bucket, Key=key)["Body"].read()


def put_bytes(uri: str, payload: bytes) -> None:
    path = file_path(uri)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return
    bucket, key = split_uri(uri)
    client().put_object(Bucket=bucket, Key=key, Body=payload)


def put_file(uri: str, path: Path) -> None:
    destination = file_path(uri)
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        return
    bucket, key = split_uri(uri)
    client().upload_file(str(path), bucket, key)


def exists(uri: str) -> bool:
    path = file_path(uri)
    if path is not None:
        return path.exists()
    bucket, key = split_uri(uri)
    try:
        client().head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if error.response["Error"]["Code"] in {"404", "NoSuchKey"}:
            return False
        raise
    return True


def delete(uri: str) -> None:
    """Remove an object, if it is still there.

    Absence is the desired state rather than an error: the caller is dropping
    something it no longer needs, and two processes arriving at that
    conclusion is not a failure of either.
    """

    path = file_path(uri)
    if path is not None:
        path.unlink(missing_ok=True)
        return
    bucket, key = split_uri(uri)
    client().delete_object(Bucket=bucket, Key=key)
