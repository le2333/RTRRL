from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

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


def get_bytes(uri: str) -> bytes:
    bucket, key = split_uri(uri)
    return client().get_object(Bucket=bucket, Key=key)["Body"].read()


def put_bytes(uri: str, payload: bytes) -> None:
    bucket, key = split_uri(uri)
    client().put_object(Bucket=bucket, Key=key, Body=payload)


def put_file(uri: str, path: Path) -> None:
    bucket, key = split_uri(uri)
    client().upload_file(str(path), bucket, key)


def exists(uri: str) -> bool:
    bucket, key = split_uri(uri)
    try:
        client().head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if error.response["Error"]["Code"] in {"404", "NoSuchKey"}:
            return False
        raise
    return True
