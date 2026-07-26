"""Pytest fixtures shared by both packages; loaded with pytest_plugins."""

import os
from collections.abc import Iterator

import boto3
import pytest
from botocore.exceptions import ClientError
from moto.server import ThreadedMotoServer


@pytest.fixture(scope="session")
def s3_endpoint() -> Iterator[str]:
    server = ThreadedMotoServer(port=0)
    server.start()
    host, port = server.get_host_and_port()
    endpoint = f"http://{host}:{port}"
    previous = os.environ.get("TRAINER_S3_ENDPOINT")
    os.environ["TRAINER_S3_ENDPOINT"] = endpoint
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
    os.environ.setdefault("AWS_DEFAULT_REGION", "eu-north-1")
    yield endpoint
    if previous is None:
        del os.environ["TRAINER_S3_ENDPOINT"]
    else:
        os.environ["TRAINER_S3_ENDPOINT"] = previous
    server.stop()


@pytest.fixture
def s3_base(s3_endpoint: str) -> str:
    bucket = "trainer-test"
    s3 = boto3.client("s3", endpoint_url=s3_endpoint)
    try:
        s3.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": "eu-north-1"},
        )
    except ClientError as error:
        if error.response["Error"]["Code"] != "BucketAlreadyOwnedByYou":
            raise
    return f"s3://{bucket}"
