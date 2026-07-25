"""Fixtures for the acceptance trainer's tests.

The launcher builds the real sinks, so running it end to end uploads a Rerun
recording when the reporter closes. Every test in this package is pointed at a
moto server for that reason: none of them may reach real S3. A test that did
would pass on a machine holding AWS credentials and fail everywhere else, which
is exactly how three of these tests reached CI broken.
"""

import boto3
import pytest
from botocore.exceptions import ClientError

pytest_plugins = ["training_sdk.testing"]

RERUN_BUCKET = "bucket"


@pytest.fixture(autouse=True)
def local_s3(s3_endpoint: str) -> None:
    client = boto3.client("s3", endpoint_url=s3_endpoint)
    try:
        client.create_bucket(
            Bucket=RERUN_BUCKET,
            CreateBucketConfiguration={"LocationConstraint": "eu-north-1"},
        )
    except ClientError as error:
        if error.response["Error"]["Code"] != "BucketAlreadyOwnedByYou":
            raise
