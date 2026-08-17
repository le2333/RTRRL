"""The AWS surface the Batch executor touches, small enough to read.

Shared rather than copied because the tests that matter for memory ask what
the executor did to a stream, not only what it returned: a fake that answers
with bytes cannot be asked that, and a second copy of it would drift.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import IO, Any


class StreamedBody:
    """What ``get_object`` hands back, plus a record of how it was read.

    ``amounts`` is the point: a body that was read whole shows one ``None``,
    and a body that was streamed shows a bounded read per chunk.
    """

    def __init__(self, source: IO[bytes]) -> None:
        self._source = source
        self.amounts: list[int | None] = []

    def read(self, amt: int | None = None) -> bytes:
        self.amounts.append(amt)
        return self._source.read() if amt is None else self._source.read(amt)

    def close(self) -> None:
        self._source.close()


class FakeS3:
    def __init__(self) -> None:
        self.data: dict[tuple[str, str], bytes] = {}
        self.files: dict[tuple[str, str], Path] = {}
        self.bodies: dict[tuple[str, str], StreamedBody] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        self.data[(Bucket, Key)] = Body

    def put_file(self, *, Bucket: str, Key: str, Source: Path) -> None:
        """Serve an object from disk, for the sizes that do not fit in a dict."""

        self.files[(Bucket, Key)] = Source

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, StreamedBody]:
        source = (
            self.files[(Bucket, Key)].open("rb")
            if (Bucket, Key) in self.files
            else BytesIO(self.data[(Bucket, Key)])
        )
        body = StreamedBody(source)
        self.bodies[(Bucket, Key)] = body
        return {"Body": body}


def split(uri: str) -> tuple[str, str]:
    without_scheme = uri.removeprefix("s3://")
    bucket, key = without_scheme.split("/", 1)
    return bucket, key


def publish_one_row(s3: FakeS3, config: dict[str, Any]) -> None:
    """The smallest complete artifact set: one row, one metric, one trial."""

    trial = config["identity"]["trial"]
    metrics = json.dumps({"step": 10, "metrics": {"objective": float(trial + 1)}}).encode()
    root_bucket, root_key = split(config["artifacts"]["root"])
    s3.put_object(Bucket=root_bucket, Key=f"{root_key}/metrics.jsonl", Body=metrics)
    publish_result(s3, config)


def publish_result(s3: FakeS3, config: dict[str, Any]) -> None:
    result = json.dumps(
        {
            "contract": 10,
            "identity": config["identity"],
            "success": True,
            "artifacts": ["metrics.jsonl"],
        }
    ).encode()
    root_bucket, root_key = split(config["artifacts"]["root"])
    s3.put_object(Bucket=root_bucket, Key=f"{root_key}/result.json", Body=result)


class FakeBatch:
    """Jobs that finish the moment they are submitted, or not at all.

    ``publish`` is what a worker would have uploaded. Passing another one is
    how a test says what the metrics for this round look like without saying
    anything about how they got there.
    """

    def __init__(
        self,
        s3: FakeS3,
        statuses: list[str] | None = None,
        publish: Callable[[FakeS3, dict[str, Any]], None] = publish_one_row,
    ) -> None:
        self.s3 = s3
        self.statuses = statuses
        self.publish = publish
        self.submitted: list[dict[str, Any]] = []
        self.terminated: list[str] = []

    def submit_job(self, **request: Any) -> dict[str, str]:
        self.submitted.append(request)
        job_id = f"job-{len(self.submitted)}"
        if self.statuses is None:
            environment = {
                item["name"]: item["value"] for item in request["containerOverrides"]["environment"]
            }
            manifest_bucket, manifest_key = split(environment["TRAINER_MANIFEST"])
            manifest = json.loads(
                self.s3.get_object(Bucket=manifest_bucket, Key=manifest_key)["Body"].read()
            )
            for config_uri in manifest["runs"]:
                bucket, key = split(config_uri)
                config = json.loads(self.s3.get_object(Bucket=bucket, Key=key)["Body"].read())
                self.publish(self.s3, config)
        return {"jobId": job_id}

    def describe_jobs(self, *, jobs: list[str]) -> dict[str, list[dict[str, str]]]:
        return {
            "jobs": [
                {
                    "jobId": job_id,
                    "status": self._status(job_id),
                    "statusReason": "entry failed",
                    "container": {"logStreamName": f"stream/{job_id}"},
                }
                for job_id in jobs
            ]
        }

    def _status(self, job_id: str) -> str:
        """Job ids run on across rounds; declared statuses are read by that number."""

        if self.statuses is None:
            return "SUCCEEDED"
        return self.statuses[int(job_id.split("-")[1]) - 1]

    def terminate_job(self, *, jobId: str, reason: str) -> None:
        self.terminated.append(jobId)


class FakeLogs:
    def get_log_events(self, **request: Any) -> dict[str, list[dict[str, str]]]:
        return {"events": [{"message": "RuntimeError: entry exploded"}]}
