from __future__ import annotations

import json
from io import BytesIO
from typing import Any

import pytest

from trainer_infra.batch import (
    REGION,
    BatchExecutionError,
    BatchRoundExecutor,
    batch_target,
)
from trainer_infra.scoring import ScoreSpec


class FakeS3:
    def __init__(self) -> None:
        self.data: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        self.data[(Bucket, Key)] = Body

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, BytesIO]:
        return {"Body": BytesIO(self.data[(Bucket, Key)])}


def split(uri: str) -> tuple[str, str]:
    without_scheme = uri.removeprefix("s3://")
    bucket, key = without_scheme.split("/", 1)
    return bucket, key


class FakeBatch:
    def __init__(self, s3: FakeS3, statuses: list[str] | None = None) -> None:
        self.s3 = s3
        self.statuses = statuses
        self.submitted: list[dict[str, Any]] = []
        self.terminated: list[str] = []

    def submit_job(self, **request: Any) -> dict[str, str]:
        self.submitted.append(request)
        job_id = f"job-{len(self.submitted)}"
        if self.statuses is None:
            environment = {
                item["name"]: item["value"] for item in request["containerOverrides"]["environment"]
            }
            manifest_uri = environment["TRAINER_MANIFEST"]
            manifest = json.loads(
                self.s3.get_object(Bucket=split(manifest_uri)[0], Key=split(manifest_uri)[1])[
                    "Body"
                ].read()
            )
            for config_uri in manifest["runs"]:
                bucket, key = split(config_uri)
                config = json.loads(self.s3.get_object(Bucket=bucket, Key=key)["Body"].read())
                root = config["artifacts"]["root"]
                trial = config["identity"]["trial"]
                metrics = json.dumps(
                    {"step": 10, "metrics": {"objective": float(trial + 1)}}
                ).encode()
                result = json.dumps(
                    {
                        "contract": 9,
                        "identity": config["identity"],
                        "success": True,
                        "artifacts": ["metrics.jsonl"],
                    }
                ).encode()
                root_bucket, root_key = split(root)
                self.s3.put_object(
                    Bucket=root_bucket, Key=f"{root_key}/metrics.jsonl", Body=metrics
                )
                self.s3.put_object(Bucket=root_bucket, Key=f"{root_key}/result.json", Body=result)
        return {"jobId": job_id}

    def describe_jobs(self, *, jobs: list[str]) -> dict[str, list[dict[str, str]]]:
        statuses = self.statuses or ["SUCCEEDED"] * len(jobs)
        return {
            "jobs": [
                {
                    "jobId": job_id,
                    "status": statuses[int(job_id.split("-")[1]) - 1],
                    "statusReason": "entry failed",
                    "container": {"logStreamName": f"stream/{job_id}"},
                }
                for job_id in jobs
            ]
        }

    def terminate_job(self, *, jobId: str, reason: str) -> None:
        self.terminated.append(jobId)


class FakeLogs:
    def get_log_events(self, **request: Any) -> dict[str, list[dict[str, str]]]:
        return {"events": [{"message": "RuntimeError: entry exploded"}]}


def configuration(trial: int) -> dict[str, Any]:
    return {
        "contract": 9,
        "identity": {
            "run_id": f"run-t{trial}",
            "experiment": "test",
            "launch_id": "launch",
            "trial": trial,
            "digest": "sha256:" + "a" * 64,
        },
        "entry": "e",
        "artifacts": {"root": f"s3://bucket/artifacts/t{trial}"},
        "algorithm": {},
        "runtime": {},
        "logging": {},
    }


def score() -> ScoreSpec:
    return ScoreSpec(
        metric="objective",
        window_steps=(0, 10),
        reduce="last",
        direction="maximize",
        non_finite="worst",
    )


def executor(s3: FakeS3, batch: FakeBatch) -> BatchRoundExecutor:
    return BatchRoundExecutor(
        s3=s3,
        batch=batch,
        logs=FakeLogs(),
        exchange="s3://bucket/control/launch",
        job_name="experiment-launch",
        job_queue="dev-cpu-c7al-queue",
        job_definition="trainer-c7al-digest",
        timeout_seconds=5400,
        parallel_jobs=2,
        poll_seconds=0,
    )


def test_batch_executor_packs_submits_collects_and_scores_a_round() -> None:
    s3 = FakeS3()
    batch = FakeBatch(s3)
    configurations = tuple(configuration(trial) for trial in range(4))

    results = executor(s3, batch)(configurations, score())

    assert results == tuple({"trial": trial, "value": float(trial + 1)} for trial in range(4))
    assert len(batch.submitted) == 2
    for request in batch.submitted:
        assert request["jobQueue"] == "dev-cpu-c7al-queue"
        assert request["jobDefinition"] == "trainer-c7al-digest"
        assert request["timeout"] == {"attemptDurationSeconds": 5400}
        environment = {
            item["name"]: item["value"] for item in request["containerOverrides"]["environment"]
        }
        assert environment["AWS_REGION"] == REGION
        assert environment["AWS_DEFAULT_REGION"] == REGION
        manifest_uri = environment["TRAINER_MANIFEST"]
        bucket, key = split(manifest_uri)
        assert len(json.loads(s3.data[(bucket, key)])["runs"]) == 2


def test_batch_failure_terminates_nonterminal_siblings_and_exposes_logs() -> None:
    s3 = FakeS3()
    batch = FakeBatch(s3, statuses=["FAILED", "RUNNING"])

    with pytest.raises(BatchExecutionError, match="entry exploded"):
        executor(s3, batch)(tuple(configuration(trial) for trial in range(2)), score())

    assert batch.terminated == ["job-2"]


def test_batch_target_routes_instance_tier_and_digest() -> None:
    target = batch_target("c7a.large", "dev", "sha256:" + "b" * 64)

    assert target.queue == "dev-cpu-c7al-queue"
    assert target.job_definition == "trainer-c7al-" + "b" * 64
