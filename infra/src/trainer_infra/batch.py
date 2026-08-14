"""AWS Batch round execution for version-8 Worker manifests."""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from trainer_infra.scoring import ScoreSpec, compute_score

REGION = "eu-north-1"
JOB_LOG_GROUP = "/trainer/jobs"
TERMINAL = {"SUCCEEDED", "FAILED"}

ACCOUNT_ID = "007122174918"
JOB_ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/rtrrl-batch-job-role"
EXECUTION_ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/rtrrl-batch-execution-role"


@dataclass(frozen=True)
class BatchProfile:
    profile: str
    run_queue: str
    dev_queue: str
    vcpus: int
    memory_mib: int
    gpus: int = 0


PROFILES = {
    "c7a.medium": BatchProfile(
        "c7am", "run-cpu-c7am-queue", "dev-cpu-c7am-queue", 1, 1600
    ),
    "c7a.large": BatchProfile(
        "c7al", "run-cpu-c7al-queue", "dev-cpu-c7al-queue", 2, 3200
    ),
    "c7a.xlarge": BatchProfile(
        "c7ax", "run-cpu-c7ax-queue", "dev-cpu-c7ax-queue", 4, 7168
    ),
    "g6.xlarge": BatchProfile(
        "g6x", "run-gpu-queue", "dev-gpu-queue", 4, 12000, 1
    ),
}


class BatchExecutionError(RuntimeError):
    """A Batch job failed before its complete round could be scored."""


@dataclass(frozen=True)
class BatchTarget:
    queue: str
    job_definition: str


def batch_target(instance_type: str, tier: str, digest: str) -> BatchTarget:
    profile = PROFILES[instance_type]
    queue = {"run": profile.run_queue, "dev": profile.dev_queue}[tier]
    return BatchTarget(
        queue=queue,
        job_definition=f"trainer-{profile.profile}-{digest.removeprefix('sha256:')}",
    )


def split_s3(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise BatchExecutionError(f"Batch execution requires an S3 URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


class BatchRoundExecutor:
    def __init__(
        self,
        *,
        s3: Any,
        batch: Any,
        logs: Any,
        exchange: str,
        job_name: str,
        job_queue: str,
        job_definition: str,
        timeout_seconds: int,
        parallel_jobs: int,
        poll_seconds: float = 20.0,
    ) -> None:
        self.s3 = s3
        self.batch = batch
        self.logs = logs
        self.exchange = exchange.rstrip("/")
        self.job_name = job_name
        self.job_queue = job_queue
        self.job_definition = job_definition
        self.timeout_seconds = timeout_seconds
        self.parallel_jobs = parallel_jobs
        self.poll_seconds = poll_seconds
        self._round = 0

    def __call__(
        self,
        configurations: tuple[dict[str, Any], ...],
        score: ScoreSpec,
    ) -> tuple[dict[str, int | float], ...]:
        round_index = self._round
        self._round += 1
        config_uris = tuple(
            self._publish_configuration(configuration, round_index)
            for configuration in configurations
        )
        groups = _groups(config_uris, self.parallel_jobs)
        job_ids = []
        for job_index, group in enumerate(groups):
            manifest_uri = self._publish_manifest(group, round_index, job_index)
            response = self.batch.submit_job(
                jobName=f"{self.job_name}-r{round_index:03d}-j{job_index}",
                jobQueue=self.job_queue,
                jobDefinition=self.job_definition,
                timeout={"attemptDurationSeconds": self.timeout_seconds},
                containerOverrides={
                    "environment": [
                        {"name": "TRAINER_MANIFEST", "value": manifest_uri},
                        {"name": "TRAINER_WORKSPACE", "value": "/tmp/trainer"},
                        {"name": "AWS_REGION", "value": REGION},
                        {"name": "AWS_DEFAULT_REGION", "value": REGION},
                    ]
                },
            )
            job_ids.append(response["jobId"])
        self._wait(job_ids)
        return self._score(configurations, score)

    def _publish_configuration(self, configuration: dict[str, Any], round_index: int) -> str:
        trial = int(configuration["identity"]["trial"])
        uri = f"{self.exchange}/round-{round_index:03d}/trial-{trial:06d}.json"
        self._put(uri, json.dumps(configuration, sort_keys=True).encode())
        return uri

    def _publish_manifest(
        self, config_uris: tuple[str, ...], round_index: int, job_index: int
    ) -> str:
        uri = f"{self.exchange}/round-{round_index:03d}/job-{job_index:03d}.json"
        self._put(uri, json.dumps({"runs": config_uris}, sort_keys=True).encode())
        return uri

    def _wait(self, job_ids: list[str]) -> None:
        pending = set(job_ids)
        while pending:
            jobs = self.batch.describe_jobs(jobs=job_ids)["jobs"]
            failed = [job for job in jobs if job["status"] == "FAILED"]
            if failed:
                for job in jobs:
                    if job["status"] not in TERMINAL:
                        self.batch.terminate_job(
                            jobId=job["jobId"], reason="trainerctl stopped failed round"
                        )
                messages = []
                for job in failed:
                    messages.append(job.get("statusReason", "Batch job failed"))
                    stream = job.get("container", {}).get("logStreamName")
                    if stream is not None:
                        events = self.logs.get_log_events(
                            logGroupName=JOB_LOG_GROUP,
                            logStreamName=stream,
                            limit=200,
                            startFromHead=False,
                        )["events"]
                        messages.extend(event["message"] for event in events)
                raise BatchExecutionError("\n".join(messages))
            pending = {job["jobId"] for job in jobs if job["status"] not in TERMINAL}
            if pending:
                time.sleep(self.poll_seconds)

    def _score(
        self,
        configurations: tuple[dict[str, Any], ...],
        score: ScoreSpec,
    ) -> tuple[dict[str, int | float], ...]:
        values = []
        with tempfile.TemporaryDirectory(prefix="trainer-score-") as directory:
            root = Path(directory)
            for configuration in configurations:
                trial = int(configuration["identity"]["trial"])
                artifacts = str(configuration["artifacts"]["root"]).rstrip("/")
                result = json.loads(self._get(f"{artifacts}/result.json"))
                if result["success"] is not True or int(result["identity"]["trial"]) != trial:
                    raise BatchExecutionError(f"artifact result does not complete trial {trial}")
                if "metrics.jsonl" not in result["artifacts"]:
                    raise BatchExecutionError(f"trial {trial} result declares no metrics.jsonl")
                metrics = root / f"trial-{trial:06d}-metrics.jsonl"
                metrics.write_bytes(self._get(f"{artifacts}/metrics.jsonl"))
                values.append({"trial": trial, "value": compute_score(metrics, score)})
        return tuple(values)

    def _put(self, uri: str, payload: bytes) -> None:
        bucket, key = split_s3(uri)
        self.s3.put_object(Bucket=bucket, Key=key, Body=payload)

    def _get(self, uri: str) -> bytes:
        bucket, key = split_s3(uri)
        return self.s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def _groups(values: tuple[str, ...], count: int) -> tuple[tuple[str, ...], ...]:
    if count < 1 or count > len(values):
        raise BatchExecutionError("parallel_jobs must be between one and the number of trials")
    base, remainder = divmod(len(values), count)
    groups = []
    offset = 0
    for index in range(count):
        size = base + (1 if index < remainder else 0)
        groups.append(values[offset : offset + size])
        offset += size
    return tuple(groups)
