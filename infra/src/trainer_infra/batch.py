"""AWS Batch round execution for version-8 Worker manifests."""

from __future__ import annotations

import codecs
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from trainer_infra.experiment import run_name
from trainer_infra.scoring import ScoreSpec, score_lines

REGION = "eu-north-1"
JOB_LOG_GROUP = "/trainer/jobs"
TERMINAL = {"SUCCEEDED", "FAILED"}
# How much of a metrics object is in memory at once. Large enough that a
# multi-gigabyte body is read in thousands of calls rather than millions,
# small enough that the control plane never notices.
CHUNK_BYTES = 1 << 20

_PROFILES = {
    "c7a.medium": ("c7am", "run-cpu-c7am-queue", "dev-cpu-c7am-queue"),
    "c7a.large": ("c7al", "run-cpu-c7al-queue", "dev-cpu-c7al-queue"),
    "c7a.xlarge": ("c7ax", "run-cpu-c7ax-queue", "dev-cpu-c7ax-queue"),
    "g6.xlarge": ("g6x", "run-gpu-queue", "dev-gpu-queue"),
}


class BatchExecutionError(RuntimeError):
    """A Batch job failed before its complete round could be scored."""


@dataclass(frozen=True)
class BatchTarget:
    queue: str
    job_definition: str


def batch_target(instance_type: str, tier: str, digest: str) -> BatchTarget:
    profile, run_queue, dev_queue = _PROFILES[instance_type]
    queue = {"run": run_queue, "dev": dev_queue}[tier]
    return BatchTarget(
        queue=queue,
        job_definition=f"trainer-{profile}-{digest.removeprefix('sha256:')}",
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
        bodies = self._manifests(configurations, config_uris)
        job_ids = []
        for job_index, body in enumerate(bodies):
            manifest_uri = self._publish_manifest(body, round_index, job_index)
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
        return self.score(configurations, score)

    def _publish_configuration(self, configuration: dict[str, Any], round_index: int) -> str:
        name = run_name(configuration)
        uri = f"{self.exchange}/round-{round_index:03d}/{name}.json"
        self._put(uri, json.dumps(configuration, sort_keys=True).encode())
        return uri

    def _manifests(
        self,
        configurations: tuple[dict[str, Any], ...],
        config_uris: tuple[str, ...],
    ) -> tuple[dict[str, Any], ...]:
        """One manifest body per job: this round's runs, split across the jobs.

        The seam an ensemble executor overrides. Nothing else about running a
        round differs between the two channels -- the same configurations are
        published, the same jobs submitted, the same scores read back -- so this
        is the only thing a subclass has to say.
        """

        del configurations
        return tuple(
            {"runs": list(uris)}
            for uris in _groups(config_uris, self.parallel_jobs)
        )

    def _publish_manifest(
        self, body: dict[str, Any], round_index: int, job_index: int
    ) -> str:
        uri = f"{self.exchange}/round-{round_index:03d}/job-{job_index:03d}.json"
        self._put(uri, json.dumps(body, sort_keys=True).encode())
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

    def score(
        self,
        configurations: tuple[dict[str, Any], ...],
        score: ScoreSpec,
    ) -> tuple[dict[str, int | float], ...]:
        """Score what the workers of these configurations have already uploaded.

        Submitting nothing is the point: a round that finished remotely can be
        scored again -- after the controller died, say -- without paying for
        the training a second time.
        """

        values = []
        for configuration in configurations:
            identity = configuration["identity"]
            trial, seed = int(identity["trial"]), int(identity["seed"])
            name = run_name(configuration)
            artifacts = str(configuration["artifacts"]["root"]).rstrip("/")
            result = json.loads(self._get(f"{artifacts}/result.json"))
            produced = result["identity"]
            if result["success"] is not True or (
                int(produced["trial"]),
                int(produced["seed"]),
            ) != (trial, seed):
                raise BatchExecutionError(f"artifact result does not complete {name}")
            if "metrics.jsonl" not in result["artifacts"]:
                raise BatchExecutionError(f"{name} result declares no metrics.jsonl")
            value = self._score_metrics(f"{artifacts}/metrics.jsonl", score)
            values.append({"trial": trial, "seed": seed, "value": value})
        return tuple(values)

    def _score_metrics(self, uri: str, score: ScoreSpec) -> float:
        """Reduce a metrics object as it arrives, keeping neither it nor a copy.

        The objects this reads are gigabytes -- larger than the control
        plane's memory, and larger than the disk a temporary copy would land
        on. What the score needs from them is a handful of numbers, so the
        body is decoded a chunk at a time and nothing else is retained.
        """

        bucket, key = split_s3(uri)
        body = self.s3.get_object(Bucket=bucket, Key=key)["Body"]
        try:
            return score_lines(_lines(body), score)
        finally:
            body.close()

    def _put(self, uri: str, payload: bytes) -> None:
        bucket, key = split_s3(uri)
        self.s3.put_object(Bucket=bucket, Key=key, Body=payload)

    def _get(self, uri: str) -> bytes:
        bucket, key = split_s3(uri)
        return self.s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def _lines(body: Any, chunk_bytes: int = CHUNK_BYTES) -> Iterator[str]:
    """Cut a byte stream into text lines, holding one chunk of it at a time.

    ``codecs`` and ``io`` both offer a reader for this, and both ask the
    stream for a line at a time, which over an S3 body is a call per line.
    Cutting up whole chunks here is the same work in three orders of magnitude
    fewer reads.
    """

    decoder = codecs.getincrementaldecoder("utf-8")()
    remainder = ""
    while True:
        chunk = body.read(chunk_bytes)
        if not chunk:
            break
        *complete, remainder = (remainder + decoder.decode(chunk)).split("\n")
        yield from complete
    remainder += decoder.decode(b"", final=True)
    if remainder:
        yield remainder


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
