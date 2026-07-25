from __future__ import annotations

import time
from collections.abc import Sequence

from trainer_infra.backends.base import JobResult
from trainer_infra.launch import Launch
from trainer_infra.queues import JOB_LOG_GROUP

TERMINAL = {"SUCCEEDED", "FAILED"}


class BatchBackend:
    def __init__(self, batch_client, logs_client, poll_seconds: float = 20.0) -> None:
        self._batch = batch_client
        self._logs = logs_client
        self._poll_seconds = poll_seconds
        self._names: dict[str, str] = {}

    def submit(self, launch: Launch, manifest_uri: str, name: str) -> str:
        compute = launch.plan.experiment.compute
        response = self._batch.submit_job(
            jobName=f"{launch.plan.experiment.name}-{launch.launch_id}-{name}",
            jobQueue=launch.plan.queue,
            jobDefinition=launch.plan.job_definition,
            timeout={"attemptDurationSeconds": compute.timeout_minutes * 60},
            containerOverrides={
                "environment": [
                    {"name": "TRAINER_MANIFEST", "value": manifest_uri},
                    {"name": "TRAINER_WORKSPACE", "value": "/tmp/trainer"},
                    {
                        "name": "TRAINER_STARTUP_SECONDS",
                        "value": str(compute.startup_minutes * 60),
                    },
                    {"name": "TRAINER_STALL_FACTOR", "value": str(compute.stall_factor)},
                ]
            },
        )
        job_id = response["jobId"]
        self._names[job_id] = name
        return job_id

    def wait(self, job_ids: Sequence[str]) -> list[JobResult]:
        pending = list(job_ids)
        finished: dict[str, dict] = {}
        while pending:
            described = self._batch.describe_jobs(jobs=pending)["jobs"]
            for job in described:
                if job["status"] in TERMINAL:
                    finished[job["jobId"]] = job
            pending = [job_id for job_id in pending if job_id not in finished]
            # Returning the moment one job fails is what lets the loop terminate
            # the survivors instead of paying for a round that is already doomed.
            if any(job["status"] == "FAILED" for job in finished.values()):
                break
            if pending:
                time.sleep(self._poll_seconds)
        terminal_ids = [job_id for job_id in job_ids if job_id in finished]
        return [
            JobResult(
                job_id=job_id,
                name=self._names.get(job_id, job_id),
                succeeded=finished[job_id]["status"] == "SUCCEEDED",
                log_stream=finished[job_id].get("container", {}).get("logStreamName"),
                reason=(
                    None
                    if finished[job_id]["status"] == "SUCCEEDED"
                    else finished[job_id].get("statusReason")
                    or f"exit code {finished[job_id].get('container', {}).get('exitCode')}"
                ),
            )
            for job_id in terminal_ids
        ]

    def terminate(self, job_ids: Sequence[str]) -> None:
        for job_id in job_ids:
            self._batch.terminate_job(jobId=job_id, reason="trainerctl stopped")

    def log_tail(self, result: JobResult, lines: int) -> str:
        if result.log_stream is None:
            return ""
        events = self._logs.get_log_events(
            logGroupName=JOB_LOG_GROUP,
            logStreamName=result.log_stream,
            limit=lines,
            startFromHead=False,
        )["events"]
        return "\n".join(event["message"] for event in events)
