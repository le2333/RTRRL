from __future__ import annotations

from dataclasses import dataclass

from trainer_infra.preflight import PreflightError

JOB_LOG_GROUP = "/trainer/jobs"


@dataclass(frozen=True)
class QueueBinding:
    instance_type: str
    profile: str
    run_queue: str
    dev_queue: str
    max_vcpus: int
    vcpus_per_job: int
    gpus_per_job: int = 0

    @property
    def concurrency(self) -> int:
        return self.max_vcpus // self.vcpus_per_job

    def queue(self, tier: str) -> str:
        if tier == "run":
            return self.run_queue
        if tier == "dev":
            return self.dev_queue
        raise PreflightError(f"unknown queue tier {tier!r}; use run or dev")


QUEUES: dict[str, QueueBinding] = {
    "c7a.medium": QueueBinding(
        "c7a.medium", "c7am", "run-cpu-c7am-queue", "dev-cpu-c7am-queue", 16, 1
    ),
    "c7a.large": QueueBinding(
        "c7a.large", "c7al", "run-cpu-c7al-queue", "dev-cpu-c7al-queue", 32, 2
    ),
    "c7a.xlarge": QueueBinding(
        "c7a.xlarge", "c7ax", "run-cpu-c7ax-queue", "dev-cpu-c7ax-queue", 16, 4
    ),
    "g6.xlarge": QueueBinding(
        "g6.xlarge", "g6x", "run-gpu-queue", "dev-gpu-queue", 32, 4, 1
    ),
}


def binding(instance_type: str) -> QueueBinding:
    try:
        return QUEUES[instance_type]
    except KeyError:
        available = ", ".join(sorted(QUEUES))
        raise PreflightError(
            f"instance_type {instance_type!r} has no queue; available: {available}"
        ) from None


def job_definition_name(entry: QueueBinding, digest: str) -> str:
    return f"trainer-{entry.profile}-{digest.removeprefix('sha256:')}"
