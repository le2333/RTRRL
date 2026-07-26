from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from training_sdk import objects
from training_sdk.contract import RunConfig

from trainer_infra.launch import Launch, config_uri


@dataclass(frozen=True)
class JobPlan:
    manifest_uri: str
    trials: tuple[int, ...]


def split(count: int, jobs: int) -> list[int]:
    if jobs < 1 or count < jobs:
        raise ValueError("jobs must be between one and the number of trials")
    base, remainder = divmod(count, jobs)
    return [base + (1 if index < remainder else 0) for index in range(jobs)]


def publish_round(
    launch: Launch, round_index: int, configs: Sequence[RunConfig], jobs: int
) -> list[JobPlan]:
    uris: list[str] = []
    for config in configs:
        uri = config_uri(launch, config.trial)
        objects.put_bytes(uri, config.model_dump_json().encode())
        uris.append(uri)

    plans: list[JobPlan] = []
    offset = 0
    for job_index, size in enumerate(split(len(configs), jobs)):
        group = slice(offset, offset + size)
        offset += size
        manifest_uri = f"{launch.prefix}/rounds/round-{round_index:03d}/job-{job_index}.json"
        objects.put_bytes(manifest_uri, json.dumps({"runs": uris[group]}).encode())
        plans.append(
            JobPlan(
                manifest_uri=manifest_uri,
                trials=tuple(config.trial for config in configs[group]),
            )
        )
    return plans
