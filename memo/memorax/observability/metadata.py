"""Run identity projected for observation backends."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    experiment: str
    launch_id: str
    trial: int
    entry: str
    digest: str
