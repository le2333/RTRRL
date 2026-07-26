from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from trainer_infra.launch import Launch


@dataclass(frozen=True)
class JobResult:
    job_id: str
    name: str
    succeeded: bool
    log_stream: str | None = None
    reason: str | None = None


class Backend(Protocol):
    def submit(self, launch: Launch, manifest_uri: str, name: str) -> str: ...

    def wait(self, job_ids: Sequence[str]) -> list[JobResult]:
        """Block until every job finishes or any job fails, whichever comes first.

        On early return after a failure, the result list covers only jobs that
        reached a terminal state and may be shorter than ``job_ids``.
        """
        ...

    def terminate(self, job_ids: Sequence[str]) -> None:
        """Stop running jobs; must tolerate ids that have already finished."""
        ...

    def log_tail(self, result: JobResult, lines: int) -> str: ...
