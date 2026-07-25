import pytest

from trainer_infra.loop import run_launch, select_best
from trainer_infra.report import TrialRecord


def test_select_best_picks_the_lowest_value_when_minimizing() -> None:
    records = [
        TrialRecord(trial=0, params={"x": 1}, value=3.0),
        TrialRecord(trial=1, params={"x": 2}, value=1.0),
        TrialRecord(trial=2, params={"x": 3}, value=2.0),
    ]

    best = select_best(records, maximize=False)

    assert best is not None
    assert best.trial == 1
    assert best.value == 1.0


def test_partial_submit_failure_terminates_already_submitted_jobs(
    launch_for_batch,
) -> None:
    class PartialSubmitFailureBackend:
        def __init__(self) -> None:
            self.terminated: list[str] = []

        def submit(self, launch, manifest_uri: str, name: str) -> str:
            del launch, manifest_uri
            if name == "round-000-job-1":
                raise RuntimeError("submit failed")
            return f"id-{name}"

        def wait(self, job_ids):
            del job_ids
            return []

        def terminate(self, job_ids) -> None:
            self.terminated.extend(job_ids)

        def log_tail(self, result, lines: int) -> str:
            del result, lines
            return ""

    backend = PartialSubmitFailureBackend()
    with pytest.raises(RuntimeError, match="submit failed"):
        run_launch(launch_for_batch, backend)
    assert backend.terminated == ["id-round-000-job-0"]


def test_select_best_picks_the_highest_value_when_maximizing() -> None:
    records = [
        TrialRecord(trial=0, params={"x": 1}, value=3.0),
        TrialRecord(trial=1, params={"x": 2}, value=1.0),
        TrialRecord(trial=2, params={"x": 3}, value=2.0),
    ]

    best = select_best(records, maximize=True)

    assert best is not None
    assert best.trial == 0
    assert best.value == 3.0
