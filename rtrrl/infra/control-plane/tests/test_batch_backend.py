import json
import time
from pathlib import Path

from botocore.session import get_session

from trainer_infra.backends.batch import BatchBackend
from trainer_infra.queues import REGION

RECORDED = json.loads(Path("tests/data/batch-describe-jobs.json").read_text())
DESCRIBE_JOBS_OUTPUT = (
    get_session().get_service_model("batch").operation_model("DescribeJobs").output_shape
)
JOB_DETAIL = DESCRIBE_JOBS_OUTPUT.members["jobs"].member


def assert_fixture_matches_shape(value: object, shape: object, path: str = "") -> None:
    type_name = shape.type_name
    if type_name == "structure":
        assert isinstance(value, dict), f"{path or 'root'} should be a structure"
        for key, nested in value.items():
            assert key in shape.members, f"unknown key {path}.{key} in fixture"
            assert_fixture_matches_shape(
                nested, shape.members[key], f"{path}.{key}" if path else key
            )
    elif type_name == "list":
        assert isinstance(value, list), f"{path} should be a list"
        for index, item in enumerate(value):
            assert_fixture_matches_shape(item, shape.member, f"{path}[{index}]")
    elif type_name == "map":
        assert isinstance(value, dict), f"{path} should be a map"
        for key, nested in value.items():
            assert_fixture_matches_shape(nested, shape.value, f"{path}[{key!r}]")
    elif type_name == "string":
        assert isinstance(value, str), f"{path} should be a string"
        if shape.enum:
            assert value in shape.enum, (
                f"{path} value {value!r} is not one of {shape.enum}"
            )
    else:
        assert value is not None, f"{path} should be a scalar value"


def test_fixture_matches_describe_jobs_output_shape() -> None:
    assert_fixture_matches_shape(RECORDED, DESCRIBE_JOBS_OUTPUT)


def test_batch_backend_status_literals_are_job_status_enum_members() -> None:
    status_shape = JOB_DETAIL.members["status"]
    assert status_shape.enum is not None
    for literal in ("SUCCEEDED", "FAILED"):
        assert literal in status_shape.enum


class FakeBatch:
    def __init__(
        self,
        sequence: list[str] | None = None,
        *,
        job_sequences: dict[str, list[str]] | None = None,
    ) -> None:
        self.sequence = list(sequence or [])
        self.job_sequences = {
            job_id: list(statuses) for job_id, statuses in (job_sequences or {}).items()
        }
        self.submitted: list[dict] = []
        self.terminated: list[str] = []
        self.describe_calls = 0

    def submit_job(self, **kwargs: object) -> dict:
        self.submitted.append(kwargs)
        return {"jobId": f"job-{len(self.submitted)}"}

    def _status_for(self, job_id: str) -> str:
        if job_id in self.job_sequences:
            sequence = self.job_sequences[job_id]
            if sequence:
                return sequence.pop(0)
            return "SUCCEEDED"
        if self.sequence:
            return self.sequence.pop(0)
        return "SUCCEEDED"

    def describe_jobs(self, jobs: list[str]) -> dict:
        self.describe_calls += 1
        payload = json.loads(json.dumps(RECORDED))
        entry = payload["jobs"][0]
        entries = []
        for job_id in jobs:
            item = json.loads(json.dumps(entry))
            status = self._status_for(job_id)
            item["jobId"] = job_id
            item["status"] = status
            item["container"]["exitCode"] = 0 if status == "SUCCEEDED" else 3
            if status == "FAILED":
                item["statusReason"] = "Essential container in task exited"
            entries.append(item)
        return {"jobs": entries}

    def terminate_job(self, jobId: str, reason: str) -> dict:
        self.terminated.append(jobId)
        return {}


class FakeLogs:
    def get_log_events(self, **kwargs: object) -> dict:
        return {
            "events": [
                {"message": "Traceback (most recent call last):"},
                {"message": "RuntimeError: boom"},
            ]
        }


def test_submit_passes_manifest_and_timeout(launch_for_batch) -> None:
    batch = FakeBatch(["SUCCEEDED"])
    backend = BatchBackend(batch, FakeLogs(), poll_seconds=0)
    backend.submit(launch_for_batch, "s3://bucket/manifest.json", "round-000-job-0")
    request = batch.submitted[0]
    environment = {
        item["name"]: item["value"]
        for item in request["containerOverrides"]["environment"]
    }
    assert environment["TRAINER_MANIFEST"] == "s3://bucket/manifest.json"
    assert environment["TRAINER_WORKSPACE"] == "/tmp/trainer"
    assert environment["TRAINER_STARTUP_SECONDS"] == "600"
    assert environment["TRAINER_STALL_FACTOR"] == "10"
    assert request["timeout"]["attemptDurationSeconds"] == 60 * 60
    assert request["jobQueue"] == "run-cpu-c7am-queue"


def test_submit_tells_the_container_which_region_it_is_in(launch_for_batch) -> None:
    """Batch says who the container is but not where, and boto3 will not guess.

    Every S3 call the worker makes — starting with reading its own manifest — needs
    a region, and an unset one raises NoRegionError on a host already being billed.
    """
    batch = FakeBatch(["SUCCEEDED"])
    backend = BatchBackend(batch, FakeLogs(), poll_seconds=0)
    backend.submit(launch_for_batch, "s3://bucket/manifest.json", "round-000-job-0")
    environment = {
        item["name"]: item["value"]
        for item in batch.submitted[0]["containerOverrides"]["environment"]
    }

    assert environment["AWS_REGION"] == REGION
    assert environment["AWS_DEFAULT_REGION"] == REGION


def test_wait_polls_until_every_job_is_terminal(launch_for_batch) -> None:
    batch = FakeBatch(["RUNNABLE", "RUNNING", "SUCCEEDED"])
    backend = BatchBackend(batch, FakeLogs(), poll_seconds=0)
    job_id = backend.submit(launch_for_batch, "s3://bucket/m.json", "job-0")
    results = backend.wait([job_id])
    assert results[0].succeeded is True
    assert results[0].reason is None
    assert not batch.sequence


def test_failed_job_exposes_its_log_tail(launch_for_batch) -> None:
    batch = FakeBatch(["FAILED"])
    backend = BatchBackend(batch, FakeLogs(), poll_seconds=0)
    job_id = backend.submit(launch_for_batch, "s3://bucket/m.json", "job-0")
    result = backend.wait([job_id])[0]
    assert result.succeeded is False
    assert result.reason is not None
    assert "RuntimeError: boom" in backend.log_tail(result, 10)


def test_terminate_calls_batch_for_every_job(launch_for_batch) -> None:
    batch = FakeBatch(["SUCCEEDED"])
    backend = BatchBackend(batch, FakeLogs(), poll_seconds=0)
    job_id = backend.submit(launch_for_batch, "s3://bucket/m.json", "job-0")
    backend.terminate([job_id])
    assert batch.terminated == [job_id]


def test_terminate_tolerates_already_finished_jobs(launch_for_batch) -> None:
    batch = FakeBatch(
        job_sequences={
            "job-1": ["SUCCEEDED"],
            "job-2": ["RUNNING", "RUNNING"],
        }
    )
    backend = BatchBackend(batch, FakeLogs(), poll_seconds=0)
    finished_id = backend.submit(launch_for_batch, "s3://bucket/finished.json", "job-finished")
    running_id = backend.submit(launch_for_batch, "s3://bucket/running.json", "job-running")
    backend.wait([finished_id])
    backend.terminate([finished_id, running_id])
    assert set(batch.terminated) == {finished_id, running_id}


def test_wait_returns_early_when_a_sibling_is_still_running(launch_for_batch) -> None:
    batch = FakeBatch(
        job_sequences={
            "job-1": ["RUNNING", "FAILED"],
            "job-2": ["RUNNING", "RUNNING", "RUNNING", "SUCCEEDED"],
        }
    )
    backend = BatchBackend(batch, FakeLogs(), poll_seconds=0)
    fail_id = backend.submit(launch_for_batch, "s3://bucket/fail.json", "job-fail")
    slow_id = backend.submit(launch_for_batch, "s3://bucket/slow.json", "job-slow")
    assert fail_id == "job-1"
    assert slow_id == "job-2"

    started = time.monotonic()
    results = backend.wait([fail_id, slow_id])
    elapsed = time.monotonic() - started

    assert elapsed < 1.0
    assert len(results) == 1
    assert results[0].job_id == fail_id
    assert results[0].succeeded is False
    assert batch.job_sequences["job-2"]
    assert batch.describe_calls == 2


def test_log_tail_without_stream_returns_empty_string(launch_for_batch) -> None:
    batch = FakeBatch(["SUCCEEDED"])
    backend = BatchBackend(batch, FakeLogs(), poll_seconds=0)
    job_id = backend.submit(launch_for_batch, "s3://bucket/m.json", "job-0")
    result = backend.wait([job_id])[0]
    result_without_stream = result.__class__(
        job_id=result.job_id,
        name=result.name,
        succeeded=result.succeeded,
        log_stream=None,
        reason=result.reason,
    )
    assert backend.log_tail(result_without_stream, 10) == ""


def test_successful_job_has_no_reason(launch_for_batch) -> None:
    batch = FakeBatch(["SUCCEEDED"])
    backend = BatchBackend(batch, FakeLogs(), poll_seconds=0)
    job_id = backend.submit(launch_for_batch, "s3://bucket/m.json", "job-0")
    result = backend.wait([job_id])[0]
    assert result.succeeded is True
    assert result.reason is None
