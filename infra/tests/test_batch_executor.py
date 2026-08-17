from __future__ import annotations

import json
import tracemalloc
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeBatch, FakeLogs, FakeS3, publish_result, split

from trainer_infra.batch import (
    CHUNK_BYTES,
    REGION,
    BatchExecutionError,
    BatchRoundExecutor,
    batch_target,
)
from trainer_infra.scoring import ScoreSpec


def configuration(trial: int) -> dict[str, Any]:
    return {
        "contract": 10,
        "identity": {
            "run_id": f"run-t{trial}-s0",
            "experiment": "test",
            "launch_id": "launch",
            "trial": trial,
            "seed": 0,
            "role": "tuning",
            "digest": "sha256:" + "a" * 64,
        },
        "entry": "e",
        "artifacts": {"root": f"s3://bucket/artifacts/t{trial}"},
        "algorithm": {},
        "runtime": {},
        "logging": {},
    }


def score(window: tuple[int, int] = (0, 10)) -> ScoreSpec:
    return ScoreSpec(
        metric="objective",
        window_steps=window,
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


def publish_metrics(s3: FakeS3, config: dict[str, Any], path: Path, rows: int) -> int:
    """A metrics object of the shape a real run leaves, served from disk.

    From disk because its size is the subject here: an in-memory object would
    put a copy of it on the heap the moment it was fetched, which is exactly
    what these tests are measuring the absence of.
    """

    padding = "x" * 400
    with path.open("w", encoding="utf-8") as handle:
        for step in range(rows):
            row = {
                "step": step,
                "metrics": {"objective": float(step), "train/loss": 0.5, "note": padding},
            }
            handle.write(json.dumps(row) + "\n")
    bucket, key = split(config["artifacts"]["root"])
    s3.put_file(Bucket=bucket, Key=f"{key}/metrics.jsonl", Source=path)
    publish_result(s3, config)
    return path.stat().st_size


def test_batch_executor_packs_submits_collects_and_scores_a_round() -> None:
    s3 = FakeS3()
    batch = FakeBatch(s3)
    configurations = tuple(configuration(trial) for trial in range(4))

    results = executor(s3, batch)(configurations, score())

    assert results == tuple(
        {"trial": trial, "seed": 0, "value": float(trial + 1)} for trial in range(4)
    )
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


def test_a_metrics_object_is_never_asked_for_whole(tmp_path: Path) -> None:
    s3 = FakeS3()
    config = configuration(0)
    size = publish_metrics(s3, config, tmp_path / "metrics.jsonl", rows=8_000)
    assert size > 2 * CHUNK_BYTES

    executor(s3, FakeBatch(s3)).score((config,), score(window=(0, 8_000)))

    reads = s3.bodies[("bucket", "artifacts/t0/metrics.jsonl")].amounts
    assert None not in reads, "the whole object was requested in one read"
    assert all(amount == CHUNK_BYTES for amount in reads if amount is not None)
    assert len(reads) > 2


def test_scoring_a_large_object_costs_a_chunk_rather_than_the_object(tmp_path: Path) -> None:
    s3 = FakeS3()
    config = configuration(0)
    size = publish_metrics(s3, config, tmp_path / "metrics.jsonl", rows=40_000)
    scored = executor(s3, FakeBatch(s3))

    tracemalloc.start()
    try:
        results = scored.score((config,), score(window=(0, 40_000)))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert results == ({"trial": 0, "seed": 0, "value": 39_999.0},)
    assert size > 12 * CHUNK_BYTES
    assert peak < 6 * CHUNK_BYTES


def test_scoring_finished_artifacts_submits_no_job() -> None:
    s3 = FakeS3()
    batch = FakeBatch(s3)
    configurations = tuple(configuration(trial) for trial in range(2))
    executor(s3, batch)(configurations, score())
    batch.submitted.clear()

    again = executor(s3, batch).score(configurations, score())

    assert again == tuple(
        {"trial": trial, "seed": 0, "value": float(trial + 1)} for trial in range(2)
    )
    assert batch.submitted == []
