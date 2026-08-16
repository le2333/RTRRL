"""A Batch HPO scored against metrics the size real runs produce.

Every other test here uses metrics small enough to hold, which is exactly the
property that hid the failure: a 20M-step trial left 3.6 GB in S3, the
controller read it into memory to score it, and the kernel killed the
controller with both trials already finished and paid for.

This is opt-in and slow -- it writes and parses a gigabyte:

    uv run pytest -m stress
    TRAINER_STRESS_METRICS_BYTES=134217728 uv run pytest -m stress   # a shorter one
"""

from __future__ import annotations

import json
import os
import tracemalloc
from pathlib import Path
from typing import Any

import optuna
import pytest
from fakes import FakeBatch, FakeLogs, FakeS3, publish_result, split

from trainer_infra import ExperimentRunner
from trainer_infra.batch import BatchRoundExecutor

pytestmark = pytest.mark.stress

LAUNCH = "20260815-090000"
METRICS_BYTES = int(os.environ.get("TRAINER_STRESS_METRICS_BYTES", str(1 << 30)))
# What the control plane may spend on scoring, whatever the object weighs. The
# instance that hosts the study has around 250 MiB free in total.
BUDGET_BYTES = 64 << 20
FINAL_RETURN = 13.5


def write_metrics(path: Path, size_bytes: int) -> int:
    """The shape of a finished run: mostly training rows, one final evaluation.

    The score metric is on every row so that the reduction really walks the
    file, and only the last row sits at the top of the score window, so the
    expected score is one number no matter how large the file grew.
    """

    padding = "z" * 400
    with path.open("w", encoding="utf-8") as handle:
        written = 0
        step = 0
        while written < size_bytes:
            row = {
                "step": step % 200,
                "metrics": {
                    "eval/episode/return_per_step": float(step % 200),
                    "train/loss": 0.5,
                    "note": padding,
                },
            }
            written += handle.write(json.dumps(row) + "\n")
            step += 1
        final = {"step": 200, "metrics": {"eval/episode/return_per_step": FINAL_RETURN}}
        handle.write(json.dumps(final) + "\n")
    return path.stat().st_size


def publish_large_metrics(metrics: Path) -> Any:
    def publish(s3: FakeS3, config: dict[str, Any]) -> None:
        bucket, key = split(config["artifacts"]["root"])
        s3.put_file(Bucket=bucket, Key=f"{key}/metrics.jsonl", Source=metrics)
        publish_result(s3, config)

    return publish


def test_a_gigabyte_of_metrics_is_scored_and_the_study_moves_on(
    experiment: Any, catalog: Any, tmp_path: Path
) -> None:
    experiment["hpo"]["rounds"] = 2
    experiment["hpo"]["trials_per_round"] = 1
    size = write_metrics(tmp_path / "metrics.jsonl", METRICS_BYTES)
    s3 = FakeS3()
    batch = FakeBatch(s3, publish=publish_large_metrics(tmp_path / "metrics.jsonl"))
    executor = BatchRoundExecutor(
        s3=s3,
        batch=batch,
        logs=FakeLogs(),
        exchange="s3://artifacts/trainer/streamac-test/control",
        job_name=f"stream-ac-test-{LAUNCH}",
        job_queue="dev-cpu-c7al-queue",
        job_definition="trainer-c7al-digest",
        timeout_seconds=5400,
        parallel_jobs=1,
        poll_seconds=0,
    )
    runner = ExperimentRunner(
        experiment=experiment,
        catalog=catalog,
        database=tmp_path / "study.db",
        launch_id=LAUNCH,
    )

    tracemalloc.start()
    try:
        study = runner.run(executor)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert size >= METRICS_BYTES
    assert len(batch.submitted) == 2, "the second round was never asked for"
    assert [trial.state for trial in study.trials] == [optuna.trial.TrialState.COMPLETE] * 2
    assert [trial.value for trial in study.trials] == [FINAL_RETURN] * 2
    assert peak < BUDGET_BYTES
