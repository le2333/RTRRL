from __future__ import annotations

from datetime import datetime, timezone
import hashlib

import pytest

from training_sdk.execution import (
    CompletionMarker,
    JobBundle,
    JobQuery,
    RunBundle,
    canonical_json,
)

IMAGE = "registry.example/repository/image@sha256:" + "a" * 64


def context() -> dict[str, object]:
    return {
        "experiment_name": "experiment",
        "experiment_id": "exp-1",
        "group": "group",
        "script": "script",
        "run_id": "experiment:group:0001",
        "run_number": 1,
        "trial_number": 1,
        "seed": 7,
        "metadata": {},
        "environment": {"name": "test"},
        "training_budget": {"env_steps": 10},
        "fixed_parameters": {"seed": 7},
        "sampled_parameters": {},
        "final_parameters": {"seed": 7},
        "image_digest": IMAGE,
        "resource_profile": "c7am",
        "artifact_directory": "/worker/generated-at-runtime",
        "logging": {},
        "objective": {},
    }


def run_bundle() -> RunBundle:
    payload = context()
    config = "parameters:\n  seed: 7\n"
    return RunBundle(
        run_id="experiment:group:0001",
        argv=("python", "train.py", "--config", "{config_path}"),
        image_digest=IMAGE,
        resource_profile="c7am",
        config_yaml=config,
        config_sha256=hashlib.sha256(config.encode()).hexdigest(),
        run_context=payload,
        run_context_sha256=hashlib.sha256(canonical_json(payload).encode()).hexdigest(),
        artifact_prefix="experiments/exp-1/groups/group/runs/run-1/input/",
    )


def test_execution_records_are_self_contained_strict_and_canonical() -> None:
    run = run_bundle()
    job = JobBundle(
        job_id="bundle-1",
        image_digest=IMAGE,
        resource_profile="c7am",
        runs=(run,),
    )
    marker = CompletionMarker(
        run_id=run.run_id,
        exit_code=1,
        error="artifact upload failed",
        started_at=datetime(2026, 7, 21, tzinfo=timezone.utc),
        finished_at=datetime(2026, 7, 21, tzinfo=timezone.utc),
    )
    query = JobQuery(job_id="aws-job", status="FAILED", status_reason="raw")

    for record in (run, job, marker, query):
        assert type(record).from_json(record.to_json()) == record
        assert record.sha256 == hashlib.sha256(record.to_json().encode()).hexdigest()
        with pytest.raises(ValueError, match="canonical"):
            type(record).from_json(" " + record.to_json())


def test_run_argv_requires_one_worker_generated_config_placeholder() -> None:
    payload = run_bundle().model_dump(mode="json")
    payload["argv"] = ["python", "train.py", "--config", "/tmp/untrusted"]
    with pytest.raises(ValueError, match="config_path"):
        RunBundle.model_validate(payload)
