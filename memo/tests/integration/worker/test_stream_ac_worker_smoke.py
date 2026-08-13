"""One real Worker child running StreamAC and publishing its trace reading."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest
from aim import Repo

from deployment.catalog import write_catalog
from deployment.contract import CONTRACT_VERSION
from entries._contract import RunSpec
from memorax.algorithms.stream_ac import PARAMETERS
from memorax.parameters import expand
from worker import objects
from worker.worker import run_manifest

pytestmark = [pytest.mark.integration, pytest.mark.service]

TRACE_METRIC = "train/episode/update.critic.trace_norm.recurrence"


def stream_ac_parameters() -> dict:
    return expand(
        PARAMETERS,
        {
            "actor.head.kind": "global_std",
            "actor.optimizer.bound.kind": "none",
            "actor.optimizer.base.kind": "sgd",
            "actor.optimizer.base.sgd.lr": 1e-4,
            "critic.head.kind": "value",
            "critic.optimizer.bound.kind": "none",
            "critic.optimizer.base.kind": "sgd",
            "critic.optimizer.base.sgd.lr": 1e-4,
            "normalization.observation.kind": "none",
            "normalization.reward.kind": "none",
            "backbone.kind": "rtu",
            "backbone.rtu.hidden_dim": 2,
            "credit.kind": "rtrl",
            "meta_rl": False,
            "gamma": 0.9,
            "trace_lambda": 0.8,
            "entropy_coefficient": 0.01,
        },
    )


def test_worker_runs_stream_ac_and_publishes_critic_trace_norm(
    s3_base: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_path = tmp_path / "catalog.json"
    write_catalog(catalog_path)
    monkeypatch.setenv("TRAINER_CATALOG", str(catalog_path))
    monkeypatch.setenv(
        "PATH", f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"
    )

    aim_path = tmp_path / "aim"
    Repo.from_path(str(aim_path), init=True)
    artifact_root = f"{s3_base}/smoke/artifacts"
    config = RunSpec.model_validate(
        {
            "contract": CONTRACT_VERSION,
            "identity": {
                "run_id": "stream-ac-worker-smoke-t0",
                "experiment": "stream-ac-worker-smoke",
                "launch_id": "20260812-000000",
                "trial": 0,
                "digest": "local@sha256:" + "a" * 64,
            },
            "entry": "stream_ac",
            "artifacts": {"root": artifact_root},
            "algorithm": {
                "environment": {
                    "id": "brax::hopper",
                    "backend": "spring",
                    "episode_length": 4,
                    "observed": [0, 2, 4],
                },
                "num_envs": 1,
                "parameters": stream_ac_parameters(),
            },
            "runtime": {
                "seed": 0,
                "total_steps": 4,
                "epoch_steps": 4,
                "evaluation_steps": 0,
            },
            "logging": {"aim": {"url": str(aim_path)}},
        }
    )
    config_uri = f"{s3_base}/smoke/config.json"
    manifest_uri = f"{s3_base}/smoke/manifest.json"
    objects.put_bytes(config_uri, config.model_dump_json().encode())
    objects.put_bytes(manifest_uri, json.dumps({"runs": [config_uri]}).encode())

    run_manifest(manifest_uri, tmp_path / "worker")

    records = [
        json.loads(line)
        for line in objects.get_bytes(f"{artifact_root}/metrics.jsonl")
        .decode()
        .splitlines()
    ]
    values = [
        row["metrics"][TRACE_METRIC]
        for row in records
        if TRACE_METRIC in row["metrics"]
    ]
    assert values
    assert all(math.isfinite(value) for value in values)

    result = json.loads(objects.get_bytes(f"{artifact_root}/result.json"))
    assert result["identity"]["run_id"] == config.identity.run_id
    assert result["success"] is True
    assert "metrics.jsonl" in result["artifacts"]

    repo = Repo.from_path(str(aim_path))
    runs = list(repo.iter_runs())
    assert len(runs) == 1
    metrics = {metric.name: metric for metric in runs[0].metrics()}
    assert TRACE_METRIC in metrics
    aim_values = metrics[TRACE_METRIC].data.values_list()[0]
    assert aim_values
    assert all(math.isfinite(value) for value in aim_values)
    assert values[-1] == aim_values[-1]
