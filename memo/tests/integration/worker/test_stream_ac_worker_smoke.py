"""One real Worker child running StreamAC and publishing its trace reading."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest
from aim import Repo

from memorax.algorithms.stream_ac import PARAMETERS
from memorax.parameters import expand
from deployment.catalog import write_catalog
from deployment.contract import CONTRACT_VERSION
from entries._contract import RunSpec
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


def test_worker_runs_stream_ac_and_records_critic_trace_norm(
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
    score_uri = f"{s3_base}/smoke/score.json"
    config = RunSpec.model_validate(
        {
            "contract": CONTRACT_VERSION,
            "run_id": "stream-ac-worker-smoke-t0",
            "experiment": "stream-ac-worker-smoke",
            "launch_id": "20260812-000000",
            "trial": 0,
            "entry": "stream_ac",
            "digest": "local@sha256:" + "a" * 64,
            "environment": {
                "id": "brax::hopper",
                "backend": "spring",
                "seed": 0,
                "episode_length": 4,
                "observed": [0, 2, 4],
            },
            "training": {"num_envs": 1, "total_steps": 4, "epoch_steps": 4},
            "evaluation": {"steps": 0},
            "params": stream_ac_parameters(),
            "logging": {"aim": str(aim_path), "enable_rerun": False},
            "score": {
                "metric": TRACE_METRIC,
                "window_steps": [0, 4],
                "reduce": "last",
                "direction": "maximize",
                "non_finite": "worst",
                "s3": score_uri,
            },
        }
    )
    config_uri = f"{s3_base}/smoke/config.json"
    manifest_uri = f"{s3_base}/smoke/manifest.json"
    objects.put_bytes(config_uri, config.model_dump_json().encode())
    objects.put_bytes(manifest_uri, json.dumps({"runs": [config_uri]}).encode())

    run_manifest(manifest_uri, tmp_path / "worker")

    score = json.loads(objects.get_bytes(score_uri))
    assert score["run_id"] == config.run_id
    assert math.isfinite(score["value"])

    repo = Repo.from_path(str(aim_path))
    runs = list(repo.iter_runs())
    assert len(runs) == 1
    run = runs[0]
    metrics = {metric.name: metric for metric in run.metrics()}
    assert TRACE_METRIC in metrics
    values = metrics[TRACE_METRIC].data.values_list()[0]
    assert values
    assert all(math.isfinite(value) for value in values)
    assert score["value"] == values[-1]
