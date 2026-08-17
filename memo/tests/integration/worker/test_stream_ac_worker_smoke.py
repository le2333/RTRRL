"""A real Worker child publishes each algorithm's declared trace readings."""

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
from memorax.algorithms.rtrrl_aaai import PARAMETERS as RTRRL_PARAMETERS
from memorax.algorithms.stream_ac import PARAMETERS as STREAM_AC_PARAMETERS
from memorax.parameters import expand
from worker import objects
from worker.worker import run_manifest

pytestmark = [pytest.mark.integration, pytest.mark.service]

STREAM_AC_TRACE_METRICS = tuple(
    name
    for base in ("train/episode/update.critic.trace_norm.recurrence",)
    for name in (base, f"{base}_variance")
)
RTRRL_TRACE_METRICS = tuple(
    name
    for base in (
        *(
            f"train/episode/update.torso.trace_norm.{place}"
            for place in ("before", "recurrence", "after")
        ),
        "train/episode/update.actor.trace_norm",
        "train/episode/update.critic.trace_norm",
    )
    for name in (base, f"{base}_variance")
)


def stream_ac_parameters() -> dict:
    return expand(
        STREAM_AC_PARAMETERS,
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
            "backbone.rtu.differentiation.kind": "exact_rtrl",
            "meta_rl": False,
            "gamma": 0.9,
            "trace_lambda": 0.8,
            "entropy_coefficient": 0.01,
        },
    )


def rtrrl_parameters() -> dict:
    return expand(
        RTRRL_PARAMETERS,
        {
            "torso.backbone.kind": "lru",
            "torso.backbone.lru.feature_dim": 4,
            "torso.backbone.lru.hidden_dim": 2,
            "torso.backbone.lru.differentiation.kind": "exact_rtrl",
            "torso.optimizer.kind": "adam",
            "torso.optimizer.adam.lr": 1e-3,
            "torso.grad_clip": 1.0,
            "torso.follow": 0.25,
            "heads.optimizer.kind": "adam",
            "heads.optimizer.adam.lr": 1e-3,
            "actor.head.kind": "state_std",
            "critic.head.kind": "value",
            "normalization.observation.kind": "none",
            "normalization.reward.kind": "none",
            "meta_rl": False,
            "gamma": 0.9,
            "lambda_pi": 0.8,
            "lambda_v": 0.7,
            "lambda_rnn": 0.6,
            "eta_pi": 0.5,
            "eta_f": 0.5,
            "entropy_rate": 1e-3,
        },
    )


@pytest.mark.parametrize(
    ("entry", "parameters", "trace_metrics"),
    (
        ("stream_ac", stream_ac_parameters, STREAM_AC_TRACE_METRICS),
        ("rtrrl", rtrrl_parameters, RTRRL_TRACE_METRICS),
    ),
    ids=("stream_ac", "rtrrl"),
)
def test_worker_runs_algorithm_and_publishes_trace_norms(
    s3_base: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
    parameters,
    trace_metrics: tuple[str, ...],
) -> None:
    catalog_path = tmp_path / "catalog.json"
    write_catalog(catalog_path)
    monkeypatch.setenv("TRAINER_CATALOG", str(catalog_path))
    monkeypatch.setenv(
        "PATH", f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"
    )

    aim_path = tmp_path / "aim"
    Repo.from_path(str(aim_path), init=True)
    artifact_root = f"{s3_base}/{entry}/artifacts"
    config = RunSpec.model_validate(
        {
            "contract": CONTRACT_VERSION,
            "identity": {
                "run_id": f"{entry}-worker-smoke-t0-s0",
                "experiment": f"{entry}-worker-smoke",
                "launch_id": "20260812-000000",
                "trial": 0,
                "seed": 0,
                "role": "tuning",
                "digest": "local@sha256:" + "a" * 64,
            },
            "entry": entry,
            "artifacts": {"root": artifact_root},
            "algorithm": {
                "environment": {
                    "id": "brax::hopper",
                    "backend": "spring",
                    "episode_length": 4,
                    "observed": [0, 2, 4],
                },
                "num_envs": 1,
                "parameters": parameters(),
            },
            "training": {"seed": 0, "total_steps": 4, "chunk_steps": 4},
            "evaluation": {
                "every_steps": 4,
                # This smoke is about the worker boundary, not the measuring:
                # four steps of Hopper end nothing, so asking for an episode
                # would be asking for one that cannot arrive.
                "episodes": 0,
                "chunk_steps": 4,
                "seed": 1000,
            },
            "logging": {
                "aim": {
                    "url": str(aim_path),
                    "training": {"episode": {"every_episodes": 1}},
                }
            },
        }
    )
    config_uri = f"{s3_base}/{entry}/config.json"
    manifest_uri = f"{s3_base}/{entry}/manifest.json"
    objects.put_bytes(config_uri, config.model_dump_json().encode())
    objects.put_bytes(manifest_uri, json.dumps({"runs": [config_uri]}).encode())

    run_manifest(manifest_uri, tmp_path / "worker")

    records = [
        json.loads(line)
        for line in objects.get_bytes(f"{artifact_root}/metrics.jsonl")
        .decode()
        .splitlines()
    ]
    values = {
        metric: [row["metrics"][metric] for row in records if metric in row["metrics"]]
        for metric in trace_metrics
    }
    assert all(values.values())
    assert all(math.isfinite(value) for series in values.values() for value in series)

    result = json.loads(objects.get_bytes(f"{artifact_root}/result.json"))
    assert result["identity"]["run_id"] == config.identity.run_id
    assert result["success"] is True
    assert "metrics.jsonl" in result["artifacts"]

    repo = Repo.from_path(str(aim_path))
    runs = list(repo.iter_runs())
    assert len(runs) == 1
    metrics = {metric.name: metric for metric in runs[0].metrics()}
    assert set(trace_metrics) <= set(metrics)
    for metric in trace_metrics:
        aim_values = metrics[metric].data.values_list()[0]
        assert aim_values
        assert all(math.isfinite(value) for value in aim_values)
        assert values[metric][-1] == aim_values[-1]
