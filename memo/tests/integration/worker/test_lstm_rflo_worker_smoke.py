"""Both LSTM-RFLO entries, driven through a real Worker child.

Everything else about this algorithm is checked by building the graph
in-process and stepping it, which says nothing about whether an *entry* runs.
Between the graph and a result there is a manifest, a worker, a run document
validated against the deployment contract, a reporter, an S3 upload and -- for
the grouped entry -- a member axis that has to hand each member its own
artifacts. That path has broken after green algorithm tests before, which is
why issue 65 named it separately and why issue 67 repeats it rather than
assuming a second entry inherits the first one's evidence.

So this is about wiring rather than arithmetic. The budgets are the smallest
that still produce a training episode and an update; what the numbers are is
the unit suite's business.

``rtrrl_lstm_rflo``
    One run, one seed, and the trace readings the algorithm declares arriving
    in both destinations.

``rtrrl_lstm_rflo_ensemble``
    Two seeds in one group, one process, one graph -- and two results, two
    metrics streams and two Aim runs coming back out under their own
    identities. Two members that are the same run would be the seeds having
    been broadcast rather than carried, which is the failure this path is most
    likely to have and least likely to show.
"""

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
from memorax.algorithms.rtrrl_lstm_rflo import PARAMETERS
from memorax.parameters import expand
from worker import objects
from worker.worker import run_manifest

pytestmark = [pytest.mark.integration, pytest.mark.service]

SEEDS = (0, 1)

# The torso's readings, split by position as the algorithm splits them. With no
# normalization behind the cell the sequence is one component, so `before` and
# `after` are empty subtrees and report a norm of zero -- the series still has
# to arrive, because a schema that names it and a run that never files it is
# the mismatch this assertion exists to catch.
TRACE_METRICS = tuple(
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
RETURN_METRIC = "train/episode/return"


def parameters() -> dict:
    """The launch document's settings, at the smallest width that recurs."""

    return expand(
        PARAMETERS,
        {
            "torso.hidden_dim": 2,
            "torso.forget_bias": 1.0,
            "torso.layer_norm": False,
            "torso.differentiation.kind": "rflo",
            "torso.optimizer.kind": "adam",
            "torso.optimizer.adam.lr": 1e-3,
            "torso.grad_clip": 1.0,
            "torso.follow": 1.0,
            "actor.optimizer.kind": "adam",
            "actor.optimizer.adam.lr": 1e-3,
            "actor.head.kind": "categorical",
            "critic.optimizer.kind": "adam",
            "critic.optimizer.adam.lr": 1e-3,
            "critic.head.kind": "value",
            "normalization.observation.kind": "none",
            "normalization.reward.kind": "none",
            "gamma": 0.9,
            "lambda_pi": 0.9,
            "lambda_v": 0.9,
            "lambda_rnn": 0.9,
            "eta_pi": 1.0,
            "eta_f": 1.0,
            "entropy_rate": 1e-3,
            "meta_rl": False,
        },
    )


def specification(*, entry: str, root: str, aim_path: Path, seed: int) -> RunSpec:
    """One run document, as the control plane would write it."""

    return RunSpec.model_validate(
        {
            "contract": CONTRACT_VERSION,
            "identity": {
                "run_id": f"{entry}-smoke-t0-s{seed}",
                "experiment": f"{entry}-smoke",
                "launch_id": "20260827-000000",
                "trial": 0,
                "seed": seed,
                "role": "tuning",
                "digest": "local@sha256:" + "a" * 64,
            },
            "entry": entry,
            "artifacts": {"root": f"{root}/seed-{seed}"},
            "algorithm": {
                "environment": {
                    "id": "gymnax::CartPole-v1",
                    "backend": None,
                    # Long enough that the pole falls before the limit does.
                    # CartPole pays 1.0 a step, so a return is an episode's
                    # length; truncating every episode at a fixed number would
                    # make the metric that number for any policy at all.
                    "episode_length": 32,
                    "observed": None,
                },
                "num_envs": 1,
                "parameters": parameters(),
            },
            "training": {"seed": seed, "total_steps": 64, "chunk_steps": 32},
            "evaluation": {
                # The worker boundary is what this exercises. Measuring would
                # add a rollout whose length the policy decides, and a smoke
                # that waited on one would be testing that instead.
                "every_steps": 32,
                "episodes": 0,
                "chunk_steps": 32,
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


def published(uri: str, specs: list[RunSpec], *, grouped: bool) -> str:
    """Write the run documents and the manifest that names them."""

    written = []
    for spec in specs:
        config = f"{uri}/config-s{spec.identity.seed}.json"
        objects.put_bytes(config, spec.model_dump_json().encode())
        written.append(config)
    manifest = f"{uri}/manifest.json"
    body = {"groups": [written]} if grouped else {"runs": written}
    objects.put_bytes(manifest, json.dumps(body).encode())
    return manifest


def series(prefix: str, metric: str) -> list[float]:
    records = [
        json.loads(line)
        for line in objects.get_bytes(f"{prefix}/metrics.jsonl").decode().splitlines()
    ]
    return [row["metrics"][metric] for row in records if metric in row["metrics"]]


@pytest.fixture
def image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A catalog built from this checkout, and a worker that can spawn python."""

    catalog_path = tmp_path / "catalog.json"
    write_catalog(catalog_path)
    monkeypatch.setenv("TRAINER_CATALOG", str(catalog_path))
    monkeypatch.setenv(
        "PATH", f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"
    )


def test_the_worker_runs_the_entry_and_publishes_what_it_declares(
    image: None, s3_base: str, tmp_path: Path
) -> None:
    """A minimal training run of ``rtrrl_lstm_rflo``, end to end.

    Every reading the algorithm's schema names has to arrive in the metrics
    object and in Aim, and the result has to say the run succeeded under its
    own id. Nothing here is about the numbers; it is about there being any.
    """

    aim_path = tmp_path / "aim"
    Repo.from_path(str(aim_path), init=True)
    root = f"{s3_base}/single"
    spec = specification(entry="rtrrl_lstm_rflo", root=root, aim_path=aim_path, seed=0)

    run_manifest(published(root, [spec], grouped=False), tmp_path / "worker")

    prefix = spec.artifacts.root
    result = json.loads(objects.get_bytes(f"{prefix}/result.json"))
    assert result["identity"]["run_id"] == spec.identity.run_id
    assert result["success"] is True
    assert "metrics.jsonl" in result["artifacts"]

    values = {metric: series(prefix, metric) for metric in TRACE_METRICS}
    assert all(
        values.values()
    ), f"nothing filed {sorted(k for k, v in values.items() if not v)}"
    assert all(math.isfinite(value) for one in values.values() for value in one)

    runs = list(Repo.from_path(str(aim_path)).iter_runs())
    assert len(runs) == 1
    assert set(TRACE_METRICS) <= {metric.name for metric in runs[0].metrics()}


def test_the_worker_runs_two_seeds_as_one_group_and_reports_each_alone(
    image: None, s3_base: str, tmp_path: Path
) -> None:
    """Two seeds through ``rtrrl_lstm_rflo_ensemble``, one graph, two results.

    The member axis is what this is for. Two members that came back identical
    would be one seed broadcast across the map -- two complete, plausible,
    indistinguishable runs, which is the failure a grouped entry is most likely
    to have and least likely to show. CartPole's early episodes are the key's
    own random walk, so two seeds cannot agree on how long the pole stayed up.
    """

    aim_path = tmp_path / "aim"
    Repo.from_path(str(aim_path), init=True)
    root = f"{s3_base}/group"
    specs = [
        specification(
            entry="rtrrl_lstm_rflo_ensemble", root=root, aim_path=aim_path, seed=seed
        )
        for seed in SEEDS
    ]

    run_manifest(published(root, specs, grouped=True), tmp_path / "worker")

    returns = {}
    for spec in specs:
        prefix = spec.artifacts.root
        result = json.loads(objects.get_bytes(f"{prefix}/result.json"))
        assert result["identity"]["run_id"] == spec.identity.run_id
        assert result["identity"]["seed"] == spec.identity.seed
        assert result["success"] is True
        assert "metrics.jsonl" in result["artifacts"]

        traced = series(prefix, TRACE_METRICS[0])
        assert traced and all(math.isfinite(value) for value in traced)
        returns[spec.identity.seed] = series(prefix, RETURN_METRIC)
        assert returns[spec.identity.seed]

    # Each member is its own run in Aim as well, not one reported twice.
    runs = list(Repo.from_path(str(aim_path)).iter_runs())
    assert len(runs) == len(specs)

    assert returns[SEEDS[0]] != returns[SEEDS[1]], (
        "the two members produced identical episodes, so the seeds did not "
        "reach them"
    )
