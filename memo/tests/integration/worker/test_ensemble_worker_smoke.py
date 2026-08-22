"""A real Worker child runs a group and publishes each member as its own run.

The wiring is the thing under test, not the arithmetic: a manifest carrying a
group, the worker handing it to one process, an ensemble entry building one
graph, and two members coming back out with separate artifacts, separate
metrics and separate results under their own identities.
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
from memorax.algorithms.drqn import PARAMETERS as DRQN_PARAMETERS
from memorax.parameters import expand
from worker import objects
from worker.worker import WorkerError, run_manifest

pytestmark = [pytest.mark.integration, pytest.mark.service]

SEEDS = (0, 1)
METRIC = "train/episode/return"


def drqn_parameters(lr: float = 0.1, hidden: int = 2) -> dict:
    return expand(
        DRQN_PARAMETERS,
        {
            "core.kind": "lru",
            "core.lru.hidden_dim": hidden,
            "core.lru.feature_dim": 2,
            "learning.kind": "truncated",
            "learning.truncated.length": 2,
            "optimizer.kind": "adadelta",
            "optimizer.adadelta.lr": lr,
            "grad_clip": 10.0,
            "replay.capacity": 64,
            "replay.minimum_size": 4,
            "replay.batch_size": 2,
            "target.update_period": 4,
            "exploration.epsilon_start": 1.0,
            "exploration.epsilon_end": 0.05,
            "exploration.epsilon_decay_steps": 64,
            "exploration.evaluation_epsilon": 0.0,
            "gamma": 0.9,
        },
    )


def member(*, root: str, aim_path: Path, seed: int, parameters: dict) -> RunSpec:
    return RunSpec.model_validate(
        {
            "contract": CONTRACT_VERSION,
            "identity": {
                "run_id": f"drqn-group-smoke-t0-s{seed}",
                "experiment": "drqn-group-smoke",
                "launch_id": "20260822-000000",
                "trial": 0,
                "seed": seed,
                "role": "tuning",
                "digest": "local@sha256:" + "a" * 64,
            },
            "entry": "drqn_ensemble",
            "artifacts": {"root": f"{root}/seed-{seed}"},
            "algorithm": {
                "environment": {
                    "id": "gymnax::CartPole-v1",
                    "backend": None,
                    # Long enough that the pole drops before the limit does.
                    # CartPole pays 1.0 every step, so a return is an episode's
                    # length and nothing else: truncate every episode at a fixed
                    # number and the metric is that number for any policy at
                    # all, which is a metric that cannot tell two members apart.
                    "episode_length": 32,
                    "observed": None,
                },
                "num_envs": 1,
                "parameters": parameters,
            },
            "training": {"seed": seed, "total_steps": 64, "chunk_steps": 32},
            "evaluation": {
                # The worker boundary is what this exercises. Measuring would
                # add a rollout whose length is the policy's to decide, and a
                # smoke that waited on one would be testing that instead.
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


def submit(s3_base: str, tmp_path: Path, specs: list[RunSpec]) -> str:
    uris = []
    for spec in specs:
        uri = f"{s3_base}/group/config-s{spec.identity.seed}.json"
        objects.put_bytes(uri, spec.model_dump_json().encode())
        uris.append(uri)
    manifest_uri = f"{s3_base}/group/manifest.json"
    objects.put_bytes(manifest_uri, json.dumps({"groups": [uris]}).encode())
    return manifest_uri


@pytest.fixture
def image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    catalog_path = tmp_path / "catalog.json"
    write_catalog(catalog_path)
    monkeypatch.setenv("TRAINER_CATALOG", str(catalog_path))
    monkeypatch.setenv(
        "PATH", f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"
    )


def test_worker_runs_a_group_and_publishes_every_member(
    image: None, s3_base: str, tmp_path: Path
) -> None:
    aim_path = tmp_path / "aim"
    Repo.from_path(str(aim_path), init=True)
    root = f"{s3_base}/group"
    parameters = drqn_parameters()
    specs = [
        member(root=root, aim_path=aim_path, seed=seed, parameters=parameters)
        for seed in SEEDS
    ]

    run_manifest(submit(s3_base, tmp_path, specs), tmp_path / "worker")

    series = {}
    for spec in specs:
        prefix = spec.artifacts.root
        result = json.loads(objects.get_bytes(f"{prefix}/result.json"))
        assert result["identity"]["run_id"] == spec.identity.run_id
        assert result["identity"]["seed"] == spec.identity.seed
        assert result["success"] is True
        assert "metrics.jsonl" in result["artifacts"]

        records = [
            json.loads(line)
            for line in objects.get_bytes(f"{prefix}/metrics.jsonl")
            .decode()
            .splitlines()
        ]
        values = [
            row["metrics"][METRIC] for row in records if METRIC in row["metrics"]
        ]
        assert values
        assert all(math.isfinite(value) for value in values)
        series[spec.identity.seed] = values

    # Every member is its own run in Aim too, not one run reported twice.
    runs = list(Repo.from_path(str(aim_path)).iter_runs())
    assert len(runs) == len(SEEDS)

    # And the seeds reached the members. A vmap that broadcast one member's key
    # would publish two complete, plausible, identical runs -- the failure this
    # whole path is most likely to have and least likely to show.
    #
    # Exploration starts at epsilon 1.0, so early episodes are the key's own
    # random walk and two seeds cannot agree on how long the pole stayed up.
    assert series[SEEDS[0]] != series[SEEDS[1]]


def test_worker_runs_a_group_that_sweeps_a_value(
    image: None, s3_base: str, tmp_path: Path
) -> None:
    """Members may differ in a leaf the graph reads arithmetically.

    Which is the whole point of a sweep, and the rule that changed once the
    algorithms stopped coercing those leaves. Under seeds alone this was a
    refusal: the graph came from one member and a difference could only have
    been dropped without saying so.

    The wiring is what is checked here -- that a differing learning rate
    survives the manifest, the worker, the entry's agreement check and the
    member axis, and that both members still publish separately. That the value
    is what separates them is isolated in the unit tests, which can hold the
    seeds still and afford to run twice.
    """

    aim_path = tmp_path / "aim"
    Repo.from_path(str(aim_path), init=True)
    root = f"{s3_base}/swept"
    specs = [
        member(
            root=root,
            aim_path=aim_path,
            seed=seed,
            parameters=drqn_parameters(lr=lr),
        )
        for seed, lr in zip(SEEDS, (0.1, 0.02))
    ]

    run_manifest(submit(s3_base, tmp_path, specs), tmp_path / "worker")

    for spec in specs:
        result = json.loads(
            objects.get_bytes(f"{spec.artifacts.root}/result.json")
        )
        assert result["identity"]["run_id"] == spec.identity.run_id
        assert result["success"] is True
    assert len(list(Repo.from_path(str(aim_path)).iter_runs())) == len(SEEDS)


def test_a_group_that_varies_a_static_parameter_is_refused(
    image: None, s3_base: str, tmp_path: Path
) -> None:
    """A width sizes an array, and the members of one map share their shapes.

    The refusal is what keeps the relaxation from becoming a trap: `hidden_dim`
    is a `param` exactly as `lr` is, and only its declaration separates them.
    Without the check this would fail somewhere inside a trace, on an instance
    already paid for.
    """

    aim_path = tmp_path / "aim"
    Repo.from_path(str(aim_path), init=True)
    root = f"{s3_base}/mixed"
    specs = [
        member(
            root=root,
            aim_path=aim_path,
            seed=seed,
            parameters=drqn_parameters(hidden=hidden),
        )
        for seed, hidden in zip(SEEDS, (2, 4))
    ]

    with pytest.raises(WorkerError, match="exited with exit code"):
        run_manifest(submit(s3_base, tmp_path, specs), tmp_path / "worker")
