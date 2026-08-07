"""One experiment file and one catalog, shaped the way the worker's are.

Written once because the point of the shape is that both sides hold the same
one; a copy per test file is how they would stop.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

DIGEST = "sha256:" + "b" * 64
IMAGE = f"registry.example/trainer@{DIGEST}"


@pytest.fixture
def catalog() -> dict[str, Any]:
    return {
        "contract": 7,
        "entries": {
            "stream_ac": {
                "command": ["python", "-m", "entries.stream_ac"],
                "metrics": ["eval/episode/return_per_step"],
                "parameters": {
                    "gamma": {
                        "valid": {"type": "float", "low": 0.0, "high": 1.0},
                        "search": {"type": "float", "low": 0.9, "high": 0.99},
                    },
                    "backbone": {
                        # A component chosen among branches: the choice lives
                        # beside them under the reserved name, so a branch is
                        # only ever read relative to the group that offers it.
                        "kind": {
                            "valid": {"type": "choice", "values": ["rtu", "mlp"]},
                            "search": {"type": "choice", "values": ["rtu", "mlp"]},
                        },
                        "rtu": {
                            "hidden_dim": {
                                "valid": {"type": "int", "low": 1, "high": 4096},
                                "search": {"type": "int", "low": 32, "high": 512},
                            }
                        },
                    },
                },
            }
        },
    }


@pytest.fixture
def experiment() -> dict[str, Any]:
    return copy.deepcopy(EXPERIMENT)


EXPERIMENT: dict[str, Any] = {
    "experiment": "streamac-test",
    "name": "stream-ac-test",
    "image": IMAGE,
    "entry": "stream_ac",
    "storage": "s3://artifacts/trainer",
    "environment": {
        "id": "brax::hopper",
        "backend": "spring",
        "seed": 0,
        "episode_length": 1000,
        "observed": [0, 2, 4],
    },
    "training": {"num_envs": 4, "total_steps": 200, "epoch_steps": 100},
    "evaluation": {"steps": 16},
    "logging": {
        "aim": "aim://aim:53800",
        "enable_rerun": True,
        "rerun_every_steps": 100,
    },
    "score": {
        "metric": "eval/episode/return_per_step",
        "window_steps": [0, 200],
        "reduce": "last",
        "non_finite": "worst",
        "direction": "maximize",
    },
    "hpo": {"rounds": 1, "trials_per_round": 2, "startup_trials": 2, "seed": 7},
    "space": {"gamma": [0.9, 0.95], "backbone": {"kind": ["rtu"], "rtu": {"hidden_dim": [32]}}},
}


EXPERIMENT_YAML = f"""\
experiment: streamac-test
name: stream-ac-test
image: {IMAGE}
entry: stream_ac
storage: s3://artifacts/trainer

environment:
  id: brax::hopper
  backend: spring
  seed: 0
  episode_length: 1000
  observed: [0, 2, 4]

training:
  num_envs: 4
  total_steps: 200
  epoch_steps: 100

evaluation:
  steps: 16

logging:
  aim: aim://aim:53800
  enable_rerun: true
  rerun_every_steps: 100

score:
  metric: eval/episode/return_per_step
  window_steps: [0, 200]
  reduce: last
  non_finite: worst
  direction: maximize

hpo:
  rounds: 2
  trials_per_round: 2
  startup_trials: 2
  seed: 7

space:
  gamma: [0.9, 0.95]
  backbone:
    kind: [rtu]
    rtu:
      hidden_dim: [32]
"""
