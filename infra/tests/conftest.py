"""One experiment file and one catalog, shaped the way the worker's are.

Written once because the point of the shape is that both sides hold the same
one; a copy per test file is how they would stop.

The catalog declares a second entry, ``blocks``, for the tests about a setting
held equal across an agent's three learners. It is the same shape as the first
-- a tree with a ``kind`` beside its branches -- at the size the question needs:
three blocks that each declare a whole optimizer, and one of the torso's
branches that is itself a pair.
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
        "contract": 11,
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
            },
            "blocks": {
                "command": ["python", "-m", "entries.blocks"],
                "metrics": ["eval/episode/return_per_step"],
                "parameters": _blocks(),
            },
            # The same entry through the parallel channel, so a round's runs can
            # be asked for as groups without changing anything they carry.
            "blocks_ensemble": {
                "command": ["python", "-m", "entries.blocks_ensemble"],
                "metrics": ["eval/episode/return_per_step"],
                "parameters": _blocks(),
                "grouped": True,
            },
        },
    }


@pytest.fixture
def experiment() -> dict[str, Any]:
    return copy.deepcopy(EXPERIMENT)


@pytest.fixture
def blocks(experiment: dict[str, Any]) -> dict[str, Any]:
    """The same experiment, against the entry that has three learners in it."""

    experiment["entry"] = "blocks"
    experiment["space"] = copy.deepcopy(BLOCKS_SPACE)
    return experiment


EXPERIMENT: dict[str, Any] = {
    "experiment": "streamac-test",
    "name": "stream-ac-test",
    "image": IMAGE,
    "entry": "stream_ac",
    "storage": "s3://artifacts/trainer",
    "environment": {
        "id": "brax::hopper",
        "backend": "spring",
        "seeds": [0],
        "episode_length": 1000,
        "observed": [0, 2, 4],
    },
    "training": {"num_envs": 4, "total_steps": 200, "chunk_steps": 100},
    "evaluation": {
        "every_steps": 100,
        "episodes": 2,
        "chunk_steps": 16,
        "seed": 1000,
    },
    "logging": {
        "aim": {
            "url": "aim://aim:53800",
            "training": {"window": {"every_steps": 100}},
        },
        "rerun": {"log_every_steps": 100},
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
  seeds: [0]
  episode_length: 1000
  observed: [0, 2, 4]

training:
  num_envs: 4
  total_steps: 200
  chunk_steps: 100

evaluation:
  every_steps: 100
  episodes: 2
  chunk_steps: 16
  seed: 1000

logging:
  aim:
    url: aim://aim:53800
    training:
      window:
        every_steps: 100
  rerun:
    log_every_steps: 100

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


# --------------------------------------------------- an agent with three blocks
FLOAT = {"type": "float", "low": 0.0, "high": 1.0}
POSITIVE = {"type": "float", "low": 1e-12, "high": 1e-2, "log": True}
RATE = {"type": "float", "low": 1e-9, "high": 10.0, "log": True}


def _parameter(valid: dict[str, Any], search: dict[str, Any]) -> dict[str, Any]:
    return {"valid": valid, "search": search}


def _structure(values: list[str]) -> dict[str, Any]:
    """A choice of component, which selects a branch and so is known at build.

    An image marks every categorical this way -- what a component does with one
    is pick -- and the parallel channel reads the mark to decide which runs can
    share a graph.
    """

    domain = {"type": "choice", "values": values}
    return {"valid": domain, "search": domain, "static": True}


def _adam() -> dict[str, Any]:
    return {
        "lr": _parameter(RATE, {"type": "float", "low": 1e-5, "high": 1e-2, "log": True}),
        "b1": _parameter(FLOAT, {"type": "choice", "values": [0.0, 0.9]}),
        "b2": _parameter(FLOAT, {"type": "choice", "values": [0.9, 0.99, 0.999, 0.9999]}),
        "eps": _parameter(POSITIVE, {"type": "choice", "values": [1e-8]}),
    }


def _iu() -> dict[str, Any]:
    """The intentional update's settings: one step size, and the rest shared-able."""

    return {
        "eta": _parameter({"type": "float", "low": 0.0, "high": 10.0}, FLOAT),
        "clip": _parameter(
            {"type": "float", "low": 0.0, "high": 1000.0},
            {"type": "choice", "values": [10.0, 20.0]},
        ),
        "beta_rms": _parameter(FLOAT, {"type": "choice", "values": [0.99, 0.999]}),
    }


def _head_optimizer() -> dict[str, Any]:
    """What a readout may step under. One derivative, so no position to name."""

    return {
        "kind": _structure(["adam", "sgd", "iu"]),
        "adam": _adam(),
        "sgd": {"lr": _parameter(RATE, {"type": "float", "low": 1e-5, "high": 1.0, "log": True})},
        "iu": _iu(),
    }


def _torso_optimizer() -> dict[str, Any]:
    """The same, plus the position the two heads' credit is combined at.

    ``output_iu`` is two whole intentional updates rather than one, which is
    what makes the fourth destination of a shared setting a real one.
    """

    return {
        "kind": _structure(["adam", "sgd", "input_iu", "output_iu"]),
        "adam": _adam(),
        "sgd": {"lr": _parameter(RATE, {"type": "float", "low": 1e-5, "high": 1.0, "log": True})},
        "input_iu": _iu(),
        "output_iu": {"actor": _iu(), "critic": _iu()},
    }


def _blocks() -> dict[str, Any]:
    return {
        "gamma": _parameter(FLOAT, {"type": "float", "low": 0.9, "high": 0.99}),
        "actor": {"optimizer": _head_optimizer()},
        "critic": {"optimizer": _head_optimizer()},
        "torso": {"optimizer": _torso_optimizer()},
    }


# Adam at every block, which is the arm the shared-beta cases are written
# against. A case that needs the intentional rules re-pins `kind` itself.
BLOCKS_SPACE: dict[str, Any] = {
    "gamma": [0.99],
    "actor": {"optimizer": {"kind": ["adam"]}},
    "critic": {"optimizer": {"kind": ["adam"]}},
    "torso": {"optimizer": {"kind": ["adam"]}},
}
