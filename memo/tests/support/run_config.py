"""Current-contract run documents shared by deployment tests."""

from __future__ import annotations

from deployment.contract import CONTRACT_VERSION
from entries._contract import RunSpec


def make_run_config(**logging: object) -> RunSpec:
    return RunSpec.model_validate(
        {
            "contract": CONTRACT_VERSION,
            "identity": {
                "run_id": "smoke-20260725-000000-t0-s0",
                "experiment": "infra-acceptance",
                "launch_id": "20260725-000000",
                "trial": 0,
                "seed": 0,
                "role": "tuning",
                "digest": "registry.example/trainer@sha256:" + "a" * 64,
            },
            "entry": "e",
            "artifacts": {"root": "s3://bucket/runs/smoke-t0"},
            "algorithm": {
                "environment": {
                    "id": "brax::hopper",
                    "backend": "spring",
                    "episode_length": 1000,
                },
                "num_envs": 1,
                "parameters": {"learning_rate": 0.0003},
            },
            "training": {
                "seed": 0,
                "total_steps": 100,
                "chunk_steps": 100,
            },
            "evaluation": {
                "every_steps": 100,
                "episodes": 0,
                "chunk_steps": 100,
                "seed": 7,
            },
            "logging": {"aim": {"url": "aim://127.0.0.1:1"}, **logging},
        }
    )
