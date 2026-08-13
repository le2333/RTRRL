"""Version-8 run documents shared by deployment tests."""

from __future__ import annotations

from deployment.contract import CONTRACT_VERSION
from entries._contract import RunSpec


def make_run_config(**logging: object) -> RunSpec:
    return RunSpec.model_validate(
        {
            "contract": CONTRACT_VERSION,
            "identity": {
                "run_id": "smoke-20260725-000000-t0",
                "experiment": "infra-acceptance",
                "launch_id": "20260725-000000",
                "trial": 0,
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
            "runtime": {
                "seed": 0,
                "total_steps": 100,
                "epoch_steps": 100,
                "evaluation_steps": 0,
            },
            "logging": {"aim": {"url": "aim://127.0.0.1:1"}, **logging},
        }
    )
