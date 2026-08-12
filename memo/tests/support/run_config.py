"""Version-7 run documents shared by deployment tests."""

from __future__ import annotations

from worker.contract import CONTRACT_VERSION, RunConfig


def make_run_config(**logging: object) -> RunConfig:
    return RunConfig.model_validate(
        {
            "contract": CONTRACT_VERSION,
            "run_id": "smoke-20260725-000000-t0",
            "experiment": "infra-acceptance",
            "launch_id": "20260725-000000",
            "trial": 0,
            "entry": "e",
            "digest": "registry.example/trainer@sha256:" + "a" * 64,
            "environment": {
                "id": "brax::hopper",
                "backend": "spring",
                "seed": 0,
            },
            "training": {"num_envs": 1, "total_steps": 100, "epoch_steps": 100},
            "evaluation": {"steps": 0},
            "params": {"learning_rate": 0.0003},
            "logging": {"aim": "aim://127.0.0.1:1", **logging},
            "score": {
                "metric": "episode_return",
                "window_steps": [0, 4],
                "reduce": "mean",
                "direction": "maximize",
                "non_finite": "worst",
                "s3": "s3://bucket/score.json",
            },
        }
    )
