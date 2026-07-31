from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import jax
from aim import Repo
from training_sdk.reporter import METRICS_FILENAME

from brax_ppo_acceptance.train import restore_checkpoint


def test_installed_module_runs_real_cpu_ppo_in_subprocess(tmp_path: Path) -> None:
    assert jax.default_backend() == "cpu"
    assert {device.platform for device in jax.devices()} == {"cpu"}

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    aim_repo = tmp_path / "aim"
    Repo.from_path(str(aim_repo), init=True)
    config_path = tmp_path / "run-config.json"
    config_path.write_text(
        json.dumps(
            {
                "contract": 4,
                "run_id": "runtime-cpu-1",
                "experiment": "brax-runtime-cpu",
                "name": "runtime",
                "launch_id": "20260725-000000",
                "trial": 0,
                "entry": "brax_ppo_acceptance",
                "digest": "registry.example/trainer@sha256:" + "a" * 64,
                "environment": {
                    "id": "brax::inverted_pendulum",
                    "backend": "generalized",
                    "num_envs": 4,
                },
                "budget": {
                    "total_steps": 128,
                    "epoch_steps": 128,
                    "eval_steps": 0,
                },
                "params": {
                    "env": "inverted_pendulum",
                    "backend": "generalized",
                    "total_steps": 128,
                    "seed": 7,
                    "learning_rate": 0.0003,
                    "num_envs": 4,
                    "episode_length": 32,
                    "failure_mode": "none",
                },
                "logging": {"aim": str(aim_repo), "every_steps": 1},
                "score": {
                    "metric": "episode_return",
                    "window_steps": [0, 128],
                    "reduce": "mean",
                    "direction": "maximize",
                    "non_finite": "worst",
                    "s3": "s3://bucket/score.json",
                },
            }
        ),
        encoding="utf-8",
    )

    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("BRAX_ACCEPTANCE_TEST_MODE", None)
    environment.pop("BRAX_ACCEPTANCE_E2E_FAST", None)
    environment["JAX_PLATFORM_NAME"] = "cpu"
    environment["TRAINER_RUN_CONFIG"] = str(config_path)
    environment["TRAINER_SCRATCH"] = str(scratch)

    completed = subprocess.run(
        [sys.executable, "-m", "brax_ppo_acceptance"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert '"platform": "cpu"' in completed.stdout
    assert '"device_platforms": ["cpu"]' in completed.stdout

    metrics_path = scratch / METRICS_FILENAME
    assert metrics_path.is_file()
    steps = [
        json.loads(line)["step"]
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert max(steps) == 128
    final_metric = json.loads(metrics_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert math.isfinite(float(final_metric["metrics"]["episode_return"]))

    checkpoint = scratch / "ppo-params.npz"
    assert checkpoint.is_file()
    restored = restore_checkpoint(checkpoint)
    assert len(jax.tree_util.tree_leaves(restored)) > 1
