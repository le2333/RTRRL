from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import jax
import training_sdk
import yaml

from brax_ppo_acceptance.train import restore_checkpoint


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    artifact_directory = tmp_path / "artifacts"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "protocol_version": "1",
                "environment": {
                    "name": "inverted_pendulum",
                    "options": {"backend": "generalized"},
                },
                "logging": {
                    "aim_every_env_steps": 1,
                    "rerun_every_episodes": 1,
                },
                "parameters": {
                    "runtime": {"seed": 7},
                    "algorithm": {
                        "learning_rate": 0.0003,
                        "num_envs": 4,
                        "episode_length": 32,
                        "failure_mode": "none",
                    },
                },
                "training_budget": {"env_steps": 128},
            }
        ),
        encoding="utf-8",
    )
    context_path = tmp_path / "run-context.json"
    context: dict[str, Any] = {
        "experiment_name": "brax-runtime-cpu",
        "experiment_id": "experiment-runtime",
        "group": "cpu",
        "script": "brax_ppo_acceptance",
        "run_id": "runtime-cpu-1",
        "run_number": 1,
        "trial_number": 1,
        "seed": 7,
        "metadata": {},
        "environment": {
            "name": "inverted_pendulum",
            "options": {"backend": "generalized"},
        },
        "training_budget": {"env_steps": 128},
        "fixed_parameters": {},
        "sampled_parameters": {},
        "final_parameters": {},
        "image_digest": "sha256:runtime-test",
        "resource_profile": "cpu",
        "artifact_directory": str(artifact_directory),
        "logging": {"aim_every_env_steps": 1, "rerun_every_episodes": 1},
        "objective": {"metric": "eval/episode_return"},
    }
    context_path.write_text(json.dumps(context), encoding="utf-8")
    return config_path, context_path, artifact_directory


def test_installed_module_runs_real_cpu_ppo_in_subprocess(tmp_path: Path) -> None:
    assert jax.default_backend() == "cpu"
    assert {device.platform for device in jax.devices()} == {"cpu"}
    config_path, context_path, artifact_directory = _write_inputs(tmp_path)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("BRAX_ACCEPTANCE_TEST_MODE", None)
    environment.pop("BRAX_ACCEPTANCE_E2E_FAST", None)
    environment["JAX_PLATFORM_NAME"] = "cpu"
    environment["TRAINER_RUN_CONTEXT_PATH"] = str(context_path)
    environment["AIM_REPO"] = str(tmp_path / "aim")

    completed = subprocess.run(
        [sys.executable, "-m", "brax_ppo_acceptance", "--config", str(config_path)],
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
    spool = training_sdk.EventSpool(
        artifact_directory / "aim-buffer" / "events.jsonl"
    )
    finalized = [
        event
        for event in spool.events
        if event.kind == "final" and event.data["finalized"]
    ]
    assert len(finalized) == 1
    assert finalized[0].data["objective_metric"] == "eval/episode_return"
    assert math.isfinite(float(finalized[0].metric_value))
    assert len(list((artifact_directory / "rerun").rglob("*.rrd"))) == 1
    checkpoint = artifact_directory / "checkpoints" / "ppo-params.npz"
    assert checkpoint.is_file()
    restored = restore_checkpoint(checkpoint)
    assert len(jax.tree_util.tree_leaves(restored)) > 1
