from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
import training_sdk
import yaml

import brax_ppo_acceptance.__main__ as launcher
import brax_ppo_acceptance.train as train_module
from brax_ppo_acceptance.train import restore_checkpoint


class RecordingAim:
    def __init__(self) -> None:
        self.started = 0
        self.events: list[training_sdk.MetricEvent] = []
        self.failures: list[dict[str, str]] = []

    def start(self, context: training_sdk.RunContext) -> None:
        del context
        self.started += 1

    def send(self, event: training_sdk.MetricEvent) -> None:
        self.events.append(event)

    def fail(self, metadata: dict[str, str]) -> None:
        self.failures.append(metadata)

    def close(self) -> None:
        return None


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


def test_real_cpu_ppo_params_round_trip_and_runtime_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert jax.default_backend() == "cpu"
    assert {device.platform for device in jax.devices()} == {"cpu"}
    config_path, context_path, artifact_directory = _write_inputs(tmp_path)
    aim = RecordingAim()
    captured: dict[str, Any] = {}
    real_ppo_train = train_module.ppo_train.train

    def capturing_ppo_train(**kwargs: Any) -> tuple[Any, Any, dict[str, Any]]:
        result = real_ppo_train(**kwargs)
        captured["params"] = result[1]
        return result

    def bootstrap() -> training_sdk.TrainingRun | None:
        return training_sdk.bootstrap_from_environment(
            {"TRAINER_RUN_CONTEXT_PATH": str(context_path)},
            aim_factory=lambda context, environ: aim,
        )

    monkeypatch.delenv("BRAX_ACCEPTANCE_TEST_MODE", raising=False)
    monkeypatch.delenv("BRAX_ACCEPTANCE_E2E_FAST", raising=False)
    monkeypatch.setattr(train_module.ppo_train, "train", capturing_ppo_train)
    monkeypatch.setattr(launcher, "bootstrap_from_environment", bootstrap)
    training_sdk.set_current_run(None)
    try:
        assert launcher.main(["--config", str(config_path)]) == 0
    finally:
        training_sdk.set_current_run(None)

    assert aim.started == 1
    assert aim.failures == []
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

    original = captured["params"]
    restored = restore_checkpoint(checkpoint)
    assert jax.tree_util.tree_structure(restored) == jax.tree_util.tree_structure(
        original
    )
    original_leaves = jax.tree_util.tree_leaves(original)
    restored_leaves = jax.tree_util.tree_leaves(restored)
    assert len(restored_leaves) == len(original_leaves)
    for expected, actual in zip(original_leaves, restored_leaves, strict=True):
        expected_array = np.asarray(jax.device_get(expected))
        actual_array = np.asarray(jax.device_get(actual))
        assert actual_array.dtype == expected_array.dtype
        assert actual_array.shape == expected_array.shape
        np.testing.assert_array_equal(actual_array, expected_array)

    with np.load(checkpoint, allow_pickle=False) as archive:
        assert all(archive[name].dtype != np.dtype("O") for name in archive.files)
