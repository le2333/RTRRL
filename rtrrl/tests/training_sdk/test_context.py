import json
from dataclasses import replace

import pytest

from training_sdk import RunContext, current_run


def write_context(tmp_path, **overrides):
    payload = {
        "experiment_name": "default",
        "experiment_id": "exp-123",
        "group": "single",
        "script": "rtrrl.py",
        "run_id": "run-456",
        "run_number": 1,
        "trial_number": 2,
        "seed": 3,
        "metadata": {"algorithm": "rtrrl", "labels": ["baseline"]},
        "environment": {"name": "hopper", "options": {"difficulty": 2}},
        "training_budget": {"env_steps": 1_000},
        "fixed_parameters": {"optimizer": {"name": "adam"}},
        "sampled_parameters": {"learning_rate": 0.001},
        "final_parameters": {"learning_rate": 0.001},
        "image_digest": "sha256:abc",
        "resource_profile": "cpu-small",
        "artifact_directory": str(tmp_path / "artifacts"),
    }
    payload.update(overrides)
    path = tmp_path / "run-context.json"
    path.write_text(json.dumps(payload))
    return path


def test_context_preserves_user_experiment_and_structured_identity(tmp_path):
    path = write_context(
        tmp_path, experiment_name="hopper", group="dual", run_number=12
    )

    context = RunContext.from_path(path)

    assert context.experiment_name == "hopper"
    assert context.run_name == "dual-0012"
    assert context.hparams["identity"]["group"] == "dual"
    assert context.hparams["identity"]["run_number"] == 12
    assert context.artifact_directory == tmp_path / "artifacts"


def test_context_nested_data_cannot_drift_after_construction(tmp_path):
    metadata = {"labels": ["original"], "nested": {"enabled": True}}
    loaded = RunContext.from_path(write_context(tmp_path))
    context = replace(loaded, metadata=metadata)
    metadata["labels"].append("changed")

    with pytest.raises(TypeError):
        context.metadata["nested"]["enabled"] = False

    assert context.hparams["metadata"] == {
        "labels": ["original"],
        "nested": {"enabled": True},
    }


def test_hparams_returns_independent_plain_json_data(tmp_path):
    context = RunContext.from_path(write_context(tmp_path))

    hparams = context.hparams
    hparams["metadata"]["labels"].append("changed")
    hparams["environment"]["options"]["difficulty"] = 99

    json.dumps(hparams)
    assert type(hparams) is dict
    assert type(hparams["metadata"]) is dict
    assert type(hparams["metadata"]["labels"]) is list
    assert context.hparams["metadata"]["labels"] == ["baseline"]
    assert context.hparams["environment"]["options"]["difficulty"] == 2


def test_current_run_fails_clearly_when_not_initialized():
    with pytest.raises(RuntimeError, match="initialized"):
        current_run()
