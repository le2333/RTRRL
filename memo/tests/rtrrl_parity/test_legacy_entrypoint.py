"""End-to-end compatibility contract for the historical ``rtrrl.py`` CLI."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from memorax.algorithms.rtrrl import entrypoint as compatibility_entrypoint


REPOSITORY_ROOT = Path(__file__).parents[3]
ENTRYPOINT = REPOSITORY_ROOT / "rtrrl" / "rtrrl.py"
EXPECTED_MOCK_EPOCH = {
    "actor_loss": -0.6000000238418579,
    "critic_loss": 20.0,
    "entropy": 0.4000000059604645,
    "lr/rnn": 0.00019999999494757503,
    "lr/td": 0.0010000000474974513,
    "mean_delta": 0.8333333134651184,
    "mean_r_bar": 0.10000000149011612,
    "mean_reward": 4.0,
    "mean_v": 20.0,
    "norms/['params']['weight']": 10.0,
    "norms/['slow_params']['weight']": 13.0,
    "norms/['z']['trace']": 5.0,
    "num_episodes": 1,
    "steps": 30,
    "total_td_loss": 19.399999618530273,
    "v_targ": 20.66666603088379,
}


def _run_entrypoint(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            (
                str(REPOSITORY_ROOT / "memo"),
                str(REPOSITORY_ROOT / "rtrrl"),
            )
        ),
    }
    return subprocess.run(
        (sys.executable, str(ENTRYPOINT), *arguments),
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


@pytest.mark.parametrize(
    ("relative_path", "expected"),
    (
        (
            "rtrrl/config/rtrrl_hop_533.yml",
            {
                "total_timesteps": 10_000_000,
                "num_epochs": 10_000,
                "num_envs": 1,
                "profile": "memo_experimental",
                "logging": "aim",
                "run_name": "RTRRL-HOP-533",
                "td_learning_rate": 3e-5,
                "rnn_learning_rate": 2e-6,
                "rnn_gradient_clip": 1.0,
            },
        ),
        (
            "memo/config/rtrrl_hopper_533.yml",
            {
                "total_timesteps": 1_000_000,
                "num_epochs": 20,
                "num_envs": 1,
                "profile": "memo_experimental",
                "logging": None,
                "run_name": "RTRRL-HOP-533-memorax",
                "td_learning_rate": 3e-5,
                "rnn_learning_rate": 2e-6,
                "rnn_gradient_clip": 1.0,
            },
        ),
    ),
)
def test_subprocess_build_preserves_effective_legacy_fields_without_environment(
    relative_path, expected
):
    result = _run_entrypoint(
        "--config_path",
        relative_path,
        "--compat-action",
        "build",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["environment_started"] is False
    assert payload["jax_imported"] is False
    assert payload["effective"] == expected


def test_subprocess_mock_epoch_matches_pinned_historical_metric_dictionary():
    result = _run_entrypoint("--compat-action", "mock-epoch")

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == EXPECTED_MOCK_EPOCH


def test_mock_epoch_uses_shared_historical_metric_translation(monkeypatch):
    observed = {}

    def translate(summary, **options):
        observed["summary"] = summary
        observed["options"] = options
        return {"delegated": True}

    monkeypatch.setattr(
        compatibility_entrypoint,
        "historical_rtrrl_metrics",
        translate,
    )

    assert compatibility_entrypoint.run_mock_epoch() == {"delegated": True}
    assert observed["options"] == {
        "log_td_lr": True,
        "log_rnn_lr": True,
        "log_norms": True,
    }
    assert observed["summary"].steps == 30


def test_audit_reports_each_migration_class_without_runtime_startup(tmp_path):
    legacy_config = tmp_path / "rtrrl" / "config"
    legacy_config.mkdir(parents=True)
    (legacy_config / "rtrrl_supported.yml").write_text("rnn_model: lru\n")
    (legacy_config / "rtrrl_ctrnn.yml").write_text("rnn_model: ctrnn\n")
    (legacy_config / "rtrrl_null.yml").write_text("rnn_model: null\n")
    (legacy_config / "rtrrl_unknown.yml").write_text("mystery: 1\n")
    (legacy_config / "rtrrl_no_op.yml").write_text("save_model: true\n")

    payload = compatibility_entrypoint.audit_repository_configs(tmp_path)

    assert payload["discovered"] == 5
    assert payload["counts"] == {
        "accepted": 1,
        "unsupported": 2,
        "unknown_fields": 1,
        "deprecated_no_op": 1,
    }
    assert {
        record["path"] for record in payload["files"]["unsupported"]
    } == {
        "rtrrl/config/rtrrl_ctrnn.yml",
        "rtrrl/config/rtrrl_null.yml",
    }
    assert payload["files"]["unknown_fields"][0]["path"].endswith(
        "rtrrl_unknown.yml"
    )
    warning = payload["files"]["deprecated_no_op"][0]["warnings"][0]
    assert warning["path"] == "save_model"


def test_subprocess_audit_classifies_every_repository_rtrrl_yaml():
    result = _run_entrypoint("--compat-action", "audit")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["discovered"] == 697
    assert sum(payload["counts"].values()) == payload["discovered"]
    assert payload["counts"] == {
        "accepted": 697,
        "unsupported": 0,
        "unknown_fields": 0,
        "deprecated_no_op": 0,
    }
    assert any(
        record["path"]
        == "memo/config/independent_rtrrl_hopper_maskP_lru.yml"
        for record in payload["files"]["accepted"]
    )
    assert payload["expected_plan_count"] == 686
    assert payload["count_delta"] == 11


def test_legacy_entrypoint_contains_no_training_mathematics_or_oracle_import():
    source = ENTRYPOINT.read_text()
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert "jax" not in imported_modules
    assert "jax.numpy" not in imported_modules
    assert "optax" not in imported_modules
    assert "distrax" not in imported_modules
    assert not any("oracle" in module for module in imported_modules)
    assert not {
        "TD",
        "RNNActorCritic",
        "step_fn",
        "trace_updates",
        "eval_model",
    } & {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
