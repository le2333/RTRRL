"""End-to-end compatibility contract for the historical ``rtrrl.py`` CLI."""

from __future__ import annotations

import ast
from dataclasses import fields
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from memorax.algorithms.rtrrl import entrypoint as compatibility_entrypoint


REPOSITORY_ROOT = Path(__file__).parents[3]
ENTRYPOINT = REPOSITORY_ROOT / "rtrrl" / "rtrrl.py"
MOCK_EPOCH_FIXTURE = (
    Path(__file__).parent / "golden" / "historical_mock_epoch_v1.json"
)
HISTORICAL_RTRRL_PARAM_FIELDS = (
    "debug",
    "seed",
    "episodes",
    "steps",
    "patience",
    "eval_every",
    "eval_steps",
    "eval_batch_size",
    "render_every_evals",
    "render_start",
    "render_steps",
    "logging",
    "log_repo",
    "run_name",
    "save_model",
    "log_norms",
    "log_code",
    "log_every",
    "env_params",
    "optimizer_params_td",
    "optimizer_params_rnn",
    "rnn_model",
    "gradient_mode",
    "hidden_size",
    "wiring",
    "trace_mode",
    "gamma",
    "lambda_v",
    "lambda_pi",
    "lambda_rnn",
    "eta_pi",
    "eta_f",
    "entropy_rate",
    "eta",
    "meta_rl",
    "f_align",
    "normalize_reward",
    "normalize_obs",
    "var_scaling",
    "layer_norm",
    "mlp_actor",
    "pass_obs",
    "align_action_logprob",
    "update_trace_before_td",
    "update_period",
    "dropout_rate",
    "act_magnitude_factor",
    "slow_rnn_factor",
)


def _run_entrypoint(
    *arguments: str,
    cwd: Path = REPOSITORY_ROOT,
) -> subprocess.CompletedProcess[str]:
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
        cwd=cwd,
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
                "environment": {
                    "env_name": "brax-hopper",
                    "mode": "F",
                    "backend": "spring",
                },
                "builder": {
                    "function": "build_rtrrl_agent",
                    "topology": "shared",
                    "recurrent_component": "memorax_memoroid_lru",
                        "feature_component": "encoder",
                    "actor_component": "unbounded_gaussian",
                        "meta_rl": True,
                        "normalize_observation": True,
                        "normalize_reward": True,
                        "pass_observation": False,
                },
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
                "environment": {
                    "env_name": "brax-hopper",
                    "mode": "F",
                    "backend": "spring",
                },
                "builder": {
                    "function": "build_rtrrl_agent",
                    "topology": "shared",
                    "recurrent_component": "memorax_memoroid_lru",
                    "feature_component": "encoder",
                    "actor_component": "unbounded_gaussian",
                        "meta_rl": True,
                        "normalize_observation": False,
                        "normalize_reward": False,
                        "pass_observation": False,
                },
            },
        ),
        (
            "memo/config/independent_rtrrl_hopper_maskP_lru.yml",
            {
                "total_timesteps": 2_000_000,
                "num_epochs": 20,
                "num_envs": 1,
                "profile": "memo_experimental",
                "logging": None,
                "run_name": "INDEPENDENT-RTRRL-LRU-hopP",
                "td_learning_rate": 3e-5,
                "rnn_learning_rate": 2e-6,
                "rnn_gradient_clip": 1.0,
                "environment": {
                    "env_name": "brax-hopper",
                    "mode": "P",
                    "backend": "spring",
                },
                "builder": {
                    "function": "build_independent_rtrrl_agent",
                    "topology": "independent",
                    "recurrent_component": "memorax_memoroid_lru",
                    "feature_component": "raw",
                    "actor_component": "unbounded_gaussian",
                        "meta_rl": True,
                        "normalize_observation": False,
                        "normalize_reward": False,
                        "pass_observation": False,
                },
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
    assert "construction" not in payload
    assert payload["effective"] == expected


def test_subprocess_mock_epoch_matches_pinned_historical_metric_dictionary():
    fixture = json.loads(MOCK_EPOCH_FIXTURE.read_text())
    result = _run_entrypoint("--compat-action", "mock-epoch")

    assert result.returncode == 0, result.stderr
    assert fixture["version"] == 1
    assert fixture["source"]["task"] == 9
    assert json.loads(result.stdout) == fixture["metrics"]


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


def test_top_level_environment_maps_to_effective_environment():
    config = compatibility_entrypoint.normalize_legacy_invocation(
        {"env_name": "halfcheetah", "mode": "P", "backend": "generalized"}
    )

    assert config.env_params.env_name == "brax-halfcheetah"
    assert config.mode == "P"
    assert config.backend == "generalized"


def test_conflicting_top_level_and_nested_environment_names_fail():
    with pytest.raises(ValueError, match="conflicting environment names"):
        compatibility_entrypoint.normalize_legacy_invocation(
            {
                "env_name": "hopper",
                "env_params": {"env_name": "brax-halfcheetah"},
            }
        )


def test_equal_top_level_and_nested_environment_names_are_canonicalized():
    config = compatibility_entrypoint.normalize_legacy_invocation(
        {
            "env_name": "hopper",
            "env_params": {"env_name": "brax-hopper"},
        }
    )

    assert config.env_params.env_name == "brax-hopper"


def test_static_build_reports_nested_backend_precedence_used_by_runner():
    config = compatibility_entrypoint.normalize_legacy_invocation(
        {
            "env_name": "hopper",
            "backend": "generalized",
            "env_params": {"init_kwargs": {"backend": "spring"}},
        }
    )

    payload = compatibility_entrypoint.describe_legacy_build(config)

    assert payload["effective"]["environment"]["backend"] == "spring"


@pytest.mark.parametrize(
    ("cwd", "config_path"),
    (
        (REPOSITORY_ROOT, "rtrrl/config/rtrrl_hop_533.yml"),
        (REPOSITORY_ROOT / "rtrrl", "config/rtrrl_hop_533.yml"),
    ),
)
def test_subprocess_cli_overrides_support_historical_forms_and_cwd(
    cwd, config_path
):
    result = _run_entrypoint(
        "--config_path",
        config_path,
        "--compat-action=build",
        "--episodes=4",
        "--steps",
        "25",
        "--optimizer_params_td.learning_rate=0.125",
        "--normalize_obs",
        "--normalize-reward",
        "--no-normalize_reward",
        "--pass_obs",
        "--no-pass-obs",
        cwd=cwd,
    )

    assert result.returncode == 0, result.stderr
    effective = json.loads(result.stdout)["effective"]
    assert effective["total_timesteps"] == 100
    assert effective["num_epochs"] == 4
    assert effective["td_learning_rate"] == 0.125
    assert effective["builder"]["normalize_observation"] is True
    assert effective["builder"]["normalize_reward"] is False
    assert effective["builder"]["pass_observation"] is False


def _load_legacy_entrypoint_module():
    spec = importlib.util.spec_from_file_location(
        "legacy_rtrrl_compatibility",
        ENTRYPOINT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rtrrl_params_restores_historical_mutable_dataclass_contract():
    module = _load_legacy_entrypoint_module()

    assert tuple(item.name for item in fields(module.RTRRLParams)) == (
        HISTORICAL_RTRRL_PARAM_FIELDS
    )
    params = module.RTRRLParams(seed=7, episodes=2)
    assert params.env_params.env_name == "StatelessCartPoleEasy"
    assert params.env_params.render is False
    assert params.env_params.max_ep_length == 1000
    assert params.env_params.batch_size == 1
    assert params.optimizer_params_td.learning_rate == 1e-4
    assert params.optimizer_params_td.gradient_clip is None
    assert params.optimizer_params_rnn.gradient_clip == 1.0
    assert params.update_trace_before_td is False

    params.align_action_logprob = True
    params.update_trace_before_td = True
    params.episodes = 3
    assert (params.align_action_logprob, params.update_trace_before_td) == (
        True,
        True,
    )
    assert params.episodes == 3


def test_train_rtrrl_normalizes_mutable_params_before_delegating(monkeypatch):
    module = _load_legacy_entrypoint_module()
    observed = {}

    class Runner:
        @staticmethod
        def train_legacy(config, logger):
            observed["config"] = config
            observed["logger"] = logger
            return "delegated"

    monkeypatch.setattr(module, "import_module", lambda _: Runner)
    params = module.RTRRLParams(episodes=2, steps=3)
    params.env_params = module.EnvironmentParams(
        env_name="brax-hopper",
        batch_size=1,
    )
    logger = object()

    assert module.train_rtrrl(params, logger) == "delegated"
    assert observed["config"].total_timesteps == 6
    assert observed["config"].num_epochs == 2
    assert observed["config"].env_params.env_name == "brax-hopper"
    assert observed["logger"] is logger


def test_direct_rtrrl_params_budget_matches_equivalent_yaml_mapping():
    module = _load_legacy_entrypoint_module()
    params = module.RTRRLParams(episodes=3, steps=5)
    params.env_params = module.EnvironmentParams(
        env_name="brax-hopper",
        batch_size=64,
    )

    direct = compatibility_entrypoint.normalize_legacy_invocation(params)
    mapping = compatibility_entrypoint.normalize_legacy_invocation(
        {
            "episodes": 3,
            "steps": 5,
            "env_params": {
                "env_name": "brax-hopper",
                "batch_size": 64,
            },
        }
    )

    assert direct.total_timesteps == mapping.total_timesteps == 960
    assert direct.num_epochs == mapping.num_epochs == 3
    assert direct.num_envs == mapping.num_envs == 64


def test_explicit_budget_overrides_take_precedence_over_nested_batch_size():
    config = compatibility_entrypoint.normalize_legacy_invocation(
        {
            "episodes": 3,
            "steps": 5,
            "num_envs": 8,
            "total_timesteps": 777,
            "env_params": {
                "env_name": "brax-hopper",
                "batch_size": 64,
            },
        }
    )

    assert config.num_envs == 8
    assert config.total_timesteps == 777
    assert config.num_epochs == 3


def test_direct_rtrrl_params_nested_optimizer_rates_become_canonical():
    module = _load_legacy_entrypoint_module()
    params = module.RTRRLParams()
    params.optimizer_params_td = module.OptimizerConfig(
        learning_rate=0.071
    )
    params.optimizer_params_rnn = module.OptimizerConfig(
        learning_rate=0.072,
        gradient_clip=1.0,
    )

    config = compatibility_entrypoint.normalize_legacy_invocation(params)

    assert config.td_lr == pytest.approx(0.071)
    assert config.rnn_lr == pytest.approx(0.072)
    assert config.optimizer_params_td.learning_rate == pytest.approx(0.071)
    assert config.optimizer_params_rnn.learning_rate == pytest.approx(0.072)


def test_direct_rtrrl_params_rnn_gradient_clip_is_canonicalized():
    module = _load_legacy_entrypoint_module()
    nested_only = module.RTRRLParams()
    nested_only.optimizer_params_rnn = module.OptimizerConfig(
        learning_rate=1e-4,
        gradient_clip=0.51,
    )
    top_only = module.RTRRLParams()
    top_only.rnn_grad_clip = 0.52
    equal_sources = module.RTRRLParams()
    equal_sources.rnn_grad_clip = 0.53
    equal_sources.optimizer_params_rnn = module.OptimizerConfig(
        learning_rate=1e-4,
        gradient_clip=0.53,
    )

    nested_config = compatibility_entrypoint.normalize_legacy_invocation(
        nested_only
    )
    top_config = compatibility_entrypoint.normalize_legacy_invocation(
        top_only
    )
    equal_config = compatibility_entrypoint.normalize_legacy_invocation(
        equal_sources
    )

    assert nested_config.rnn_grad_clip == pytest.approx(0.51)
    assert nested_config.optimizer_params_rnn.gradient_clip == pytest.approx(
        0.51
    )
    assert top_config.rnn_grad_clip == pytest.approx(0.52)
    assert top_config.optimizer_params_rnn.gradient_clip == pytest.approx(
        0.52
    )
    assert equal_config.rnn_grad_clip == pytest.approx(0.53)
    assert equal_config.optimizer_params_rnn.gradient_clip == pytest.approx(
        0.53
    )


def test_direct_rtrrl_params_conflicting_rnn_gradient_clips_fail():
    module = _load_legacy_entrypoint_module()
    params = module.RTRRLParams()
    params.rnn_grad_clip = 0.54
    params.optimizer_params_rnn = module.OptimizerConfig(
        learning_rate=1e-4,
        gradient_clip=0.55,
    )

    with pytest.raises(
        ValueError,
        match=(
            r"rnn_grad_clip.*"
            r"optimizer_params_rnn\.gradient_clip"
        ),
    ):
        compatibility_entrypoint.normalize_legacy_invocation(params)


def test_normalize_invocation_preserves_direct_legacy_config_sources():
    config = compatibility_entrypoint.normalize_legacy_invocation(
        compatibility_entrypoint.LegacyRTRRLConfig(
            rnn_grad_clip=0.56
        )
    )

    assert config.rnn_grad_clip == pytest.approx(0.56)
    assert config.optimizer_params_rnn.gradient_clip == pytest.approx(0.56)


def test_rtrrl_fixed_parse_then_assignment_remains_executable(monkeypatch):
    module = _load_legacy_entrypoint_module()
    observed = {}
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "rtrrl"))
    monkeypatch.setattr(
        "logging_util.with_logger",
        lambda _, hparams, **__: observed.setdefault("hparams", hparams),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rtrrl_fixed.py",
            "--config_path",
            str(REPOSITORY_ROOT / "rtrrl/config/rtrrl_hop_533.yml"),
        ],
    )
    monkeypatch.setitem(sys.modules, "rtrrl", module)

    runpy.run_path(
        str(REPOSITORY_ROOT / "rtrrl/rtrrl_fixed.py"),
        run_name="__main__",
    )

    assert observed["hparams"].align_action_logprob is True
    assert observed["hparams"].update_trace_before_td is True


def test_audit_reports_each_migration_class_without_runtime_startup(tmp_path):
    legacy_config = tmp_path / "rtrrl" / "config"
    legacy_config.mkdir(parents=True)
    (legacy_config / "rtrrl_supported.yml").write_text("rnn_model: lru\n")
    (legacy_config / "rtrrl_ctrnn.yml").write_text("rnn_model: ctrnn\n")
    (legacy_config / "rtrrl_null.yml").write_text("rnn_model: null\n")
    (legacy_config / "rtrrl_unknown.yml").write_text("mystery: 1\n")
    (legacy_config / "rtrrl_no_op.yml").write_text("save_model: true\n")
    (legacy_config / "rtrrl_bad_profile.yml").write_text(
        "profile: imaginary\n"
    )
    (legacy_config / "rtrrl_bad_value.yml").write_text(
        "profile: memo_experimental\nbackbone: transformer\n"
    )
    (legacy_config / "rtrrl_bad_root.yml").write_text("- not\n- a mapping\n")
    (legacy_config / "rtrrl_bad_yaml.yml").write_text("[unterminated\n")

    payload = compatibility_entrypoint.audit_repository_configs(tmp_path)

    assert payload["discovered"] == 9
    assert payload["counts"] == {
        "accepted": 1,
        "unsupported": 2,
        "unknown_fields": 1,
        "deprecated_no_op": 1,
        "invalid_config": 4,
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
    assert {
        record["path"] for record in payload["files"]["invalid_config"]
    } == {
        "rtrrl/config/rtrrl_bad_profile.yml",
        "rtrrl/config/rtrrl_bad_root.yml",
        "rtrrl/config/rtrrl_bad_value.yml",
        "rtrrl/config/rtrrl_bad_yaml.yml",
    }


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
        "invalid_config": 0,
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
