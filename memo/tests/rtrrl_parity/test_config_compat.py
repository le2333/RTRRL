from dataclasses import FrozenInstanceError
from pathlib import Path
import runpy

import pytest
import yaml

from memorax.algorithms.rtrrl.compatibility import (
    LegacyOptimizerConfig,
    RTRRLComponentConfig,
    UnsupportedRTRRLBranch,
    normalize_legacy_config,
    to_component_config,
)


REPOSITORY_ROOT = Path(__file__).parents[3]
REPRESENTATIVE_CONFIGS = (
    "rtrrl/config/rtrrl_hop_533.yml",
    "rtrrl/config/rtrrl_hop_534.yml",
    "memo/config/rtrrl_hopper_533.yml",
    "memo/config/rtrrl_hopper_newlru_base.yml",
)


def test_configuration_interfaces_are_exported_from_rtrrl_package():
    from memorax.algorithms import rtrrl

    assert rtrrl.LegacyRTRRLConfig is not None
    assert rtrrl.RTRRLComponentConfig is not None
    assert rtrrl.normalize_legacy_config is normalize_legacy_config
    assert rtrrl.to_component_config is to_component_config


@pytest.mark.parametrize("relative_path", REPRESENTATIVE_CONFIGS)
def test_representative_yaml_loads_without_edits(relative_path):
    raw = yaml.safe_load((REPOSITORY_ROOT / relative_path).read_text())

    config = normalize_legacy_config(raw)

    assert config.rnn_model == "lru"
    assert config.run_name
    assert isinstance(config.entropy_rate, float)


def test_nested_environment_and_optimizer_fields_survive_and_numeric_strings_coerce():
    config = normalize_legacy_config(
        {
            "env_params": {
                "env_name": "brax-hopper",
                "batch_size": 1,
                "init_kwargs": {"backend": "spring"},
            },
            "optimizer_params_td": {"learning_rate": "3.0e-05"},
            "optimizer_params_rnn": {
                "learning_rate": "2e-6",
                "gradient_clip": "1e0",
            },
        }
    )

    assert config.env_params.env_name == "brax-hopper"
    assert config.env_params.init_kwargs["backend"] == "spring"
    assert config.optimizer_params_td.learning_rate == pytest.approx(3e-5)
    assert config.optimizer_params_rnn.learning_rate == pytest.approx(2e-6)
    assert config.optimizer_params_rnn.gradient_clip == pytest.approx(1.0)
    assert isinstance(config.optimizer_params_td.learning_rate, float)


def test_component_config_preserves_complete_grouped_optimizer_settings():
    legacy = normalize_legacy_config(
        {
            "optimizer_params_td": {
                "opt_name": "adam",
                "learning_rate": "3e-4",
                "kwargs": {"b1": 0.81, "b2": 0.92, "eps": 2e-6},
                "decay_type": "cosine",
                "lr_kwargs": {"decay_steps": 9, "alpha": 0.2},
                "weight_decay": "0.03",
                "gradient_clip": "0.7",
                "multi_step": 2,
            },
            "optimizer_params_rnn": {
                "opt_name": "adam",
                "learning_rate": "4e-5",
                "kwargs": {"b1": 0.71, "b2": 0.82, "eps": 3e-5},
                "decay_type": "exponential",
                "lr_kwargs": {
                    "transition_steps": 3,
                    "decay_rate": 0.5,
                    "staircase": True,
                },
                "weight_decay": "0.04",
                "gradient_clip": "0.6",
                "multi_step": 3,
            },
        }
    )

    effective = to_component_config(legacy)

    assert effective.optimizer_params_td == legacy.optimizer_params_td
    assert effective.optimizer_params_rnn == legacy.optimizer_params_rnn


@pytest.mark.parametrize("limit", [-1, 4])
def test_component_config_rejects_debug_bounds_above_three(limit):
    with pytest.raises(ValueError, match="debug_max_steps"):
        RTRRLComponentConfig(
            profile="aaai25_strict_lru",
            backbone="aaai25_lru",
            trace_timing="incoming",
            logprob_reduction="mean",
            debug_max_steps=limit,
        )


def test_legacy_optimizer_config_remains_the_canonical_nested_type():
    effective = to_component_config(normalize_legacy_config({}))

    assert isinstance(effective.optimizer_params_td, LegacyOptimizerConfig)
    assert isinstance(effective.optimizer_params_rnn, LegacyOptimizerConfig)


def test_explicit_lru_is_supported():
    assert normalize_legacy_config({"rnn_model": "lru"}).rnn_model == "lru"


@pytest.mark.parametrize("unsupported", ["ctrnn", None])
def test_removed_recurrent_branches_fail_clearly(unsupported):
    with pytest.raises(UnsupportedRTRRLBranch, match="rnn_model"):
        normalize_legacy_config({"rnn_model": unsupported})


def test_legacy_no_op_fields_are_retained_with_warning_records():
    config = normalize_legacy_config({"save_model": True, "log_code": True})

    assert config.save_model is True
    assert config.log_code is True
    assert {record.path for record in config.warning_records} == {
        "save_model",
        "log_code",
    }


@pytest.mark.parametrize(
    ("raw", "dotted_path"),
    [
        ({"mystery": 1}, "mystery"),
        ({"env_params": {"mystery": 1}}, "env_params.mystery"),
        (
            {"optimizer_params_td": {"mystery": 1}},
            "optimizer_params_td.mystery",
        ),
    ],
)
def test_unknown_fields_report_their_dotted_path(raw, dotted_path):
    with pytest.raises(ValueError, match=rf"\b{dotted_path}\b"):
        normalize_legacy_config(raw)


def test_normalized_configuration_is_frozen():
    config = normalize_legacy_config({})

    with pytest.raises(FrozenInstanceError):
        config.gamma = 0.1  # pyright: ignore[reportAttributeAccessIssue]


def test_strict_profile_rejects_experimental_branch_flags():
    legacy = normalize_legacy_config(
        {
            "profile": "aaai25_strict_lru",
            "backbone": "rtu",
            "update_trace_before_td": True,
            "logprob_reduction": "sum",
        }
    )

    with pytest.raises(ValueError, match="strict profile.*experimental"):
        to_component_config(legacy)


def test_experimental_profile_records_effective_overrides():
    legacy = normalize_legacy_config(
        {
            "profile": "memo_experimental",
            "backbone": "rtu",
            "update_trace_before_td": False,
            "logprob_reduction": "mean",
        }
    )

    effective = to_component_config(legacy)

    assert effective.backbone == "rtu"
    assert effective.trace_timing == "incoming"
    assert effective.logprob_reduction == "mean"
    assert dict(effective.experimental_overrides) == {
        "backbone": "rtu",
        "logprob_reduction": "mean",
        "trace_timing": "incoming",
    }


def test_memo_experimental_omitted_trace_flag_preserves_builder_fresh_default():
    legacy = normalize_legacy_config({"profile": "memo_experimental"})

    assert legacy.update_trace_before_td is True
    assert to_component_config(legacy).trace_timing == "fresh"


def test_unknown_profile_fails_during_component_resolution():
    legacy = normalize_legacy_config({"profile": "future"})

    with pytest.raises(ValueError, match="future"):
        to_component_config(legacy)


def test_current_hopper_defaults_remain_the_experimental_training_semantics():
    run_module = runpy.run_path(
        str(REPOSITORY_ROOT / "memo/experiments/rtrrl_hopper/run.py")
    )
    config_type = run_module["RTRRLHopperConfig"]

    effective = to_component_config(normalize_legacy_config(config_type()))

    assert effective.profile == "memo_experimental"
    assert effective.backbone == "lru"
    assert effective.trace_timing == "fresh"
    assert effective.logprob_reduction == "sum"
