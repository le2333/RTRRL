from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import runpy

import pytest
import yaml

from memorax.algorithms.rtrrl.compatibility import (
    LegacyOptimizerConfig,
    LegacyRTRRLConfig,
    RTRRLComponentConfig,
    UnsupportedRTRRLBranch,
    normalize_legacy_config,
    to_component_config,
)
from memorax.algorithms.rtrrl.entrypoint import describe_legacy_build


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


@pytest.mark.parametrize(
    ("profile", "raw_rates", "td_rate", "rnn_rate"),
    (
        (
            "aaai25_strict_lru",
            {"td_lr": 0.011, "rnn_lr": 0.012},
            0.011,
            0.012,
        ),
        (
            "aaai25_strict_lru",
            {
                "optimizer_params_td": {"learning_rate": 0.021},
                "optimizer_params_rnn": {"learning_rate": 0.022},
            },
            0.021,
            0.022,
        ),
        (
            "memo_experimental",
            {"td_lr": 0.031, "rnn_lr": 0.032},
            0.031,
            0.032,
        ),
        (
            "memo_experimental",
            {
                "optimizer_params_td": {"learning_rate": 0.041},
                "optimizer_params_rnn": {"learning_rate": 0.042},
            },
            0.041,
            0.042,
        ),
    ),
)
def test_final_optimizer_rates_match_report_and_component_construction(
    profile, raw_rates, td_rate, rnn_rate
):
    config = normalize_legacy_config({"profile": profile, **raw_rates})
    effective = to_component_config(config)
    described = describe_legacy_build(config)["effective"]

    assert config.td_lr == pytest.approx(td_rate)
    assert config.rnn_lr == pytest.approx(rnn_rate)
    assert effective.optimizer_params_td.learning_rate == pytest.approx(
        td_rate
    )
    assert effective.optimizer_params_rnn.learning_rate == pytest.approx(
        rnn_rate
    )
    assert described["td_learning_rate"] == pytest.approx(td_rate)
    assert described["rnn_learning_rate"] == pytest.approx(rnn_rate)


@pytest.mark.parametrize(
    ("top_field", "nested_field"),
    (
        ("td_lr", "optimizer_params_td"),
        ("rnn_lr", "optimizer_params_rnn"),
    ),
)
def test_conflicting_optimizer_rate_sources_fail_with_dotted_paths(
    top_field, nested_field
):
    with pytest.raises(
        ValueError,
        match=rf"{top_field}.*{nested_field}\.learning_rate",
    ):
        normalize_legacy_config(
            {
                top_field: 0.051,
                nested_field: {"learning_rate": 0.052},
            }
        )


def test_equal_top_level_and_nested_optimizer_rates_are_accepted():
    config = normalize_legacy_config(
        {
            "td_lr": 0.055,
            "optimizer_params_td": {"learning_rate": 0.055},
            "rnn_lr": 0.056,
            "optimizer_params_rnn": {"learning_rate": 0.056},
        }
    )

    assert config.optimizer_params_td.learning_rate == pytest.approx(0.055)
    assert config.optimizer_params_rnn.learning_rate == pytest.approx(0.056)


def test_direct_legacy_config_optimizer_sources_are_canonicalized():
    top_only = normalize_legacy_config(LegacyRTRRLConfig(td_lr=0.061))
    nested_only = normalize_legacy_config(
        LegacyRTRRLConfig(
            optimizer_params_td=LegacyOptimizerConfig(
                learning_rate=0.062
            )
        )
    )

    assert top_only.optimizer_params_td.learning_rate == pytest.approx(0.061)
    assert nested_only.td_lr == pytest.approx(0.062)
    with pytest.raises(
        ValueError,
        match=r"td_lr.*optimizer_params_td\.learning_rate",
    ):
        normalize_legacy_config(
            LegacyRTRRLConfig(
                td_lr=0.063,
                optimizer_params_td=LegacyOptimizerConfig(
                    learning_rate=0.064
                ),
            )
        )


@pytest.mark.parametrize(
    ("profile", "raw_clip", "expected_clip"),
    (
        ("aaai25_strict_lru", {"rnn_grad_clip": 0.11}, 0.11),
        (
            "aaai25_strict_lru",
            {"optimizer_params_rnn": {"gradient_clip": 0.12}},
            0.12,
        ),
        ("memo_experimental", {"rnn_grad_clip": 0.21}, 0.21),
        (
            "memo_experimental",
            {"optimizer_params_rnn": {"gradient_clip": 0.22}},
            0.22,
        ),
    ),
)
def test_final_rnn_gradient_clip_matches_report_and_component_construction(
    profile, raw_clip, expected_clip
):
    config = normalize_legacy_config({"profile": profile, **raw_clip})
    effective = to_component_config(config)
    described = describe_legacy_build(config)["effective"]

    assert config.rnn_grad_clip == pytest.approx(expected_clip)
    assert effective.optimizer_params_rnn.gradient_clip == pytest.approx(
        expected_clip
    )
    assert described["rnn_gradient_clip"] == pytest.approx(expected_clip)


def test_conflicting_rnn_gradient_clip_sources_name_both_paths():
    with pytest.raises(
        ValueError,
        match=(
            r"rnn_grad_clip.*"
            r"optimizer_params_rnn\.gradient_clip"
        ),
    ):
        normalize_legacy_config(
            {
                "rnn_grad_clip": 0.31,
                "optimizer_params_rnn": {"gradient_clip": 0.32},
            }
        )


def test_equal_rnn_gradient_clip_sources_are_accepted():
    config = normalize_legacy_config(
        {
            "rnn_grad_clip": 0.33,
            "optimizer_params_rnn": {"gradient_clip": 0.33},
        }
    )

    assert config.rnn_grad_clip == pytest.approx(0.33)
    assert config.optimizer_params_rnn.gradient_clip == pytest.approx(0.33)


def test_direct_legacy_config_rnn_gradient_clip_is_canonicalized():
    top_only = normalize_legacy_config(
        LegacyRTRRLConfig(rnn_grad_clip=0.41)
    )
    nested_only = normalize_legacy_config(
        LegacyRTRRLConfig(
            optimizer_params_rnn=LegacyOptimizerConfig(
                gradient_clip=0.42
            )
        )
    )

    assert top_only.optimizer_params_rnn.gradient_clip == pytest.approx(0.41)
    assert nested_only.rnn_grad_clip == pytest.approx(0.42)
    with pytest.raises(
        ValueError,
        match=(
            r"rnn_grad_clip.*"
            r"optimizer_params_rnn\.gradient_clip"
        ),
    ):
        normalize_legacy_config(
            LegacyRTRRLConfig(
                rnn_grad_clip=0.43,
                optimizer_params_rnn=LegacyOptimizerConfig(
                    gradient_clip=0.44
                ),
            )
        )


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rtrrl_topology", "shared"),
        ("rnn_model", "lru"),
        ("gradient_mode", "rflo"),
        ("wiring", "fully_connected"),
        ("meta_rl", True),
        ("f_align", False),
        ("normalize_obs", False),
        ("normalize_reward", False),
        ("var_scaling", False),
        ("layer_norm", False),
        ("mlp_actor", False),
        ("pass_obs", False),
        ("align_action_logprob", False),
        ("update_trace_before_td", True),
        ("act_magnitude_factor", 0.0),
        ("slow_rnn_factor", 0.0),
        ("use_encoder", True),
        ("encoder_dim", 32),
        ("lru_output_dim", None),
        ("backbone", "lru"),
        ("bound_actor", False),
        ("act_clip", 0.0),
        ("freeze_gamma", False),
        ("pred_obs", False),
        ("pred_coeff", 1.0),
        ("logprob_reduction", "sum"),
    ],
)
def test_strict_profile_rejects_every_explicit_experimental_field(
    field, value
):
    with pytest.raises(
        ValueError, match=rf"strict profile.*experimental.*{field}"
    ):
        to_component_config(
            normalize_legacy_config(
                {"profile": "aaai25_strict_lru", field: value}
            )
        )


@pytest.mark.parametrize(
    "field",
    [
        "num_envs",
        "hidden_dim",
        "gamma",
        "lambda_pi",
        "lambda_v",
        "lambda_rnn",
        "eta_pi",
        "eta_f",
        "entropy_rate",
        "td_lr",
        "rnn_lr",
        "optimizer_params_td",
        "optimizer_params_rnn",
    ],
)
def test_strict_profile_accepts_required_dimensions_and_optimizers(field):
    default = getattr(LegacyRTRRLConfig(), field)
    effective = to_component_config(
        normalize_legacy_config(
            {"profile": "aaai25_strict_lru", field: default}
        )
    )

    assert effective.profile == "aaai25_strict_lru"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backbone", "rtu"),
        ("normalize_obs", True),
        ("act_magnitude_factor", 0.25),
    ],
)
def test_direct_strict_constructor_rejects_nondefault_experimental_values(
    field, value
):
    with pytest.raises(
        ValueError, match=rf"strict profile.*experimental.*{field}"
    ):
        LegacyRTRRLConfig(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backbone", "rtu"),
        ("normalize_obs", True),
        ("act_magnitude_factor", 0.25),
    ],
)
def test_replace_of_normalized_strict_config_revalidates_values(field, value):
    strict = normalize_legacy_config({"profile": "aaai25_strict_lru"})

    with pytest.raises(
        ValueError, match=rf"strict profile.*experimental.*{field}"
    ):
        replace(strict, **{field: value})


def test_direct_strict_constructor_contract_cannot_infer_same_value_presence():
    direct = LegacyRTRRLConfig(backbone="lru")

    assert direct.backbone == "lru"
    assert direct.explicit_fields == ()


def test_raw_mapping_rejects_same_value_presence_during_normalization():
    with pytest.raises(
        ValueError, match=r"strict profile.*experimental.*backbone"
    ):
        normalize_legacy_config(
            {"profile": "aaai25_strict_lru", "backbone": "lru"}
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backbone", "rtu"),
        ("normalize_obs", True),
        ("act_magnitude_factor", 0.25),
    ],
)
def test_normalizer_revalidates_direct_strict_values_at_its_boundary(
    field, value
):
    invalid = LegacyRTRRLConfig()
    object.__setattr__(invalid, field, value)

    with pytest.raises(
        ValueError, match=rf"strict profile.*experimental.*{field}"
    ):
        normalize_legacy_config(invalid)


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
