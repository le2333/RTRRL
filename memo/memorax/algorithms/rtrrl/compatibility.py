"""Frozen compatibility schema for historical RTRRL configuration."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, fields
from typing import Any


class UnsupportedRTRRLBranch(ValueError):
    """Raised when a historical configuration selects an unsupported branch."""


@dataclass(frozen=True)
class FrozenMapping(Mapping[str, Any]):
    """Small immutable mapping used for intentionally extensible keyword bags."""

    entries: tuple[tuple[str, Any], ...] = ()

    def __getitem__(self, key: str) -> Any:
        for item_key, value in self.entries:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self.entries)

    def __len__(self) -> int:
        return len(self.entries)


@dataclass(frozen=True)
class LegacyOptimizerConfig:
    opt_name: str = "adam"
    learning_rate: float = 1e-4
    kwargs: FrozenMapping = field(default_factory=FrozenMapping)
    decay_type: str | None = None
    lr_kwargs: FrozenMapping = field(default_factory=FrozenMapping)
    weight_decay: float = 0.0
    gradient_clip: float | None = None
    multi_step: int | None = None


@dataclass(frozen=True)
class LegacyEnvironmentConfig:
    env_name: str = "StatelessCartPoleEasy"
    reward_scaling: float = 1.0
    obs_mask: tuple[int, ...] | str | None = None
    init_kwargs: FrozenMapping = field(default_factory=FrozenMapping)
    env_kwargs: FrozenMapping = field(default_factory=FrozenMapping)
    max_ep_length: int = 1000
    batch_size: int | None = 1
    render: bool = False


@dataclass(frozen=True)
class LegacyConfigWarning:
    path: str
    message: str


@dataclass(frozen=True)
class LegacyRTRRLConfig:
    """Complete typed union of supported AAAI25 and Memorax RTRRL fields."""

    profile: str = "aaai25_strict_lru"

    debug: int | bool = 0
    seed: int | None = None
    episodes: int = 150_000
    steps: int = 1000
    total_timesteps: int = 500_000
    num_epochs: int = 50
    num_envs: int = 1
    patience: int = 100

    eval_every: int = 100
    eval_steps: int = 1000
    eval_batch_size: int = 10
    render_every_evals: int = 10
    render_start: int = 0
    render_steps: int = 200

    logging: str | None = None
    log_repo: str | None = None
    run_name: str | None = None
    save_model: bool = False
    log_norms: bool = False
    log_code: bool = False
    log_every: int = 1

    experiment: str = "rtrrl_hopper"
    rtrrl_topology: str = "shared"
    agent_type: str = "rtu_rtrl"
    env_name: str = "hopper"
    mode: str = "F"
    backend: str = "spring"
    env_params: LegacyEnvironmentConfig = field(default_factory=LegacyEnvironmentConfig)

    optimizer_params_td: LegacyOptimizerConfig = field(
        default_factory=LegacyOptimizerConfig
    )
    optimizer_params_rnn: LegacyOptimizerConfig = field(
        default_factory=lambda: LegacyOptimizerConfig(gradient_clip=1.0)
    )

    rnn_model: str = "lru"
    gradient_mode: str = "rflo"
    hidden_size: int = 32
    hidden_dim: int = 32
    encoder_dim: int = 32
    wiring: str = "fully_connected"

    trace_mode: str = "accumulate"
    gamma: float = 0.99
    lambda_v: float = 0.9
    lambda_pi: float = 0.9
    lambda_rnn: float = 0.9
    eta_pi: float = 1.0
    eta_f: float = 1.0
    entropy_rate: float = 1e-5
    eta: float | None = None

    td_lr: float = 1e-4
    rnn_lr: float = 1e-4
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    rnn_grad_clip: float = 1.0

    trace_lambda: float = 0.9
    actor_lr: float = 1.0
    critic_lr: float = 1.0
    actor_kappa: float = 3.0
    critic_kappa: float = 2.0
    entropy_coefficient: float = 0.01
    adaptive: bool = False
    beta2: float = 0.999
    gradient_correction: bool = True
    reg_coeff: float = 1.0
    q_lr: float = 1e-4
    h_lr: float = 1e-5
    epsilon_end: float = 0.01
    epsilon_fraction: float = 0.2

    meta_rl: bool = True
    f_align: bool = False
    normalize_reward: bool = False
    normalize_obs: bool = False
    var_scaling: bool = False
    layer_norm: bool = False
    mlp_actor: bool = False
    pass_obs: bool = False
    align_action_logprob: bool = False
    update_trace_before_td: bool = True
    update_period: float = 1.0
    dropout_rate: float = 0.0
    act_magnitude_factor: float = 0.0
    slow_rnn_factor: float = 0.0

    use_encoder: bool = True
    lru_output_dim: int | None = None
    backbone: str = "lru"
    bound_actor: bool = False
    act_clip: float = 0.0
    freeze_gamma: bool = False
    pred_obs: bool = False
    pred_coeff: float = 1.0
    logprob_reduction: str = "sum"

    warning_records: tuple[LegacyConfigWarning, ...] = ()


@dataclass(frozen=True)
class RTRRLComponentConfig:
    """Effective construction-time component choices."""

    profile: str
    backbone: str
    trace_timing: str
    logprob_reduction: str
    experimental_overrides: tuple[tuple[str, str], ...] = ()
    observation_dim: int = 1
    action_dim: int = 1
    hidden_dim: int = 32
    num_envs: int = 1
    discrete: bool = False
    meta_rl: bool = True
    gamma: float = 0.99
    lambda_actor: float = 0.9
    lambda_critic: float = 0.9
    lambda_recurrent: float = 0.9
    actor_scale: float = 1.0
    recurrent_scale: float = 1.0
    entropy_rate: float = 1e-5
    average_reward_rate: float | None = None
    trace_mode: str = "accumulate"
    optimizer_params_td: LegacyOptimizerConfig = field(
        default_factory=LegacyOptimizerConfig
    )
    optimizer_params_rnn: LegacyOptimizerConfig = field(
        default_factory=lambda: LegacyOptimizerConfig(gradient_clip=1.0)
    )
    update_period: float = 1.0
    normalize_observation: bool = False
    normalize_reward: bool = False
    action_magnitude_factor: float = 0.0
    debug_max_steps: int = 3

    def __post_init__(self) -> None:
        if not 0 <= self.debug_max_steps <= 3:
            raise ValueError("debug_max_steps must be between 0 and 3")


_NO_OP_FIELDS = {
    "save_model": "accepted for compatibility; model saving remains disabled",
    "log_code": "accepted for compatibility; code logging remains disabled",
}
_FLOAT_FIELDS = {
    "act_clip",
    "act_magnitude_factor",
    "actor_kappa",
    "actor_lr",
    "b1",
    "b2",
    "beta2",
    "critic_kappa",
    "critic_lr",
    "dropout_rate",
    "entropy_coefficient",
    "entropy_rate",
    "eps",
    "epsilon_end",
    "epsilon_fraction",
    "eta",
    "eta_f",
    "eta_pi",
    "gamma",
    "h_lr",
    "lambda_pi",
    "lambda_rnn",
    "lambda_v",
    "pred_coeff",
    "q_lr",
    "reg_coeff",
    "rnn_grad_clip",
    "rnn_lr",
    "slow_rnn_factor",
    "td_lr",
    "trace_lambda",
    "update_period",
}
_INT_FIELDS = {
    "encoder_dim",
    "episodes",
    "eval_batch_size",
    "eval_every",
    "eval_steps",
    "hidden_dim",
    "hidden_size",
    "log_every",
    "lru_output_dim",
    "num_envs",
    "num_epochs",
    "patience",
    "render_every_evals",
    "render_start",
    "render_steps",
    "seed",
    "steps",
    "total_timesteps",
}


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenMapping(tuple((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _raw_mapping(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    try:
        return dict(vars(raw))
    except TypeError as error:
        raise TypeError("RTRRL configuration must be a mapping or object") from error


def _unknown_fields(raw: Mapping[str, Any], schema: type, prefix: str = "") -> None:
    known = {item.name for item in fields(schema)}
    for key in raw:
        if key not in known:
            path = f"{prefix}.{key}" if prefix else key
            raise ValueError(f"unknown RTRRL configuration field: {path}")


def _coerce_optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _normalize_optimizer(raw: Any, path: str, default: LegacyOptimizerConfig):
    if raw is None:
        return default
    values = _raw_mapping(raw)
    _unknown_fields(values, LegacyOptimizerConfig, path)
    for key in ("learning_rate", "weight_decay", "gradient_clip"):
        if key in values:
            values[key] = _coerce_optional_float(values[key])
    if "multi_step" in values and values["multi_step"] is not None:
        values["multi_step"] = int(values["multi_step"])
    for key in ("kwargs", "lr_kwargs"):
        if key in values:
            values[key] = _freeze(values[key])
    return LegacyOptimizerConfig(
        **{
            **{item.name: getattr(default, item.name) for item in fields(default)},
            **values,
        }
    )


def _normalize_environment(raw: Any):
    default = LegacyEnvironmentConfig()
    if raw is None:
        return default
    values = _raw_mapping(raw)
    _unknown_fields(values, LegacyEnvironmentConfig, "env_params")
    if "reward_scaling" in values:
        values["reward_scaling"] = float(values["reward_scaling"])
    for key in ("max_ep_length", "batch_size"):
        if key in values and values[key] is not None:
            values[key] = int(values[key])
    for key in ("obs_mask", "init_kwargs", "env_kwargs"):
        if key in values:
            values[key] = _freeze(values[key])
    return LegacyEnvironmentConfig(
        **{
            **{item.name: getattr(default, item.name) for item in fields(default)},
            **values,
        }
    )


def normalize_legacy_config(raw: Any) -> LegacyRTRRLConfig:
    """Validate and freeze a supported historical RTRRL configuration."""

    values = _raw_mapping(raw)
    _unknown_fields(values, LegacyRTRRLConfig)
    explicit = set(values)

    if "rnn_model" in values and values["rnn_model"] != "lru":
        raise UnsupportedRTRRLBranch(
            f"rnn_model={values['rnn_model']!r} is unsupported; only 'lru' is supported"
        )
    values.setdefault("rnn_model", "lru")

    td_default = LegacyOptimizerConfig()
    rnn_default = LegacyOptimizerConfig(gradient_clip=1.0)
    values["optimizer_params_td"] = _normalize_optimizer(
        values.get("optimizer_params_td"), "optimizer_params_td", td_default
    )
    values["optimizer_params_rnn"] = _normalize_optimizer(
        values.get("optimizer_params_rnn"), "optimizer_params_rnn", rnn_default
    )
    values["env_params"] = _normalize_environment(values.get("env_params"))

    for key in _FLOAT_FIELDS & values.keys():
        values[key] = _coerce_optional_float(values[key])
    for key in _INT_FIELDS & values.keys():
        if values[key] is not None:
            values[key] = int(values[key])

    if "td_lr" not in explicit:
        values["td_lr"] = values["optimizer_params_td"].learning_rate
    if "rnn_lr" not in explicit:
        values["rnn_lr"] = values["optimizer_params_rnn"].learning_rate
    if "rnn_grad_clip" not in explicit:
        clip = values["optimizer_params_rnn"].gradient_clip
        values["rnn_grad_clip"] = 1.0 if clip is None else clip
    if "num_envs" not in explicit and values["env_params"].batch_size is not None:
        values["num_envs"] = values["env_params"].batch_size
    if "hidden_dim" not in explicit and "hidden_size" in explicit:
        values["hidden_dim"] = values["hidden_size"]

    values["warning_records"] = tuple(
        LegacyConfigWarning(path=path, message=_NO_OP_FIELDS[path])
        for path in _NO_OP_FIELDS
        if path in explicit
    )
    return LegacyRTRRLConfig(**values)


def to_component_config(legacy: LegacyRTRRLConfig) -> RTRRLComponentConfig:
    """Resolve static component choices without constructing training code."""

    numerical = {
        "action_dim": 1,
        "hidden_dim": legacy.hidden_dim,
        "num_envs": legacy.num_envs,
        "meta_rl": legacy.meta_rl,
        "gamma": legacy.gamma,
        "lambda_actor": legacy.lambda_pi,
        "lambda_critic": legacy.lambda_v,
        "lambda_recurrent": legacy.lambda_rnn,
        "actor_scale": legacy.eta_pi,
        "recurrent_scale": legacy.eta_f,
        "entropy_rate": legacy.entropy_rate,
        "average_reward_rate": legacy.eta,
        "trace_mode": legacy.trace_mode,
        "optimizer_params_td": legacy.optimizer_params_td,
        "optimizer_params_rnn": legacy.optimizer_params_rnn,
        "update_period": legacy.update_period,
        "normalize_observation": legacy.normalize_obs,
        "normalize_reward": legacy.normalize_reward,
        "action_magnitude_factor": legacy.act_magnitude_factor,
    }
    if legacy.profile == "aaai25_strict_lru":
        return RTRRLComponentConfig(
            profile=legacy.profile,
            backbone="aaai25_lru",
            trace_timing="incoming",
            logprob_reduction="mean",
            **numerical,
        )
    if legacy.profile == "memo_experimental":
        trace_timing = "fresh" if legacy.update_trace_before_td else "incoming"
        overrides = (
            ("backbone", legacy.backbone),
            ("logprob_reduction", legacy.logprob_reduction),
            ("trace_timing", trace_timing),
        )
        return RTRRLComponentConfig(
            profile=legacy.profile,
            backbone=legacy.backbone,
            trace_timing=trace_timing,
            logprob_reduction=legacy.logprob_reduction,
            experimental_overrides=overrides,
            **numerical,
        )
    raise ValueError(f"unknown RTRRL component profile: {legacy.profile!r}")
