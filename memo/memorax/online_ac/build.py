"""Host-only composition, validation, and legacy lifecycle adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import lox

from memorax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
)
from memorax.networks import RNN, Memoroid

from memorax.rl import (
    NormalizationConfig,
    environment_owns_normalization,
)

from .meta import make_meta_program
from .standard import make_standard_program
from .types import (
    AgentProgram,
    ExactRTRLConfig,
    JAXEnvAdapter,
    MetaProgramConfig,
    SlowSubtreeTargetConfig,
    StandardProgramConfig,
    WholeTreeOBGDConfig,
)


@dataclass(frozen=True)
class _MetaParts:
    env: Any
    env_params: Any
    feature_extractor: Any
    torso: Any
    actor_head: Any
    critic_head: Any
    pred_head: Any
    activation: Any


@dataclass(frozen=True)
class _StandardParts:
    env: Any
    env_params: Any
    actor_network: Any
    critic_network: Any


@dataclass(frozen=True)
class LegacyProgram:
    """Expose an ``AgentProgram`` through the historical algorithm lifecycle."""

    program: AgentProgram
    program_config: MetaProgramConfig | StandardProgramConfig

    def init(self, key):
        return self.program.init_fn(key)

    def warmup(self, key, state, num_steps):
        del key, num_steps
        return state

    def train(self, key, state, num_steps):
        state, metrics = self.program.train_epoch_fn(key, state, num_steps)
        if isinstance(self.program_config, MetaProgramConfig):
            _emit_step_logs(_meta_legacy_logs(metrics))
        elif isinstance(self.program_config, StandardProgramConfig):
            _emit_step_logs(_standard_legacy_logs(metrics))
        else:
            raise TypeError("unsupported legacy program config")
        return state

    def evaluate(self, key, state, num_steps):
        state, summary = self.program.evaluate_fn(key, state, num_steps)
        _emit_step_logs(
            {
                "info": summary.info,
                **_normalization_legacy_logs(summary.normalization),
            }
        )
        return state


def _emit_step_logs(logs):
    """Emit one lox event per fixed scan row from the host-side façade."""

    def emit(carry, step_logs):
        lox.log(step_logs)
        return carry, None

    jax.lax.scan(emit, None, logs)


def _meta_legacy_logs(metrics):
    return {
        "info": metrics.info,
        "critic/td_error": metrics.td_error.mean(axis=-1),
        "actor/entropy": metrics.entropy,
        "critic/value": metrics.value.mean(axis=-1),
        "emphasis/I": metrics.emphasis,
        "diag/lambda_max": metrics.diag_lambda_max,
        "diag/gamma_max": metrics.diag_gamma_max,
        "diag/sens_norm": metrics.diag_sens_norm,
        "diag/carry_norm": metrics.diag_carry_norm,
        "diag/z_rnn": metrics.diag_z_rnn,
        "diag/z_actor": metrics.diag_z_actor,
        "diag/z_critic": metrics.diag_z_critic,
        "diag/grad_rnn": metrics.diag_grad_rnn,
        "diag/grad_actor": metrics.diag_grad_actor,
        "diag/grad_critic": metrics.diag_grad_critic,
        "diag/upd_rnn": metrics.diag_upd_rnn,
        "diag/p_torso": metrics.diag_p_torso,
        "diag/p_actor": metrics.diag_p_actor,
        "diag/p_critic": metrics.diag_p_critic,
        "diag/value_abs": metrics.diag_value_abs,
        "diag/td_abs": metrics.diag_td_abs,
        "diag/actor_loc_abs": metrics.diag_actor_loc_abs,
        "diag/actor_scale": metrics.diag_actor_scale,
        "diag/act_abs": metrics.diag_act_abs,
        **_normalization_legacy_logs(getattr(metrics, "normalization", None)),
    }


def _standard_legacy_logs(metrics):
    return {
        "info": metrics.info,
        "critic/td_error": metrics.td_error.mean(axis=-1),
        "actor/entropy": metrics.entropy,
        "critic/value": metrics.value.mean(axis=-1),
        **_normalization_legacy_logs(getattr(metrics, "normalization", None)),
    }


def _normalization_legacy_logs(metrics):
    if metrics is None:
        return {}
    logs = {}
    for field, key in (
        ("observation_mean", "normalize_observation/mean"),
        ("observation_std", "normalize_observation/std"),
        ("reward_mean", "normalize_reward/mean"),
        ("reward_std", "normalize_reward/std"),
    ):
        value = getattr(metrics, field)
        if value is not None:
            logs[key] = value
    return logs


def _normalization_flags(env):
    normalize_observation = False
    normalize_reward = False
    current = env
    while current is not None:
        normalize_observation |= isinstance(current, NormalizeObservationWrapper)
        normalize_reward |= isinstance(current, NormalizeRewardWrapper)
        current = getattr(current, "_env", None)
    return normalize_observation, normalize_reward


def _strip_outer_normalization(env):
    current = env
    while isinstance(current, (NormalizeObservationWrapper, NormalizeRewardWrapper)):
        current = current._env
    if any(_normalization_flags(current)):
        raise ValueError(
            "inner normalization wrapper cannot be stripped safely; "
            "normalization wrappers must be outermost"
        )
    return current


def legacy_env_adapter(env, env_params, *, strip_normalization=True):
    """Capture a legacy environment as an explicit host-side JAX boundary."""

    normalize_observation, normalize_reward = _normalization_flags(env)
    concrete_env = _strip_outer_normalization(env) if strip_normalization else env
    return JAXEnvAdapter(
        reset_fn=concrete_env.reset,
        step_fn=concrete_env.step,
        env_params=env_params,
        build_context={
            "env": concrete_env,
            "normalize_observation": normalize_observation,
            "normalize_reward": normalize_reward,
        },
    )


def legacy_normalization_config(env, cfg) -> NormalizationConfig:
    """Translate legacy wrapper/flag normalization with fixed reward gamma."""

    wrapped_observation, wrapped_reward = _normalization_flags(env)
    normalize_observation = bool(getattr(cfg, "normalize_obs", wrapped_observation))
    normalize_reward = bool(getattr(cfg, "normalize_reward", wrapped_reward))
    return NormalizationConfig(
        normalize_observation=normalize_observation,
        normalize_reward=normalize_reward,
        eps=float(getattr(cfg, "normalization_eps", 1e-8)),
        reward_gamma=0.99,
        reset_on_start=True,
        update_during_eval=True,
    )


def _validate_evaluation(config):
    evaluation = config.evaluation
    if (
        type(evaluation.reset_on_start) is not bool
        or type(evaluation.update_during_eval) is not bool
    ):
        raise ValueError("evaluation flags must be bool values")
    if evaluation.reset_on_start and not evaluation.update_during_eval:
        raise ValueError(
            "evaluation reset_on_start=True requires update_during_eval=True"
        )


def _validate_normalization(config, env):
    normalization = config.normalization or NormalizationConfig()
    if (
        normalization.normalize_observation or normalization.normalize_reward
    ) and environment_owns_normalization(env):
        raise ValueError(
            "normalization owner conflict: wrapper and program normalization are both enabled"
        )
    return normalization


def _validate_exact_core(core, credit):
    if not isinstance(credit, ExactRTRLConfig) or not isinstance(core, (Memoroid, RNN)):
        raise ValueError("unsupported core-credit composition")
    if not hasattr(core, "initialize_sensitivity"):
        raise ValueError("unsupported core-credit composition")


def build_meta_program(config: MetaProgramConfig, env: JAXEnvAdapter) -> AgentProgram:
    """Exhaustively resolve a meta recipe before creating any JIT closure."""

    if not isinstance(config, MetaProgramConfig):
        raise TypeError("config must be MetaProgramConfig")
    if not isinstance(env, JAXEnvAdapter):
        raise TypeError("env must be JAXEnvAdapter")
    _validate_evaluation(config)
    _validate_exact_core(config.torso, config.credit)
    if not isinstance(config.target, SlowSubtreeTargetConfig):
        raise ValueError("unsupported target composition")
    if config.target.subtree != "torso":
        raise ValueError("target subtree must be 'torso'")
    if config.target.gradient_domain != "fast":
        raise ValueError("target domain must route gradients to fast parameters")
    normalization = _validate_normalization(config, env.build_context["env"])
    required = (
        config.static_config,
        config.feature_extractor,
        config.actor_head,
        config.critic_head,
    )
    if any(value is None for value in required):
        raise ValueError("meta composition is missing required parts")
    parts = _MetaParts(
        env=env.build_context["env"],
        env_params=env.env_params,
        feature_extractor=config.feature_extractor,
        torso=config.torso,
        actor_head=config.actor_head,
        critic_head=config.critic_head,
        pred_head=config.pred_head,
        activation=config.activation or jax.nn.silu,
    )
    return make_meta_program(
        parts,
        config.static_config,
        normalization_config=normalization,
        reset_on_start=config.evaluation.reset_on_start,
        update_during_eval=config.evaluation.update_during_eval,
    )


def build_standard_program(
    config: StandardProgramConfig, env: JAXEnvAdapter
) -> AgentProgram:
    """Exhaustively resolve a standard recipe before creating any JIT closure."""

    if not isinstance(config, StandardProgramConfig):
        raise TypeError("config must be StandardProgramConfig")
    if not isinstance(env, JAXEnvAdapter):
        raise TypeError("env must be JAXEnvAdapter")
    _validate_evaluation(config)
    if not isinstance(config.credit, ExactRTRLConfig):
        raise ValueError("unsupported core-credit composition")
    if not isinstance(config.update, WholeTreeOBGDConfig):
        raise ValueError("unsupported update composition")
    if config.update.domain != "whole_tree":
        raise ValueError("update domain must be whole_tree")
    required = (
        config.static_config,
        config.actor_network,
        config.critic_network,
    )
    if any(value is None for value in required):
        raise ValueError("standard composition is missing required parts")
    _validate_exact_core(config.actor_network.torso, config.credit)
    _validate_exact_core(config.critic_network.torso, config.credit)
    normalization = _validate_normalization(config, env.build_context["env"])
    parts = _StandardParts(
        env=env.build_context["env"],
        env_params=env.env_params,
        actor_network=config.actor_network,
        critic_network=config.critic_network,
    )
    return make_standard_program(
        parts,
        config.static_config,
        normalization_config=normalization,
        reset_on_start=config.evaluation.reset_on_start,
        update_during_eval=config.evaluation.update_during_eval,
    )
