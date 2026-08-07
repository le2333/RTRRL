"""RTRRL over one recurrent torso shared by an actor and a critic.

The sharing is the wiring this file fixes: both heads read the same torso, and
both can push gradients into it. Whether they actually do is the ablation the
entry exists for -- ``actor_to_recurrent`` and ``critic_to_recurrent`` cut
either path independently, which is how a collapse gets attributed to one head
conflicting with the other rather than to the algorithm as a whole.

Every hyperparameter is in ``SPACE`` next to the constructor call that reads it.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

import flax.linen as nn
import jax

from memorax.algorithms.rtrrl import RTRRL, RTRRLConfig
from memorax.environments import make
from memorax.networks import (
    RECURRENT_BACKBONES,
    UPSTREAM_BACKBONES,
    FeatureExtractor,
    backbone,
    heads,
)
from memorax.rl import NormalizationConfig
from memorax.runtime import EPISODE_FIELDS, Runtime
from memorax.runtime.episode import metric_names
from worker.reporter import Reporter

_UNIT = {"type": "float", "low": 0.0, "high": 1.0}
_RATE = {"type": "float", "low": 1e-9, "high": 10.0, "log": True}

SPACE: dict[str, Any] = {
    # The two published revisions of the LRU are offered here and nowhere else:
    # this is the entry whose reproduction they are the reference for.
    "backbone": [*RECURRENT_BACKBONES, *UPSTREAM_BACKBONES],
    "hidden_dim": {"type": "int", "low": 1, "high": 512},
    "feature_dim": {"type": "int", "low": 1, "high": 512},
    "meta_rl": [False, True],
    "normalize_observation": [False, True],
    "normalize_reward": [False, True],
    "bound_actor": [False, True],
    "gamma": {"type": "float", "low": 0.5, "high": 0.9999},
    "lambda_pi": _UNIT,
    "lambda_v": _UNIT,
    "lambda_rnn": _UNIT,
    "td_lr": _RATE,
    "rnn_lr": _RATE,
    "eta_pi": {"type": "float", "low": 0.0, "high": 10.0},
    "eta_f": {"type": "float", "low": 0.0, "high": 10.0},
    "entropy_rate": {"type": "float", "low": 1e-8, "high": 1.0, "log": True},
    "update_period": _UNIT,
    "b1": _UNIT,
    "b2": _UNIT,
    "eps": {"type": "float", "low": 1e-12, "high": 1e-2, "log": True},
    "rnn_grad_clip": {"type": "float", "low": 0.0, "high": 1e4},
    "act_clip": {"type": "float", "low": 0.0, "high": 1e4},
    "freeze_gamma": [False, True],
    "update_trace_before_td": [False, True],
    "logprob_scale": {"type": "float", "low": 0.0, "high": 100.0},
    # Adam keeps a moment estimate per parameter; OBGD instead bounds the step
    # by the trace and the TD error, and only then do kappa and the two knobs
    # below mean anything.
    "update_rule": ["adam", "obgd"],
    "kappa": {"type": "float", "low": 0.0, "high": 100.0},
    "obgd_beta2": _UNIT,
    "obgd_rule": ["obgd", "adaptive_obgd", "adaptive_obgd_fixed"],
    # The ablation this entry exists to run.
    "actor_to_recurrent": [False, True],
    "critic_to_recurrent": [False, True],
}

TRAINING_METRICS: tuple[str, ...] = (
    "forward.log_prob",
    "forward.value",
    "forward.next_value",
    "update.td_error",
    "forward.entropy",
    "update.emphasis",
    # Only OBGD bounds a step, so this is absent under Adam.
    "update.step_size",
    # How far the torso's own dynamics are from the edge of stability.
    "forward.diag_lambda_max",
    "forward.diag_gamma_max",
    "update.diag_sens_norm",
    "forward.diag_carry_norm",
    # Trace, gradient and update magnitudes per destination. Reading the actor
    # and critic columns against each other is what the ablation compares.
    "update.diag_z_rnn",
    "update.diag_z_actor",
    "update.diag_z_critic",
    "update.diag_grad_rnn",
    "update.diag_grad_actor",
    "update.diag_grad_critic",
    "update.diag_grad_actor_rnn",
    "update.diag_grad_critic_rnn",
    "update.diag_grad_cosine",
    "update.diag_upd_rnn",
    "forward.diag_p_torso",
    "forward.diag_p_actor",
    "forward.diag_p_critic",
    "forward.diag_value_abs",
    "update.diag_td_abs",
    "forward.diag_actor_loc_abs",
    "forward.diag_actor_scale",
    "forward.diag_act_abs",
)

METRICS: tuple[str, ...] = metric_names("train", TRAINING_METRICS) + metric_names(
    "eval"
)

RECORD = frozenset(EPISODE_FIELDS) | set(TRAINING_METRICS)


def build(params: Mapping[str, Any], environment, training) -> RTRRL:
    """Assemble the agent this file is about."""

    env, env_params = make(
        environment.id,
        observed=environment.observed,
        backend=environment.backend,
        episode_length=environment.episode_length,
    )
    gamma = float(params["gamma"])
    feature_dim = int(params["feature_dim"])
    meta_rl = bool(params["meta_rl"])

    def encoder():
        return nn.Sequential((nn.Dense(feature_dim), nn.relu))

    config = RTRRLConfig(
        num_envs=training.num_envs,
        gamma=gamma,
        lambda_pi=float(params["lambda_pi"]),
        lambda_v=float(params["lambda_v"]),
        lambda_rnn=float(params["lambda_rnn"]),
        td_lr=float(params["td_lr"]),
        rnn_lr=float(params["rnn_lr"]),
        eta_pi=float(params["eta_pi"]),
        eta_f=float(params["eta_f"]),
        entropy_rate=float(params["entropy_rate"]),
        update_period=float(params["update_period"]),
        b1=float(params["b1"]),
        b2=float(params["b2"]),
        eps=float(params["eps"]),
        rnn_grad_clip=float(params["rnn_grad_clip"]),
        act_clip=float(params["act_clip"]),
        freeze_gamma=bool(params["freeze_gamma"]),
        update_trace_before_td=bool(params["update_trace_before_td"]),
        logprob_scale=float(params["logprob_scale"]),
        update_rule=str(params["update_rule"]),
        kappa=float(params["kappa"]),
        obgd_beta2=float(params["obgd_beta2"]),
        obgd_rule=str(params["obgd_rule"]),
        actor_to_recurrent=bool(params["actor_to_recurrent"]),
        critic_to_recurrent=bool(params["critic_to_recurrent"]),
    )
    return RTRRL(
        config,
        env,
        env_params,
        FeatureExtractor(
            observation_extractor=encoder(),
            action_extractor=encoder() if meta_rl else None,
            reward_extractor=encoder() if meta_rl else None,
        ),
        backbone(
            str(params["backbone"]),
            features=feature_dim * (3 if meta_rl else 1),
            hidden_dim=int(params["hidden_dim"]),
            output_dim=feature_dim,
        )[0],
        (heads.BoundedGaussian if bool(params["bound_actor"]) else heads.Gaussian)(
            action_dim=int(env.action_space(env_params).shape[0])
        ),
        heads.VNetwork(),
        activation=jax.nn.silu,
        observation_normalization=(
            NormalizationConfig(center=True)
            if bool(params["normalize_observation"])
            else None
        ),
        reward_normalization=(
            NormalizationConfig(center=False, discount=gamma)
            if bool(params["normalize_reward"])
            else None
        ),
        record=RECORD,
    )


def run(reporter, config) -> None:
    agent = build(config.params, config.environment, config.training)
    Runtime.from_config(agent, config, series=TRAINING_METRICS).run(reporter)


def main(argv: list[str] | None = None) -> int:
    del argv
    with Reporter.from_env() as reporter:
        run(reporter, reporter.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
