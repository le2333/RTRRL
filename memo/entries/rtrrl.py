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
from training_sdk.reporter import Reporter

from memorax.algorithms.rtrrl import RTRRL, RTRRLConfig
from memorax.environments import make
from memorax.networks import (
    RECURRENT_TORSOS,
    UPSTREAM_TORSOS,
    FeatureExtractor,
    heads,
    make_torso,
)
from memorax.rl import BOUNDED_RULES, NormalizationConfig
from runner.loop import drive, named_scalars

_UNIT = {"type": "float", "low": 0.0, "high": 1.0}
_RATE = {"type": "float", "low": 1e-9, "high": 10.0, "log": True}

SPACE: dict[str, Any] = {
    # The two published revisions of the LRU are offered here and nowhere else:
    # this is the entry whose reproduction they are the reference for.
    "backbone": [*RECURRENT_TORSOS, *UPSTREAM_TORSOS],
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
    "obgd_rule": list(BOUNDED_RULES),
    # The ablation this entry exists to run.
    "actor_to_recurrent": [False, True],
    "critic_to_recurrent": [False, True],
}

METRICS: tuple[str, ...] = ("eval/episode_return", "eval/episode_length")

TRAINING_METRICS: tuple[str, ...] = (
    "log_prob",
    "value",
    "next_value",
    "td_error",
    "entropy",
    "emphasis",
    # Only OBGD bounds a step, so this is absent under Adam.
    "step_size",
    # How far the torso's own dynamics are from the edge of stability.
    "diag_lambda_max",
    "diag_gamma_max",
    "diag_sens_norm",
    "diag_carry_norm",
    # Trace, gradient and update magnitudes per destination. Reading the actor
    # and critic columns against each other is what the ablation compares.
    "diag_z_rnn",
    "diag_z_actor",
    "diag_z_critic",
    "diag_grad_rnn",
    "diag_grad_actor",
    "diag_grad_critic",
    "diag_grad_actor_rnn",
    "diag_grad_critic_rnn",
    "diag_grad_cosine",
    "diag_upd_rnn",
    "diag_p_torso",
    "diag_p_actor",
    "diag_p_critic",
    "diag_value_abs",
    "diag_td_abs",
    "diag_actor_loc_abs",
    "diag_actor_scale",
    "diag_act_abs",
)


def build(params: Mapping[str, Any], environment, training) -> RTRRL:
    """Assemble the agent this file is about."""

    env, env_params = make(
        environment.id,
        observed=environment.observed,
        backend=environment.backend,
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
        make_torso(
            str(params["backbone"]),
            features=feature_dim * (3 if meta_rl else 1),
            hidden_dim=int(params["hidden_dim"]),
            output_dim=feature_dim,
        ),
        heads.Gaussian(
            action_dim=int(env.action_space(env_params).shape[0]),
            bound=bool(params["bound_actor"]),
        ),
        heads.VNetwork(),
        activation=jax.nn.silu,
        normalization=NormalizationConfig(
            normalize_observation=bool(params["normalize_observation"]),
            normalize_reward=bool(params["normalize_reward"]),
            reward_gamma=gamma,
        ),
    )


def training_report(metrics) -> dict[str, float]:
    """The scalars worth watching while an epoch runs, named one by one."""

    return named_scalars(metrics, TRAINING_METRICS, prefix="train/")


def run(reporter, config) -> None:
    agent = build(config.params, config.environment, config.training)
    drive(
        reporter,
        init_fn=agent.init,
        train_fn=agent.train,
        evaluate_fn=agent.evaluate,
        total_steps=config.training.total_steps,
        epoch_steps=config.training.epoch_steps,
        eval_steps=config.evaluation.steps,
        num_envs=config.training.num_envs,
        seed=config.environment.seed,
        training_report=training_report,
    )


def main(argv: list[str] | None = None) -> int:
    del argv
    with Reporter.from_env() as reporter:
        run(reporter, reporter.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
