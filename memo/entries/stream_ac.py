"""StreamAC over a recurrent actor and critic, credited either way.

What this file fixes is the wiring: the actor and the critic get their own
feature extractor, their own torso and their own head, sharing nothing. Change
that and it is a different algorithm, which is why it is written here rather
than exposed. Every hyperparameter is in ``SPACE``, and an experiment file
narrows it by pinning single values.

``credit`` is what makes one entry enough for what used to be two. Under
``rtrl`` a sensitivity is carried forward and recurrent parameters are credited
exactly; under ``tbptt`` nothing is carried and the gradient stops at the
incoming carry, which is StreamAC as published. Same objective, same bounded
update, same everything else, so a pair of runs that differ only here is an
ablation of exact recurrent credit rather than a comparison of two programs.

``SPACE`` is the only place these names and their limits are written down. The
constructor call below is the only place they are read. Both are on one screen,
which is the whole of the arrangement.

The score to beat, Aim run d9fe0986 on masked Brax Hopper: evaluation return
sits at 85 after the first epoch, climbs to 269 by 900k steps, jumps to 1043.66
at 1M and then holds between 1014 and 1026.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import flax.linen as nn
from training_sdk.parameters import describe_parameters, param, structure
from training_sdk.reporter import Reporter

from memorax.algorithms.stream_ac import StreamAC, StreamACConfig
from memorax.environments import make
from memorax.networks import (
    FeatureExtractor,
    Network,
    heads,
    make_torso,
)
from memorax.networks.torso import Mlp, Rtu
from memorax.rl import CREDITS, NormalizationConfig
from memorax.rl.normalization import (
    NORMALIZATION_BRANCHES,
    REWARD_NORMALIZATION_BRANCHES,
)
from memorax.rl.updates import BASE_BRANCHES, BOUND_BRANCHES
from runner.loop import drive, named_scalars

BACKBONE_BRANCHES = {"rtu": Rtu, "mlp": Mlp}


@dataclass(frozen=True)
class StreamACParameters:
    backbone: str = structure(placeholder="rtu", branches=BACKBONE_BRANCHES)
    meta_rl: bool = param(valid=[False, True], search=[False, True], placeholder=False)
    credit: str = param(valid=list(CREDITS), search=list(CREDITS), placeholder="tbptt")
    gamma: float = param(valid=(0.5, 0.9999), search=(0.9, 0.9999), placeholder=0.99)
    trace_lambda: float = param(valid=(0.0, 1.0), search=(0.0, 1.0), placeholder=0.9)
    entropy_coefficient: float = param(
        valid=(1e-8, 1.0), search=(1e-8, 1e-2), placeholder=1e-4, log=True
    )
    observation_normalization: str = structure(
        placeholder="running", branches=NORMALIZATION_BRANCHES
    )
    reward_normalization: str = structure(
        placeholder="running", branches=REWARD_NORMALIZATION_BRANCHES
    )
    actor_optimizer_bound: str = structure(placeholder="ob", branches=BOUND_BRANCHES)
    actor_optimizer_base: str = structure(placeholder="sgd", branches=BASE_BRANCHES)
    critic_optimizer_bound: str = structure(placeholder="ob", branches=BOUND_BRANCHES)
    critic_optimizer_base: str = structure(placeholder="sgd", branches=BASE_BRANCHES)


PARAMETERS = describe_parameters(StreamACParameters)

METRICS: tuple[str, ...] = ("eval/episode_return", "eval/episode_length")

PARTS: tuple[str, ...] = ("feature_extractor", "torso", "head")

TRAINING_METRICS: tuple[str, ...] = (
    "td_error",
    "value",
    "log_prob",
    "entropy",
    "actor_step_size",
    "critic_step_size",
    *(
        f"{domain}_{reading}_norm/{part}"
        for domain in ("actor", "critic")
        for reading in ("grad", "trace")
        for part in PARTS
    ),
)


def _bound(params: Mapping[str, Any], role: str) -> tuple[str, float, float, float]:
    field = f"{role}_optimizer_bound"
    branch = str(params[field])
    if branch == "none":
        raise ValueError(
            f"{field}=none needs the bound and base decomposition; this kernel "
            "always applies a bound"
        )
    kappa = float(params[f"{field}.{branch}.kappa"])
    if branch == "ob":
        return "obgd", kappa, 0.0, 1e-8
    return (
        {"adaptive_ob": "adaptive_obgd", "adaptive_ob_fixed": "adaptive_obgd_fixed"}[
            branch
        ],
        kappa,
        float(params[f"{field}.{branch}.beta2"]),
        float(params[f"{field}.{branch}.eps"]),
    )


def _rate(params: Mapping[str, Any], role: str) -> float:
    field = f"{role}_optimizer_base"
    branch = str(params[field])
    if branch != "sgd":
        raise ValueError(
            f"{field}={branch} needs the bound and base decomposition; this "
            "kernel folds the rate into the bound"
        )
    return float(params[f"{field}.{branch}.lr"])


def _normalization(params: Mapping[str, Any], *, reward_gamma: float):
    observation = str(params["observation_normalization"])
    reward = str(params["reward_normalization"])
    settings = {}
    for field, branch in (
        ("observation_normalization", observation),
        ("reward_normalization", reward),
    ):
        if branch == "none":
            continue
        prefix = f"{field}.{branch}."
        found = (
            str(params[prefix + "cold_start"]),
            str(params[prefix + "variance"]),
            float(params[prefix + "eps"]),
            bool(params[prefix + "reset_on_start"]),
            bool(params[prefix + "update_during_eval"]),
        )
        settings.setdefault(found, []).append(field)
    if len(settings) > 1:
        raise ValueError(
            "this kernel carries one set of running statistics for both streams; "
            f"{' and '.join(sorted(name for names in settings.values() for name in names))} "
            "ask for different ones"
        )
    cold_start, variance, eps, reset_on_start, update_during_eval = next(
        iter(settings), ("seeded", "population", 1e-8, False, True)
    )
    return NormalizationConfig(
        normalize_observation=observation != "none",
        normalize_reward=reward != "none",
        eps=eps,
        reward_gamma=reward_gamma,
        reset_on_start=reset_on_start,
        update_during_eval=update_during_eval,
        cold_start=cold_start,
        variance=variance,
        reward_trace_reset_on_done=(
            True
            if reward == "none"
            else bool(params[f"reward_normalization.{reward}.reset_on_done"])
        ),
    )


def build(params: Mapping[str, Any], environment, training) -> StreamAC:
    """Assemble the agent this file is about."""

    env, env_params = make(
        environment.id,
        observed=environment.observed,
        backend=environment.backend,
    )
    gamma = float(params["gamma"])
    backbone = str(params["backbone"])
    hidden_dim = int(params[f"backbone.{backbone}.hidden_dim"])
    feature_dim = (
        int(params[f"backbone.{backbone}.feature_dim"]) if backbone == "rtu" else hidden_dim
    )
    meta_rl = bool(params["meta_rl"])

    def encoder():
        return nn.Sequential((nn.Dense(feature_dim), nn.relu))

    def network(head):
        return Network(
            feature_extractor=FeatureExtractor(
                observation_extractor=encoder(),
                action_extractor=encoder() if meta_rl else None,
                reward_extractor=encoder() if meta_rl else None,
            ),
            torso=make_torso(
                backbone,
                features=feature_dim * (3 if meta_rl else 1),
                hidden_dim=hidden_dim,
                output_dim=feature_dim,
            ),
            head=head,
        )

    actor_rule, actor_kappa, actor_beta2, actor_eps = _bound(params, "actor")
    critic_rule, critic_kappa, critic_beta2, critic_eps = _bound(params, "critic")
    if (actor_rule, actor_beta2, actor_eps) != (critic_rule, critic_beta2, critic_eps):
        raise ValueError(
            "this kernel carries one bounded rule for both roles; actor and "
            f"critic ask for {actor_rule} and {critic_rule}"
        )

    action_dim = int(env.action_space(env_params).shape[0])
    return StreamAC(
        StreamACConfig(
            num_envs=training.num_envs,
            gamma=gamma,
            trace_lambda=float(params["trace_lambda"]),
            actor_lr=_rate(params, "actor"),
            critic_lr=_rate(params, "critic"),
            actor_kappa=actor_kappa,
            critic_kappa=critic_kappa,
            entropy_coefficient=float(params["entropy_coefficient"]),
            credit=str(params["credit"]),
            bounded_rule=actor_rule,
            beta2=actor_beta2,
            eps=actor_eps,
        ),
        env,
        env_params,
        network(heads.Gaussian(action_dim=action_dim)),
        network(heads.VNetwork()),
        normalization=_normalization(params, reward_gamma=gamma),
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
