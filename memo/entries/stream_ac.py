"""StreamAC over a recurrent actor and critic, credited either way.

What this file fixes is the wiring: the actor and the critic get their own
feature extractor, their own torso and their own head, sharing nothing. Change
that and it is a different algorithm, which is why it is written here rather
than exposed. Every hyperparameter is in ``PARAMETERS``, and an experiment file
narrows it by pinning single values.

``credit`` is what makes one entry enough for what used to be two. Under
``rtrl`` a sensitivity is carried forward and recurrent parameters are credited
exactly; under ``tbptt`` nothing is carried and the gradient stops at the
incoming carry, which is StreamAC as published. Same objective, same bounded
update, same everything else, so a pair of runs that differ only here is an
ablation of exact recurrent credit rather than a comparison of two programs.

``PARAMETERS`` is the only place these names and their limits are written down.
The constructor call below is the only place they are read. Both are on one
screen, which is the whole of the arrangement.

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
from training_sdk.episode import metric_names
from training_sdk.parameters import (
    describe_parameters,
    param,
    read_branch,
    structure,
)
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
    RunningNormalization,
)
from memorax.rl.updates import BASE_BRANCHES, BOUND_BRANCHES
from runner.loop import EPISODE_FIELDS, drive

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

PARTS: tuple[str, ...] = ("feature_extractor", "torso", "head")

TRAINING_METRICS: tuple[str, ...] = (
    "td_error",
    "value",
    "log_prob",
    "entropy",
    "actor_step_size",
    "critic_step_size",
    *(
        f"{domain}_{reading}_norm.{part}"
        for domain in ("actor", "critic")
        for reading in ("grad", "trace")
        for part in PARTS
    ),
)

METRICS: tuple[str, ...] = metric_names("train", TRAINING_METRICS) + metric_names(
    "eval"
)

RECORD = frozenset(EPISODE_FIELDS) | set(TRAINING_METRICS)


_RULES = {
    "ob": "obgd",
    "adaptive_ob": "adaptive_obgd",
    "adaptive_ob_fixed": "adaptive_obgd_fixed",
}


def _optimizer(
    params: Mapping[str, Any], role: str
) -> tuple[str, float, float, float, float]:
    bound_name, bound = read_branch(params, f"{role}_optimizer_bound", BOUND_BRANCHES)
    base_name, base = read_branch(params, f"{role}_optimizer_base", BASE_BRANCHES)
    if bound is None:
        raise ValueError(
            f"{role}_optimizer_bound=none needs the bound and base decomposition; "
            "this kernel always applies a bound"
        )
    if base_name != "sgd":
        raise ValueError(
            f"{role}_optimizer_base={base_name} needs the bound and base "
            "decomposition; this kernel folds the rate into the bound"
        )
    beta2 = getattr(bound, "beta2", 0.0)
    eps = getattr(bound, "eps", 1e-8)
    return _RULES[bound_name], base.lr, bound.kappa, beta2, eps


def _normalization(params: Mapping[str, Any], *, reward_gamma: float):
    _, observation = read_branch(
        params, "observation_normalization", NORMALIZATION_BRANCHES
    )
    _, reward = read_branch(
        params, "reward_normalization", REWARD_NORMALIZATION_BRANCHES
    )
    running = [one for one in (observation, reward) if one is not None]
    shared = {
        (
            one.cold_start,
            one.variance,
            one.eps,
            one.reset_on_start,
            one.update_during_eval,
        )
        for one in running
    }
    if len(shared) > 1:
        raise ValueError(
            "this kernel carries one set of running statistics for both streams, "
            "and the two normalisers ask for different ones"
        )
    first = running[0] if running else RunningNormalization()
    return NormalizationConfig(
        normalize_observation=observation is not None,
        normalize_reward=reward is not None,
        eps=first.eps,
        reward_gamma=reward_gamma,
        reset_on_start=first.reset_on_start,
        update_during_eval=first.update_during_eval,
        cold_start=first.cold_start,
        variance=first.variance,
        reward_trace_reset_on_done=True if reward is None else reward.reset_on_done,
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
        int(params[f"backbone.{backbone}.feature_dim"])
        if backbone == "rtu"
        else hidden_dim
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

    actor_rule, actor_lr, actor_kappa, actor_beta2, actor_eps = _optimizer(
        params, "actor"
    )
    critic_rule, critic_lr, critic_kappa, critic_beta2, critic_eps = _optimizer(
        params, "critic"
    )
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
            actor_lr=actor_lr,
            critic_lr=critic_lr,
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
        record=RECORD,
    )


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
        series=TRAINING_METRICS,
    )


def main(argv: list[str] | None = None) -> int:
    del argv
    with Reporter.from_env() as reporter:
        run(reporter, reporter.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
