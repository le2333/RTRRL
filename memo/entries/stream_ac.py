"""StreamAC entry: declares the parameter surface and builds the agent.

The actor and the critic each get their own sequence, sharing nothing.
``PARAMETERS`` is the only place the names and their limits are written down;
``build`` is the only place they are read.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

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
from memorax.networks import Readout, Sequence, backbone, heads
from memorax.networks.backbones import Mlp, Rtu
from memorax.networks.sequence import PLACES
from memorax.rl import CREDITS, declared_normalizer
from memorax.rl.normalization import (
    DISCOUNTED_NORMALIZATION_BRANCHES,
    NORMALIZATION_BRANCHES,
)
from memorax.rl.updates import BASE_BRANCHES, BOUND_BRANCHES
from runner.loop import EPISODE_FIELDS, drive

BACKBONE_BRANCHES = {"rtu": Rtu, "mlp": Mlp}
CREDIT_BRANCHES = {name: () for name in CREDITS}


# 算法接线参数
@dataclass(frozen=True)
class StreamACParameters:
    backbone: str = structure(placeholder="rtu", branches=BACKBONE_BRANCHES)
    meta_rl: bool = param(valid=[False, True], search=[False, True], placeholder=False)
    credit: str = structure(placeholder="tbptt", branches=CREDIT_BRANCHES)
    gamma: float = param(valid=(0.5, 0.9999), search=(0.9, 0.9999), placeholder=0.99)
    trace_lambda: float = param(valid=(0.0, 1.0), search=(0.0, 1.0), placeholder=0.9)
    entropy_coefficient: float = param(
        valid=(1e-8, 1.0), search=(1e-8, 1e-2), placeholder=1e-4, log=True
    )
    observation_normalization: str = structure(
        placeholder="running", branches=NORMALIZATION_BRANCHES
    )
    reward_normalization: str = structure(
        placeholder="running", branches=DISCOUNTED_NORMALIZATION_BRANCHES
    )
    actor_optimizer_bound: str = structure(placeholder="ob", branches=BOUND_BRANCHES)
    actor_optimizer_base: str = structure(placeholder="sgd", branches=BASE_BRANCHES)
    critic_optimizer_bound: str = structure(placeholder="ob", branches=BOUND_BRANCHES)
    critic_optimizer_base: str = structure(placeholder="sgd", branches=BASE_BRANCHES)


PARAMETERS = describe_parameters(StreamACParameters)

# Position groups, not component names: the component count varies with the
# backbone and METRICS is fixed at import.
PARTS: tuple[str, ...] = PLACES

TRAINING_METRICS: tuple[str, ...] = (
    "update.td_error",
    "update.actor_step_size",
    "update.critic_step_size",
    "forward.value",
    "forward.log_prob",
    "forward.entropy",
    *(
        f"update.{domain}_{reading}_norm.{part}"
        for domain in ("actor", "critic")
        for reading in ("grad", "trace")
        for part in PARTS
    ),
)

METRICS: tuple[str, ...] = metric_names("train", TRAINING_METRICS) + metric_names(
    "eval"
)

RECORD = frozenset(EPISODE_FIELDS) | set(TRAINING_METRICS)


def _optimizer(params: Mapping[str, Any], role: str, axis: str):
    """One axis of one role's optimiser, read back as the component it names."""

    branches = BOUND_BRANCHES if axis == "bound" else BASE_BRANCHES
    _, component = read_branch(params, f"{role}_optimizer_{axis}", branches)
    return component


def _estimator(params: Mapping[str, Any], name: str, branches, *, discount=None):
    """The estimator one stream declared, or none if it declared none."""

    _, component = read_branch(params, name, branches)
    normalizer = declared_normalizer(component, discount=discount)
    return None if normalizer is None else normalizer.config


def build(params: Mapping[str, Any], environment, training) -> StreamAC:
    """Assemble the agent this file is about."""

    # 参数
    env, env_params = make(
        environment.id,
        observed=environment.observed,
        backend=environment.backend,
        episode_length=environment.episode_length,
    )
    gamma = float(params["gamma"])
    chosen = str(params["backbone"])
    hidden_dim = int(params[f"backbone.{chosen}.hidden_dim"])
    feature_dim = (
        int(params[f"backbone.{chosen}.feature_dim"]) if chosen == "rtu" else hidden_dim
    )

    def network(head):
        return Sequence(
            components=(
                *backbone(
                    chosen,
                    features=feature_dim,
                    hidden_dim=hidden_dim,
                    output_dim=feature_dim,
                ),
                Readout(module=head),
            )
        )

    action_dim = int(env.action_space(env_params).shape[0])
    return StreamAC(
        StreamACConfig(
            num_envs=training.num_envs,
            gamma=gamma,
            trace_lambda=float(params["trace_lambda"]),
            actor_bound=_optimizer(params, "actor", "bound"),
            actor_base=_optimizer(params, "actor", "base"),
            critic_bound=_optimizer(params, "critic", "bound"),
            critic_base=_optimizer(params, "critic", "base"),
            entropy_coefficient=float(params["entropy_coefficient"]),
            credit=read_branch(params, "credit", CREDIT_BRANCHES)[0],
            meta_rl=bool(params["meta_rl"]),
        ),
        env,
        env_params,
        network(heads.Gaussian(action_dim=action_dim)),
        network(heads.VNetwork()),
        observation_normalization=_estimator(
            params, "observation_normalization", NORMALIZATION_BRANCHES
        ),
        reward_normalization=_estimator(
            params,
            "reward_normalization",
            DISCOUNTED_NORMALIZATION_BRANCHES,
            discount=gamma,
        ),
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
