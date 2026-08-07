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

from memorax.algorithms.stream_ac import StreamAC, StreamACConfig
from memorax.environments import make
from memorax.networks import Readout, Sequence, backbone
from memorax.networks.backbones import Mlp, Rtu
from memorax.networks.initialization import initializer_at
from memorax.networks.readouts import (
    ACTOR_HEAD_BRANCHES,
    CRITIC_HEAD_BRANCHES,
    actor_head,
    critic_head,
)
from memorax.networks.sequence import PLACES
from memorax.parameters import (
    KIND,
    describe_parameters,
    group,
    param,
    read_branch,
    structure,
)
from memorax.rl import CREDITS, declared_normalizer
from memorax.rl.normalization import (
    DISCOUNTED_NORMALIZATION_BRANCHES,
    NORMALIZATION_BRANCHES,
)
from memorax.rl.updates import BASE_BRANCHES, BOUND_BRANCHES
from memorax.runtime import EPISODE_FIELDS, Runtime
from memorax.runtime.episode import metric_names
from worker.reporter import Reporter

BACKBONE_BRANCHES = {"rtu": Rtu, "mlp": Mlp}
CREDIT_BRANCHES = {name: () for name in CREDITS}


# 结构参数表
#
# 分组的层是作用域，选择的层是 structure。``actor`` 不是"哪一个 actor"——没有
# 第二个可选——但它是 actor 的参数住的地方，于是两个角色共用一份优化器声明，
# 而不是四个靠名字里的前缀区分的字段。
@dataclass(frozen=True)
class Optimizer:
    """两个轴，一条界一条底。两个角色各有一个，声明只有这一份。"""

    bound: str = structure(branches=BOUND_BRANCHES)
    base: str = structure(branches=BASE_BRANCHES)


@dataclass(frozen=True)
class Actor:
    head: str = structure(branches=ACTOR_HEAD_BRANCHES)
    optimizer: Optimizer = group(of=Optimizer)


@dataclass(frozen=True)
class Critic:
    # 今天只有一条分支，仍然是 structure：它是选择点，不是作用域。
    head: str = structure(branches=CRITIC_HEAD_BRANCHES)
    optimizer: Optimizer = group(of=Optimizer)


@dataclass(frozen=True)
class Normalization:
    observation: str = structure(branches=NORMALIZATION_BRANCHES)
    reward: str = structure(branches=DISCOUNTED_NORMALIZATION_BRANCHES)


@dataclass(frozen=True)
class StreamACParameters:
    actor: Actor = group(of=Actor)
    critic: Critic = group(of=Critic)
    normalization: Normalization = group(of=Normalization)
    backbone: str = structure(branches=BACKBONE_BRANCHES)
    credit: str = structure(branches=CREDIT_BRANCHES)
    meta_rl: bool = param(valid=[False, True], search=[False, True])
    gamma: float = param(valid=(0.5, 0.9999), search=(0.9, 0.9999))
    trace_lambda: float = param(valid=(0.0, 1.0), search=(0.0, 1.0))
    entropy_coefficient: float = param(valid=(1e-8, 1.0), search=(1e-8, 1e-2), log=True)


# 展开完整参数表
PARAMETERS = describe_parameters(StreamACParameters)

# Position groups, not component names: the component count varies with the
# backbone and METRICS is fixed at import.
PARTS: tuple[str, ...] = PLACES

# 需记录的字段
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

# 指标汇总
METRICS: tuple[str, ...] = metric_names("train", TRAINING_METRICS) + metric_names(
    "eval"
)

RECORD = frozenset(EPISODE_FIELDS) | set(TRAINING_METRICS)


def _optimizer(params: Mapping[str, Any], role: str, axis: str):
    """One axis of one role's optimiser, read back as the component it names.

    The role is a prefix rather than part of the name: both roles read the same
    two branch tables, from their own scope.
    """

    branches = BOUND_BRANCHES if axis == "bound" else BASE_BRANCHES
    _, component = read_branch(params, axis, branches, prefix=f"{role}.optimizer.")
    return component


def _estimator(params: Mapping[str, Any], name: str, branches, *, discount=None):
    """The estimator one stream declared, or none if it declared none."""

    _, component = read_branch(params, name, branches, prefix="normalization.")
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
    chosen = str(params[f"backbone.{KIND}"])
    hidden_dim = int(params[f"backbone.{chosen}.hidden_dim"])
    action_dim = int(env.action_space(env_params).shape[0])
    # What the first component is handed: the observation, and beside it the
    # previous action and reward when the kernel composes them in.
    width = int(env.observation_space(env_params).shape[0])
    if bool(params["meta_rl"]):
        width += action_dim + 1

    # Declared by whatever has kernels: ``mlp`` does, ``rtu`` is the cell and a
    # head and the cell draws its own. A branch that declares none is handed
    # none, and its layers keep the framework's default.
    kernel_init = initializer_at(params, f"backbone.{chosen}.")
    acting = read_branch(params, "head", ACTOR_HEAD_BRANCHES, prefix="actor.")[0]
    valuing = read_branch(params, "head", CRITIC_HEAD_BRANCHES, prefix="critic.")[0]

    def network(head):
        return Sequence(
            components=(
                *backbone(
                    chosen,
                    features=width,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    kernel_init=kernel_init,
                ),
                Readout(module=head),
            )
        )

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
        network(
            actor_head(
                acting,
                action_dim=action_dim,
                kernel_init=initializer_at(params, f"actor.head.{acting}."),
            )
        ),
        network(
            critic_head(
                valuing, kernel_init=initializer_at(params, f"critic.head.{valuing}.")
            )
        ),
        observation_normalization=_estimator(
            params, "observation", NORMALIZATION_BRANCHES
        ),
        reward_normalization=_estimator(
            params,
            "reward",
            DISCOUNTED_NORMALIZATION_BRANCHES,
            discount=gamma,
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
