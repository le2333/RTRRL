"""StreamAC over a recurrent actor and critic, credited either way.

What this file fixes is the wiring: the actor and the critic get their own
feature extractor, their own torso and their own head, sharing nothing. Change
that and it is a different algorithm, which is why it is written here rather
than exposed. Everything else -- which task, which cell, how wide, how long,
how the recurrence is credited, every hyperparameter -- is in ``SPACE``, and an
experiment file narrows it by pinning single values.

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
from typing import Any

import flax.linen as nn
from training_sdk.reporter import Reporter

from memorax.algorithms.stream_ac import StreamAC, StreamACConfig
from memorax.environments import make
from memorax.environments.brax import masks
from memorax.networks import (
    TORSOS,
    FeatureExtractor,
    Network,
    heads,
    make_torso,
)
from memorax.rl import BOUNDED_RULES, CREDITS, NormalizationConfig
from runner.loop import drive, named_scalars

_UNIT = {"type": "float", "low": 0.0, "high": 1.0}
_RATE = {"type": "float", "low": 1e-9, "high": 10.0, "log": True}

SPACE: dict[str, Any] = {
    # Exactly the Brax tasks a mask has been worked out for, because masking is
    # the point of this recipe and an unmasked task belongs in another file.
    # Taken from the table rather than copied out of it, so the list cannot
    # come to name a task that has no mask.
    "environment": [f"brax::{task}" for task in sorted(masks)],
    # F is fully observed; P leaves only positions and V only velocities, which
    # is what makes these tasks worth a recurrent policy at all.
    "env_mode": ["F", "P", "V"],
    "env_backend": ["generalized", "spring", "positional", "mjx"],
    "backbone": list(TORSOS),
    "hidden_dim": {"type": "int", "low": 1, "high": 512},
    "feature_dim": {"type": "int", "low": 1, "high": 512},
    # Whether the previous action and reward are fed back in alongside the
    # observation, which is what lets the agent condition on its own history.
    "meta_rl": [False, True],
    # On unless an experiment says otherwise. A bounded step reads the size of
    # its own inputs, so an unnormalised observation changes what the bound
    # means; the published runs normalise, and a run that does not is the
    # unusual one. First in the list is what the cheapest sampled point takes.
    "normalize_observation": [True, False],
    "normalize_reward": [True, False],
    # Exact online credit, or one step of backpropagation and no sensitivity.
    "credit": list(CREDITS),
    # Independent streams whose updates are averaged. This divides the step
    # budget: sixteen streams over two million steps make 125k updates.
    "num_envs": {"type": "int", "low": 1, "high": 256},
    "total_steps": {"type": "int", "low": 1, "high": 100_000_000},
    "epoch_steps": {"type": "int", "low": 1, "high": 10_000_000},
    # Iterations rather than environment steps, unlike the two budgets above:
    # the evaluation rollout is this long in each of ``num_envs`` streams.
    # Zero skips evaluation, which leaves the run without a score.
    "eval_steps": {"type": "int", "low": 0, "high": 100_000},
    "seed": {"type": "int", "low": 0, "high": 1_000_000},
    "gamma": {"type": "float", "low": 0.5, "high": 0.9999},
    "trace_lambda": _UNIT,
    "actor_lr": _RATE,
    "critic_lr": _RATE,
    "actor_kappa": {"type": "float", "low": 0.0, "high": 100.0},
    "critic_kappa": {"type": "float", "low": 0.0, "high": 100.0},
    "entropy_coefficient": {"type": "float", "low": 1e-8, "high": 1.0, "log": True},
    "bounded_rule": list(BOUNDED_RULES),
    "beta2": _UNIT,
    "eps": {"type": "float", "low": 1e-12, "high": 1e-2, "log": True},
}

METRICS: tuple[str, ...] = ("eval/episode_return", "eval/episode_length")

# The parts of the networks this file wires, which is what makes a norm per
# part reportable: the kernel measures whatever parts it is given and cannot
# name them.
PARTS: tuple[str, ...] = ("feature_extractor", "torso", "head")

TRAINING_METRICS: tuple[str, ...] = (
    "td_error",
    "value",
    "log_prob",
    "entropy",
    "actor_step_size",
    "critic_step_size",
    # Split by part because the parts are not credited alike: under
    # ``credit=rtrl`` the torso's cell sees the whole stream while what feeds it
    # sees one step, and one norm over the whole network hides that. Read them
    # beside the step sizes, since a bounded step can shrink a large gradient
    # into a small update and the two cases look alike from the update alone.
    *(
        f"{domain}_{reading}_norm/{part}"
        for domain in ("actor", "critic")
        for reading in ("grad", "trace")
        for part in PARTS
    ),
)


def build(params: Mapping[str, Any]) -> StreamAC:
    """Assemble the agent this file is about."""

    env, env_params = make(
        str(params["environment"]),
        mode=str(params["env_mode"]),
        backend=str(params["env_backend"]),
    )
    gamma = float(params["gamma"])
    feature_dim = int(params["feature_dim"])
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
                str(params["backbone"]),
                features=feature_dim * (3 if meta_rl else 1),
                hidden_dim=int(params["hidden_dim"]),
                output_dim=feature_dim,
            ),
            head=head,
        )

    action_dim = int(env.action_space(env_params).shape[0])
    return StreamAC(
        StreamACConfig(
            num_envs=int(params["num_envs"]),
            gamma=gamma,
            trace_lambda=float(params["trace_lambda"]),
            actor_lr=float(params["actor_lr"]),
            critic_lr=float(params["critic_lr"]),
            actor_kappa=float(params["actor_kappa"]),
            critic_kappa=float(params["critic_kappa"]),
            entropy_coefficient=float(params["entropy_coefficient"]),
            credit=str(params["credit"]),
            bounded_rule=str(params["bounded_rule"]),
            beta2=float(params["beta2"]),
            eps=float(params["eps"]),
        ),
        env,
        env_params,
        network(heads.Gaussian(action_dim=action_dim)),
        network(heads.VNetwork()),
        normalization=NormalizationConfig(
            normalize_observation=bool(params["normalize_observation"]),
            normalize_reward=bool(params["normalize_reward"]),
            reward_gamma=gamma,
        ),
    )


def training_report(metrics) -> dict[str, float]:
    """The scalars worth watching while an epoch runs, named one by one."""

    return named_scalars(metrics, TRAINING_METRICS, prefix="train/")


def run(reporter, params: Mapping[str, Any]) -> None:
    agent = build(params)
    drive(
        reporter,
        init_fn=agent.init,
        train_fn=agent.train,
        evaluate_fn=agent.evaluate,
        total_steps=int(params["total_steps"]),
        epoch_steps=int(params["epoch_steps"]),
        eval_steps=int(params["eval_steps"]),
        num_envs=int(params["num_envs"]),
        seed=int(params["seed"]),
        training_report=training_report,
    )


def main(argv: list[str] | None = None) -> int:
    del argv
    with Reporter.from_env() as reporter:
        run(reporter, reporter.config.params)
    return 0


if __name__ == "__main__":
    sys.exit(main())
