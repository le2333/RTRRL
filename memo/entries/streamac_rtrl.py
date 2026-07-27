"""StreamAC-RTRL with separate actor and critic recurrent networks.

What this file fixes is the wiring: the actor and the critic get their own
feature extractor, their own torso and their own head, sharing nothing. Change
that and it is a different algorithm, which is why it is written here rather
than exposed. Everything else -- which task, which cell, how wide, how long,
every hyperparameter -- is in ``SPACE``, and an experiment file narrows it by
pinning single values.

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
import jax
import jax.numpy as jnp
from training_sdk.reporter import Reporter

from memorax.algorithms.stream_ac_rtrl import StreamACRTRL, StreamACRTRLConfig
from memorax.environments import make
from memorax.networks import (
    TORSOS,
    FeatureExtractor,
    Network,
    heads,
    make_torso,
)
from memorax.rl import NormalizationConfig
from runner.episodes import complete_episodes

_UNIT = {"type": "float", "low": 0.0, "high": 1.0}
_RATE = {"type": "float", "low": 1e-9, "high": 10.0, "log": True}

SPACE: dict[str, Any] = {
    # Brax only, because the mask below is a Brax knob and masking is the
    # point of this recipe. A task from another family would be reached
    # through a different file rather than through a branch here.
    "environment": [
        "brax::hopper",
        "brax::walker2d",
        "brax::halfcheetah",
        "brax::ant",
        "brax::inverted_pendulum",
    ],
    # The observation mask. F is fully observed; P leaves only positions and V
    # only velocities, which is what makes these tasks worth a recurrent
    # policy at all.
    "env_mode": ["F", "P", "V"],
    "env_backend": ["generalized", "spring", "positional", "mjx"],
    "backbone": list(TORSOS),
    "hidden_dim": {"type": "int", "low": 1, "high": 512},
    "feature_dim": {"type": "int", "low": 1, "high": 512},
    # Whether the previous action and reward are fed back in alongside the
    # observation, which is what lets the agent condition on its own history.
    "meta_rl": [False, True],
    "normalize_observation": [False, True],
    "normalize_reward": [False, True],
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
    "adaptive": [False, True],
    "beta2": _UNIT,
    "eps": {"type": "float", "low": 1e-12, "high": 1e-2, "log": True},
}

METRICS = ("eval/episode_return", "eval/episode_length")

TRAINING_METRICS = (
    "td_error",
    "value",
    "log_prob",
    "entropy",
    "actor_step_size",
    "critic_step_size",
)


def build(params: Mapping[str, Any]) -> StreamACRTRL:
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
    return StreamACRTRL(
        StreamACRTRLConfig(
            num_envs=int(params["num_envs"]),
            gamma=gamma,
            trace_lambda=float(params["trace_lambda"]),
            actor_lr=float(params["actor_lr"]),
            critic_lr=float(params["critic_lr"]),
            actor_kappa=float(params["actor_kappa"]),
            critic_kappa=float(params["critic_kappa"]),
            entropy_coefficient=float(params["entropy_coefficient"]),
            adaptive=bool(params["adaptive"]),
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

    return {
        f"train/{name}": float(jnp.nanmean(getattr(metrics, name)))
        for name in TRAINING_METRICS
    }


def evaluation_report(reporter, summary, *, done: int, num_envs: int, number: int):
    """Report the score, and hand every whole episode to the viewer."""

    returns: list[float] = []
    lengths: list[int] = []
    for episode in complete_episodes(
        summary,
        phase="eval",
        start_env_steps=done,
        num_envs=num_envs,
        first_number=number,
    ):
        reporter.log_episode(episode)
        number = episode.number + 1
        returns.append(float(sum(episode.rewards)))
        lengths.append(len(episode.actions))

    report = {"eval/reward": float(jnp.nanmean(summary.reward))}
    # A mean over no episodes would be reported as zero and read as a score.
    if returns:
        report["eval/episode_return"] = sum(returns) / len(returns)
        report["eval/episode_length"] = sum(lengths) / len(lengths)
    reporter.report(done, report)
    return number


def run(reporter, params: Mapping[str, Any]) -> None:
    total = int(params["total_steps"])
    epoch = int(params["epoch_steps"])
    evaluation = int(params["eval_steps"])
    num_envs = int(params["num_envs"])
    # A budget that does not divide has to be rounded, and either direction is
    # a lie: down reports a step count the run never reached, up spends money
    # nobody asked for.
    if epoch % num_envs:
        raise ValueError(f"epoch_steps {epoch} is not {num_envs} streams' worth")
    if total % epoch:
        raise ValueError(f"total_steps {total} is not whole epochs of {epoch}")

    agent = build(params)
    train = jax.jit(agent.train, static_argnums=2)
    evaluate = jax.jit(agent.evaluate, static_argnums=2)

    key = jax.random.key(int(params["seed"]))
    key, init_key = jax.random.split(key)
    state = jax.jit(agent.init)(init_key)

    number = 1
    for done in range(epoch, total + 1, epoch):
        key, epoch_key = jax.random.split(key)
        state, metrics = train(epoch_key, state, epoch)
        reporter.report(done, training_report(metrics))

        if not evaluation:
            continue
        key, eval_key = jax.random.split(key)
        _, summary = evaluate(eval_key, state, evaluation)
        number = evaluation_report(
            reporter, summary, done=done, num_envs=num_envs, number=number
        )


def main(argv: list[str] | None = None) -> int:
    del argv
    with Reporter.from_env() as reporter:
        run(reporter, reporter.config.params)
    return 0


if __name__ == "__main__":
    sys.exit(main())
