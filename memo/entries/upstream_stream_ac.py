"""StreamAC as upstream wrote it, for one comparison and nothing else.

This exists to answer a question our tests cannot. They compare block against
block and leaf against leaf, and they say the two implementations compute the
same arithmetic to within a fraction of a last bit. They do not say two million
steps of it behave alike, because a fraction of a bit at every step is a
different trajectory, and the only way to find out how different is to run both
and look.

So this entry is the same experiment as ``stream_ac`` with one thing swapped:
the algorithm is ``memorax.algorithms.upstream_stream_ac``, which is upstream's
file with its arithmetic untouched. The networks are wired exactly as the other
entry wires them, because a comparison in which two things differ answers about
neither.

Two differences are not swappable and are worth naming rather than hiding:

Normalisation sits in environment wrappers here, which is where upstream puts
it. Our kernel normalises inside itself over a batch of streams. The statistics
are the same Welford update either way -- a test asserts that leaf for leaf --
but the wrapper folds the reset observation in one stream at a time, so this is
the placement rather than the arithmetic.

There is no ``credit`` parameter. Upstream stops the gradient at the incoming
carry, which is one step of backpropagation and no sensitivity, and it does so
by construction rather than by setting. Comparing against it therefore means
comparing against ``credit=tbptt``, and the difference from ``credit=rtrl`` is
the other half of the ablation this entry exists for.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

import flax.linen as nn
from training_sdk.reporter import Reporter

from memorax.algorithms.upstream_stream_ac import (
    UpstreamStreamAC,
    UpstreamStreamACConfig,
)
from memorax.environments import make
from memorax.environments.wrappers.normalize_observation import (
    NormalizeObservationWrapper,
)
from memorax.environments.wrappers.normalize_reward import NormalizeRewardWrapper
from memorax.networks import (
    TORSOS,
    FeatureExtractor,
    Network,
    heads,
    make_torso,
)
from runner.loop import drive, named_scalars

_UNIT = {"type": "float", "low": 0.0, "high": 1.0}
_RATE = {"type": "float", "low": 1e-9, "high": 10.0, "log": True}

# The same names and the same limits as ``stream_ac``, minus the one parameter
# upstream does not have, so that one experiment file can be pointed at either
# entry and mean the same thing.
SPACE: dict[str, Any] = {
    "backbone": list(TORSOS),
    "hidden_dim": {"type": "int", "low": 1, "high": 512},
    "feature_dim": {"type": "int", "low": 1, "high": 512},
    "meta_rl": [False, True],
    "normalize_observation": [True, False],
    "normalize_reward": [True, False],
    "seed": {"type": "int", "low": 0, "high": 1_000_000},
    "gamma": {"type": "float", "low": 0.5, "high": 0.9999},
    "trace_lambda": _UNIT,
    "actor_lr": _RATE,
    "critic_lr": _RATE,
    "actor_kappa": {"type": "float", "low": 0.0, "high": 100.0},
    "critic_kappa": {"type": "float", "low": 0.0, "high": 100.0},
    "entropy_coefficient": {"type": "float", "low": 1e-8, "high": 1.0, "log": True},
    # Upstream's two rules under the names the other entry gives them, since it
    # spells the choice as a boolean and a boolean is not a name.
    "bounded_rule": ["obgd", "adaptive_obgd"],
    "beta2": _UNIT,
    "eps": {"type": "float", "low": 1e-12, "high": 1e-2, "log": True},
}

METRICS: tuple[str, ...] = ("eval/episode_return", "eval/episode_length")

# What upstream's step metrics carry. The step sizes and the per-part norms the
# other entry reports are not here: the bound lives inside upstream's own update
# and does not hand its size back, and asking it to would be editing the file
# this entry exists to leave alone.
TRAINING_METRICS: tuple[str, ...] = (
    "td_error",
    "value",
    "log_prob",
    "entropy",
)


def build(params: Mapping[str, Any], environment) -> UpstreamStreamAC:
    """The other entry's wiring, upstream's algorithm, wrapped normalisation."""

    env, env_params = make(
        environment.id,
        observed=environment.observed,
        backend=environment.backend,
    )
    gamma = float(params["gamma"])
    eps = float(params["eps"])
    if bool(params["normalize_observation"]):
        env = NormalizeObservationWrapper(env)
    if bool(params["normalize_reward"]):
        env = NormalizeRewardWrapper(env, gamma=gamma, eps=eps)

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
    return UpstreamStreamAC(
        UpstreamStreamACConfig(
            num_envs=environment.num_envs,
            gamma=gamma,
            trace_lambda=float(params["trace_lambda"]),
            actor_lr=float(params["actor_lr"]),
            critic_lr=float(params["critic_lr"]),
            actor_kappa=float(params["actor_kappa"]),
            critic_kappa=float(params["critic_kappa"]),
            entropy_coefficient=float(params["entropy_coefficient"]),
            adaptive=str(params["bounded_rule"]) != "obgd",
            beta2=float(params["beta2"]),
            eps=eps,
        ),
        env,
        env_params,
        network(heads.Gaussian(action_dim=action_dim)),
        network(heads.VNetwork()),
    )


def training_report(metrics) -> dict[str, float]:
    return named_scalars(metrics, TRAINING_METRICS, prefix="train/")


def run(reporter, config) -> None:
    agent = build(config.params, config.environment)
    scaled = bool(config.params["normalize_reward"])
    drive(
        reporter,
        init_fn=agent.init,
        train_fn=agent.train,
        evaluate_fn=agent.evaluate,
        total_steps=config.budget.total_steps,
        epoch_steps=config.budget.epoch_steps,
        eval_steps=config.budget.eval_steps,
        num_envs=config.environment.num_envs,
        seed=int(config.params["seed"]),
        training_report=training_report,
        # The wrapper replaced the reward the algorithm sees, and the algorithm
        # put that into the summary. The one the environment paid is beside it,
        # and this is the file that knows to look, since it is the file that
        # wrapped the environment.
        eval_reward=(
            (lambda summary: summary.info["environment_reward"]) if scaled else None
        ),
    )


def main(argv: list[str] | None = None) -> int:
    del argv
    with Reporter.from_env() as reporter:
        run(reporter, reporter.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
