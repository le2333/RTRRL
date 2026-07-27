"""The one place that maps an entry name to an algorithm.

Everything downstream of here works through ``AgentProgram`` and the shapes of
what it returns, so adding a topology means adding a builder and nothing else.
A builder reads a flat mapping of parameters, because that is what an HPO
sampler produces and what the run config carries.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import Any

import flax.linen as nn
import jax

from memorax.algorithms.contract import AgentProgram
from memorax.algorithms.rtrrl import RTRRLConfig, RTRRLParts, build_rtrrl
from memorax.algorithms.stream_ac_rtrl import (
    StreamACRTRLConfig,
    StreamACRTRLParts,
    build_stream_ac_rtrl,
)
from memorax.environments import make
from memorax.networks import TORSOS, FeatureExtractor, Network, heads, make_torso
from memorax.rl import NormalizationConfig

# What the runner computes from complete evaluation episodes, for every
# topology, and therefore what a score can always be asked for.
EPISODE_METRICS = ("eval/episode_return", "eval/episode_length")


@dataclass(frozen=True)
class Topology:
    """A named way to build a program, and the parameters it will accept.

    ``space`` is the valid domain of each parameter, not a suggestion: an
    experiment narrows it, and searching all of it is what happens when the
    experiment says nothing. ``metrics`` names what a score may be computed
    from, so a typo in an objective is caught before a job is submitted.
    """

    name: str
    builder: Callable[[Mapping[str, Any]], AgentProgram]
    space: Mapping[str, Any]
    metrics: tuple[str, ...] = EPISODE_METRICS

    def build(self, params: Mapping[str, Any]) -> AgentProgram:
        """Build the program, refusing a parameter the space never declared.

        A misspelling that is silently dropped is a run that reports a setting
        it never used, so it is an error rather than a default.
        """

        unknown = sorted(set(params) - set(self.space))
        if unknown:
            raise ValueError(f"{self.name} has no parameter named {', '.join(unknown)}")
        return self.builder(params)


_FLOAT_UNIT = {"type": "float", "low": 0.0, "high": 1.0}
_BOOL = [False, True]

# The domain the runner itself owns. Every topology accepts these, so they are
# written once rather than repeated per entry.
_RUNNER_SPACE: dict[str, Any] = {
    "environment": [
        "brax::hopper",
        "brax::walker2d",
        "brax::halfcheetah",
        "brax::ant",
        "brax::inverted_pendulum",
        "gymnax::Pendulum-v1",
    ],
    # Brax knobs. "mode" masks the observation: F is fully observed, P leaves
    # only positions and V only velocities, which is what makes these tasks
    # partially observed and therefore worth a recurrent policy at all.
    "env_mode": ["F", "P", "V"],
    "env_backend": ["generalized", "spring", "positional", "mjx"],
    "total_steps": {"type": "int", "low": 1, "high": 100_000_000},
    "epoch_steps": {"type": "int", "low": 1, "high": 10_000_000},
    "eval_steps": {"type": "int", "low": 0, "high": 100_000},
    "seed": {"type": "int", "low": 0, "high": 1_000_000},
    "num_envs": {"type": "int", "low": 1, "high": 256},
    "hidden_dim": {"type": "int", "low": 1, "high": 512},
    "feature_dim": {"type": "int", "low": 1, "high": 512},
    "backbone": list(TORSOS),
    "meta_rl": _BOOL,
    "normalize_observation": _BOOL,
    "normalize_reward": _BOOL,
    "record_trajectory": _BOOL,
}


def _select(params: Mapping[str, Any], config_type) -> dict[str, Any]:
    """Take the subset of parameters the kernel config declares."""

    known = {entry.name for entry in fields(config_type)}
    return {name: value for name, value in params.items() if name in known}


def _environment(params: Mapping[str, Any]):
    """Build the environment, forwarding the knobs its namespace understands.

    Only knobs the experiment actually set are passed on, because a namespace
    rejects a keyword it does not have and most of them have neither of these.
    """

    options = {
        keyword: params[f"env_{keyword}"]
        for keyword in ("mode", "backend")
        if f"env_{keyword}" in params
    }
    return make(params["environment"], **options)


def _normalization(params: Mapping[str, Any]) -> NormalizationConfig:
    """Whether observations and rewards are normalized, and on what scale.

    Both are off unless asked for. Reward normalization divides by the running
    deviation of the discounted return, so it reads the same gamma the critic
    bootstraps with rather than keeping a second one.
    """

    return NormalizationConfig(
        normalize_observation=bool(params.get("normalize_observation", False)),
        normalize_reward=bool(params.get("normalize_reward", False)),
        reward_gamma=float(params.get("gamma", 0.99)),
    )


def _dimensions(env, env_params) -> int:
    return int(env.action_space(env_params).shape[0])


def _encoder(width: int):
    return nn.Sequential((nn.Dense(width), nn.relu))


def _recurrent(backbone: str, features: int, hidden_dim: int, output_dim: int | None):
    return make_torso(
        backbone,
        features=features,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )


def build_rtrrl_topology(params: Mapping[str, Any]) -> AgentProgram:
    """RTRRL over one recurrent torso shared by both heads."""

    env, env_params = _environment(params)
    action_dim = _dimensions(env, env_params)
    feature_dim = int(params.get("feature_dim", 3))
    meta_rl = bool(params.get("meta_rl", True))
    hidden_dim = int(params.get("hidden_dim", 8))

    parts = RTRRLParts(
        env=env,
        env_params=env_params,
        feature_extractor=FeatureExtractor(
            observation_extractor=_encoder(feature_dim),
            action_extractor=_encoder(feature_dim) if meta_rl else None,
            reward_extractor=_encoder(feature_dim) if meta_rl else None,
        ),
        torso=_recurrent(
            str(params.get("backbone", "lru")),
            features=feature_dim * (3 if meta_rl else 1),
            hidden_dim=hidden_dim,
            output_dim=feature_dim,
        ),
        actor_head=heads.Gaussian(action_dim=action_dim),
        critic_head=heads.VNetwork(),
        activation=jax.nn.silu,
        normalization=_normalization(params),
        record_trajectory=bool(params.get("record_trajectory", False)),
    )
    return build_rtrrl(RTRRLConfig(**_select(params, RTRRLConfig)), parts)


def build_stream_ac_rtrl_topology(params: Mapping[str, Any]) -> AgentProgram:
    """StreamAC-RTRL over separate actor and critic recurrent networks."""

    env, env_params = _environment(params)
    action_dim = _dimensions(env, env_params)
    feature_dim = int(params.get("feature_dim", 3))
    meta_rl = bool(params.get("meta_rl", False))
    hidden_dim = int(params.get("hidden_dim", 8))

    def network(head):
        # Actor and critic get their own extractor and torso. Nothing is
        # shared between them, which is what separates this from RTRRL.
        return Network(
            feature_extractor=FeatureExtractor(
                observation_extractor=_encoder(feature_dim),
                action_extractor=_encoder(feature_dim) if meta_rl else None,
                reward_extractor=_encoder(feature_dim) if meta_rl else None,
            ),
            torso=_recurrent(
                str(params.get("backbone", "rtu")),
                features=feature_dim * (3 if meta_rl else 1),
                hidden_dim=hidden_dim,
                output_dim=feature_dim,
            ),
            head=head,
        )

    parts = StreamACRTRLParts(
        env=env,
        env_params=env_params,
        actor_network=network(heads.Gaussian(action_dim=action_dim)),
        critic_network=network(heads.VNetwork()),
        normalization=_normalization(params),
        record_trajectory=bool(params.get("record_trajectory", False)),
    )
    return build_stream_ac_rtrl(
        StreamACRTRLConfig(**_select(params, StreamACRTRLConfig)), parts
    )


TOPOLOGIES: dict[str, Topology] = {
    topology.name: topology
    for topology in (
        Topology(
            name="rtrrl",
            builder=build_rtrrl_topology,
            space={
                **_RUNNER_SPACE,
                "gamma": {"type": "float", "low": 0.5, "high": 0.9999},
                "lambda_pi": _FLOAT_UNIT,
                "lambda_v": _FLOAT_UNIT,
                "lambda_rnn": _FLOAT_UNIT,
                "td_lr": {"type": "float", "low": 1e-9, "high": 10.0, "log": True},
                "rnn_lr": {"type": "float", "low": 1e-9, "high": 10.0, "log": True},
                "eta_pi": {"type": "float", "low": 0.0, "high": 10.0},
                "eta_f": {"type": "float", "low": 0.0, "high": 10.0},
                "entropy_rate": {
                    "type": "float",
                    "low": 1e-8,
                    "high": 1.0,
                    "log": True,
                },
                "update_period": {"type": "float", "low": 0.0, "high": 1.0},
                "b1": _FLOAT_UNIT,
                "b2": _FLOAT_UNIT,
                "eps": {"type": "float", "low": 1e-12, "high": 1e-2, "log": True},
                "rnn_grad_clip": {"type": "float", "low": 0.0, "high": 1e4},
                "act_clip": {"type": "float", "low": 0.0, "high": 1e4},
                "freeze_gamma": _BOOL,
                "update_trace_before_td": _BOOL,
                "logprob_scale": {"type": "float", "low": 0.0, "high": 100.0},
                "pred_coeff": {"type": "float", "low": 0.0, "high": 100.0},
                "update_rule": ["adam", "obgd"],
                "kappa": {"type": "float", "low": 0.0, "high": 100.0},
                "obgd_beta2": _FLOAT_UNIT,
                "obgd_adaptive": _BOOL,
                # The ablation this topology exists to run: either head can be
                # stopped from reaching the shared torso.
                "actor_to_recurrent": _BOOL,
                "critic_to_recurrent": _BOOL,
            },
        ),
        Topology(
            name="stream_ac_rtrl",
            builder=build_stream_ac_rtrl_topology,
            space={
                **_RUNNER_SPACE,
                "gamma": {"type": "float", "low": 0.5, "high": 0.9999},
                "trace_lambda": _FLOAT_UNIT,
                "actor_lr": {"type": "float", "low": 1e-9, "high": 10.0, "log": True},
                "critic_lr": {"type": "float", "low": 1e-9, "high": 10.0, "log": True},
                "actor_kappa": {"type": "float", "low": 0.0, "high": 100.0},
                "critic_kappa": {"type": "float", "low": 0.0, "high": 100.0},
                "entropy_coefficient": {
                    "type": "float",
                    "low": 1e-8,
                    "high": 1.0,
                    "log": True,
                },
                "adaptive": _BOOL,
                "beta2": _FLOAT_UNIT,
                "eps": {"type": "float", "low": 1e-12, "high": 1e-2, "log": True},
            },
        ),
    )
}


def topology(name: str) -> Topology:
    try:
        return TOPOLOGIES[name]
    except KeyError:
        known = ", ".join(sorted(TOPOLOGIES))
        raise ValueError(f"unknown topology {name!r}; registered: {known}") from None
