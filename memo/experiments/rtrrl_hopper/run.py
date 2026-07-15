"""Experiment: RTRRL (AAAI'25) x Brax Hopper.

Algorithm-environment binding is FIXED by this file: RTRRL (shared LRU-RTRL
torso feeding a linear Gaussian actor + linear V critic, AC(lambda) with three
eligibility traces, adam, Polyak-averaged recurrent target) on Brax Hopper.

Defaults reproduce streaming-rtrrl's best HPO run RTRRL-HOP-533
(config/rtrrl_hop_533.yml, 1M-step Hopper return ~808.67): gamma 0.95,
lambda_v/pi/rnn = 0.9/0.97/0.945, td_lr 3e-5, rnn_lr 2e-6, eta_pi 0.38,
eta_f 0.5, entropy_rate 3e-5, update_period 0.1, obs/reward normalisation,
hidden 32, spring backend, single-env streaming (num_envs=1).
"""
import os
import sys
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_EXP)
for _p in (_EXP, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import simple_parsing

from base.experiment import (
    ExperimentConfig,
    build_rtrrl_agent,
    run_experiment,
    train_loop,
)
from memorax.environments import environment
from memorax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
    RecordEpisodeStatistics,
)

ALGORITHM = "rtrrl"
ENVIRONMENT = "hopper"


@dataclass
class RTRRLHopperConfig(ExperimentConfig):
    experiment: str = "rtrrl_hopper"

    # Brax env.
    env_name: str = "hopper"
    mode: str = "F"  # full observation
    backend: str = "spring"

    # Streaming: single environment (streaming-rtrrl batch_size=1).
    num_envs: int = 1
    total_timesteps: int = 1_000_000
    num_epochs: int = 20  # eval_points
    eval_every: int = 1
    eval_steps: int = 1000
    patience: int = 1_000_000  # effectively disabled

    # Network (meta-RL input [o, a, r]).
    hidden_dim: int = 32
    encoder_dim: int = 32
    meta_rl: bool = True
    # Recurrent backbone: "lru" (baseline, linear SSM + free gamma gain) or "rtu"
    # (complex rotation-decay + tanh => bounded state, gain tied to nu_log, no free
    # gamma). RTU is the backbone-stability probe: it structurally removes both
    # divergence drivers we identified. On Hopper (mode F, ~Markov) it mainly tests
    # stability, not memory capacity.
    backbone: str = "lru"

    # RTRRL hyperparameters (RTRRL-HOP-533).
    gamma: float = 0.95
    lambda_pi: float = 0.97
    lambda_v: float = 0.9
    lambda_rnn: float = 0.945
    td_lr: float = 3e-5
    rnn_lr: float = 2e-6
    eta_pi: float = 0.38
    eta_f: float = 0.5
    entropy_rate: float = 3e-5
    update_period: float = 0.1
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    rnn_grad_clip: float = 1.0

    # Diagnostic / faithfulness ablation switches (default off => reproduces the
    # RTRRL-HOP-533 baseline). See streaming-rtrrl faithfulness notes:
    #   bound_actor: state-dependent loc+log_scale bounded via sigmoid_between
    #                (loc->[-1,1], log_scale->[-2,2], std=softplus) instead of an
    #                unbounded mean + global learnable log_std.
    #   act_clip:    clip the env-facing action to [-act_clip, act_clip] (brax
    #                actions live in [-1,1]); 0 disables.
    #   freeze_gamma: pin the LRU input gain gamma_log at init (no gradient).
    bound_actor: bool = False
    act_clip: float = 0.0
    freeze_gamma: bool = False
    #   pred_obs: auxiliary linear head predicts (next_obs, next_reward) off the
    #            shared torso; its MSE gradient (scaled by pred_coeff) anchors the
    #            representation scale, opposing the target-less gamma/h inflation.
    pred_obs: bool = False
    pred_coeff: float = 1.0


def make_env(cfg: RTRRLHopperConfig):
    env, env_params = environment.make(
        f"brax::{cfg.env_name}", mode=cfg.mode, backend=cfg.backend
    )
    env = RecordEpisodeStatistics(env)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env)
    return env, env_params


def train(cfg: RTRRLHopperConfig, logger):
    env, env_params = make_env(cfg)
    agent = build_rtrrl_agent(cfg, env, env_params)
    train_loop(agent, cfg, logger)


if __name__ == "__main__":
    cfg = simple_parsing.parse(RTRRLHopperConfig, add_config_path_arg=True)
    run_experiment(train, cfg, project_name="memorax-rtrl")
