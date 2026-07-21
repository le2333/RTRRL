"""Experiment: Stream AC(λ) x Masked MuJoCo (arXiv 2605.24709 §4.3).

Algorithm-environment binding is FIXED by this file: Stream AC(λ) on a Brax
MuJoCo env with a partial-observation mask. `agent_type` selects the backbone
variant (rtu_rtrl / rtu_tbptt); the algorithm and environment family cannot be
overridden via config. The mask is selected by `mode`:
  F = full obs, P = proprio (position/external state masked), V = velocity
masked — reproducing the paper's masked-position / masked-velocity POMDPs.
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
    build_stream_ac_agent,
    run_experiment,
    train_loop,
)
from memorax.environments import environment
from memorax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
    RecordEpisodeStatistics,
)

ALGORITHM = "stream_ac"
ENVIRONMENT = "mujoco_masked"


@dataclass
class StreamACMujocoMaskedConfig(ExperimentConfig):
    experiment: str = "stream_ac_mujoco_masked"

    # Brax MuJoCo env (one of: ant, halfcheetah, hopper, walker2d).
    env_name: str = "halfcheetah"
    # F=full, P=proprio (masked position), V=velocity masked.
    mode: str = "P"
    backend: str = "generalized"

    # Network (continuous control: [o, a, r] meta-RL input).
    hidden_dim: int = 32
    encoder_dim: int = 32
    meta_rl: bool = True

    # Stream AC(λ) for Masked MuJoCo.
    num_envs: int = 16
    trace_lambda: float = 0.9
    entropy_coefficient: float = 1e-5
    total_timesteps: int = 5_000_000
    eval_steps: int = 1000


def make_env(cfg: StreamACMujocoMaskedConfig):
    env, env_params = environment.make(
        f"brax::{cfg.env_name}",
        mode=cfg.mode,
        backend=cfg.backend,
        max_episode_steps=cfg.max_episode_steps,
    )
    env = RecordEpisodeStatistics(env)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env)
    return env, env_params


def train(cfg: StreamACMujocoMaskedConfig, logger):
    env, env_params = make_env(cfg)
    agent = build_stream_ac_agent(cfg, env, env_params)
    train_loop(agent, cfg, logger)


if __name__ == "__main__":
    cfg = simple_parsing.parse(StreamACMujocoMaskedConfig, add_config_path_arg=True)
    run_experiment(train, cfg, project_name="memorax-rtrl")
