"""Experiment: QRC(λ) x POPGym (arXiv 2605.24709 §4.2).

Algorithm-environment binding is FIXED by this file: QRC(λ) on a POPGym task.
`agent_type` selects the backbone variant (rtu_rtrl / rtu_tbptt); the algorithm
and environment family cannot be overridden via config. The specific POPGym
task is chosen by `env_id`.
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
    build_qrc_agent,
    run_experiment,
    train_loop,
)
from memorax.environments import environment
from memorax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
    RecordEpisodeStatistics,
)

ALGORITHM = "qrc"
ENVIRONMENT = "popgym"


@dataclass
class QRCPopGymConfig(ExperimentConfig):
    experiment: str = "qrc_popgym"

    env_id: str = "popjym::CountRecall"
    difficulty: float | None = None

    hidden_dim: int = 192
    encoder_dim: int = 64
    meta_rl: bool = False

    # QRC(λ) for POPGym.
    num_envs: int = 1
    trace_lambda: float = 0.8
    gradient_correction: bool = True
    reg_coeff: float = 1.0
    q_lr: float = 1e-4
    h_lr: float = 1e-5
    total_timesteps: int = 5_000_000
    eval_steps: int = 10_000


def make_env(cfg: QRCPopGymConfig):
    kwargs = {}
    if cfg.difficulty is not None:
        kwargs["difficulty"] = cfg.difficulty
    env, env_params = environment.make(cfg.env_id, **kwargs)
    env = RecordEpisodeStatistics(env)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env)
    return env, env_params


def train(cfg: QRCPopGymConfig, logger):
    env, env_params = make_env(cfg)
    agent = build_qrc_agent(cfg, env, env_params)
    train_loop(agent, cfg, logger)


if __name__ == "__main__":
    cfg = simple_parsing.parse(QRCPopGymConfig, add_config_path_arg=True)
    run_experiment(train, cfg, project_name="memorax-rtrl")
