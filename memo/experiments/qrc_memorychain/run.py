"""Experiment: QRC(λ) x MemoryChain (arXiv 2605.24709 §4.1).

Algorithm-environment binding is FIXED by this file: QRC(λ) on MemoryChain.
`agent_type` selects the backbone variant (rtu_rtrl / rtu_tbptt); the algorithm
and environment cannot be overridden via config. Sweeping `chain_length` L
isolates temporal credit assignment (paper Figure 1b, QRC curve).
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
from memorax.environments.memory_chain import MemoryChain
from memorax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
    RecordEpisodeStatistics,
)

ALGORITHM = "qrc"
ENVIRONMENT = "memory_chain"


@dataclass
class QRCMemoryChainConfig(ExperimentConfig):
    experiment: str = "qrc_memorychain"

    chain_length: int = 16

    hidden_dim: int = 192
    encoder_dim: int = 64
    meta_rl: bool = False

    # QRC(λ) for MemoryChain.
    num_envs: int = 1
    trace_lambda: float = 0.8
    gradient_correction: bool = True
    reg_coeff: float = 1.0
    q_lr: float = 1e-4
    h_lr: float = 1e-5
    total_timesteps: int = 500_000
    eval_steps: int = 1000


def make_env(cfg: QRCMemoryChainConfig):
    env = MemoryChain(L=cfg.chain_length)
    env = RecordEpisodeStatistics(env)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env)
    return env, env.default_params


def train(cfg: QRCMemoryChainConfig, logger):
    env, env_params = make_env(cfg)
    agent = build_qrc_agent(cfg, env, env_params)
    train_loop(agent, cfg, logger)


if __name__ == "__main__":
    cfg = simple_parsing.parse(QRCMemoryChainConfig, add_config_path_arg=True)
    run_experiment(train, cfg, project_name="memorax-rtrl")
