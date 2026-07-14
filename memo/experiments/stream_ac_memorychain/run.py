"""Experiment: Stream AC(λ) x MemoryChain (arXiv 2605.24709 §4.1).

Algorithm-environment binding is FIXED by this file: Stream AC(λ) on
MemoryChain. `agent_type` selects the backbone variant (rtu_rtrl / rtu_tbptt);
the algorithm and environment cannot be overridden via config. Sweeping
`chain_length` L isolates temporal credit assignment (paper Figure 1b).
"""
import os
import sys
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.dirname(_HERE)          # experiments/
_ROOT = os.path.dirname(_EXP)           # memorax-rtrl/
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
from memorax.environments.memory_chain import MemoryChain
from memorax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
    RecordEpisodeStatistics,
)

ALGORITHM = "stream_ac"
ENVIRONMENT = "memory_chain"


@dataclass
class StreamACMemoryChainConfig(ExperimentConfig):
    experiment: str = "stream_ac_memorychain"

    # MemoryChain (paper §4.1): chain length L is the sweep variable.
    chain_length: int = 16

    # Network (paper Appendix A: RTU hidden=192, encoder=64).
    hidden_dim: int = 192
    encoder_dim: int = 64
    # Discrete action env: observation-only input (matches stream_ac_minatar).
    meta_rl: bool = False

    # Stream AC(λ) for MemoryChain.
    num_envs: int = 16
    trace_lambda: float = 0.9
    entropy_coefficient: float = 0.01
    total_timesteps: int = 500_000
    eval_steps: int = 1000


def make_env(cfg: StreamACMemoryChainConfig):
    env = MemoryChain(L=cfg.chain_length)
    env = RecordEpisodeStatistics(env)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env)
    return env, env.default_params


def train(cfg: StreamACMemoryChainConfig, logger):
    env, env_params = make_env(cfg)
    agent = build_stream_ac_agent(cfg, env, env_params)
    train_loop(agent, cfg, logger)


if __name__ == "__main__":
    cfg = simple_parsing.parse(StreamACMemoryChainConfig, add_config_path_arg=True)
    run_experiment(train, cfg, project_name="memorax-rtrl")
