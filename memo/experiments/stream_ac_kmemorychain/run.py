"""Experiment: Stream AC(λ) x KMemoryChain (arXiv 2605.24709 §4.4).

Algorithm-environment binding is FIXED by this file: Stream AC(λ) on
KMemoryChain. `agent_type` selects the backbone variant (rtu_rtrl / rtu_tbptt);
the algorithm and environment cannot be overridden via config. Sweeping (K, L)
probes staleness of the RTRL credit-assignment signal (paper §4.4, stream AC
curve).
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
from memorax.environments.kmemory_chain import KMemoryChain
from memorax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
    RecordEpisodeStatistics,
)

ALGORITHM = "stream_ac"
ENVIRONMENT = "kmemory_chain"


@dataclass
class StreamACKMemoryChainConfig(ExperimentConfig):
    experiment: str = "stream_ac_kmemorychain"

    chain_length: int = 16
    num_bits: int = 4

    hidden_dim: int = 192
    encoder_dim: int = 64
    meta_rl: bool = False

    # Stream AC(λ) for KMemoryChain.
    num_envs: int = 16
    trace_lambda: float = 0.9
    entropy_coefficient: float = 0.01
    total_timesteps: int = 500_000
    eval_steps: int = 1000


def make_env(cfg: StreamACKMemoryChainConfig):
    env = KMemoryChain(L=cfg.chain_length, K=cfg.num_bits)
    env = RecordEpisodeStatistics(env)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env)
    return env, env.default_params


def train(cfg: StreamACKMemoryChainConfig, logger):
    env, env_params = make_env(cfg)
    agent = build_stream_ac_agent(cfg, env, env_params)
    train_loop(agent, cfg, logger)


if __name__ == "__main__":
    cfg = simple_parsing.parse(StreamACKMemoryChainConfig, add_config_path_arg=True)
    run_experiment(train, cfg, project_name="memorax-rtrl")
