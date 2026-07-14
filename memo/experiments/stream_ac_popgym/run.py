"""Experiment: Stream AC(λ) x POPGym (arXiv 2605.24709 §4.2).

Algorithm-environment binding is FIXED by this file: Stream AC(λ) on a POPGym
task. `agent_type` selects the backbone variant (rtu_rtrl / rtu_tbptt); the
algorithm and environment family cannot be overridden via config. The specific
POPGym task is chosen by `env_id` (a POPGym/POPJym id understood by
memorax.environments.environment.make).
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
ENVIRONMENT = "popgym"


@dataclass
class StreamACPopGymConfig(ExperimentConfig):
    experiment: str = "stream_ac_popgym"

    # POPGym task (memorax.environments.environment.make id).
    env_id: str = "popjym::CountRecall"
    difficulty: float | None = None

    # Network (paper Appendix A).
    hidden_dim: int = 192
    encoder_dim: int = 64
    # Discrete action env: observation-only input.
    meta_rl: bool = False

    # Stream AC(λ) for POPGym.
    num_envs: int = 16
    trace_lambda: float = 0.9
    entropy_coefficient: float = 0.01
    total_timesteps: int = 1_000_000
    eval_steps: int = 1000


def make_env(cfg: StreamACPopGymConfig):
    kwargs = {}
    if cfg.difficulty is not None:
        kwargs["difficulty"] = cfg.difficulty
    env, env_params = environment.make(cfg.env_id, **kwargs)
    env = RecordEpisodeStatistics(env)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env)
    return env, env_params


def train(cfg: StreamACPopGymConfig, logger):
    env, env_params = make_env(cfg)
    agent = build_stream_ac_agent(cfg, env, env_params)
    train_loop(agent, cfg, logger)


if __name__ == "__main__":
    cfg = simple_parsing.parse(StreamACPopGymConfig, add_config_path_arg=True)
    run_experiment(train, cfg, project_name="memorax-rtrl")
