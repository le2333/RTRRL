"""Uniform-random policy baseline for Brax Hopper."""

import os
import sys
from dataclasses import asdict, dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_EXP)
for _path in (_EXP, _ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import jax
import jax.numpy as jnp
import simple_parsing

from base.experiment import (
    DummyLogger,
    ExperimentConfig,
    run_experiment,
)
from memorax.environments import environment
from memorax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
    RecordEpisodeStatistics,
)


@dataclass
class RandomPolicyHopperConfig(ExperimentConfig):
    experiment: str = "random_policy_hopper"
    env_name: str = "hopper"
    mode: str = "P"
    backend: str = "spring"
    num_evals: int = 20
    eval_steps: int = 1_000
    comparison_step_interval: int = 50_000
    normalize_obs: bool = False
    normalize_reward: bool = False


def make_env(cfg: RandomPolicyHopperConfig):
    env, env_params = environment.make(
        f"brax::{cfg.env_name}", mode=cfg.mode, backend=cfg.backend
    )
    env = RecordEpisodeStatistics(env)
    if cfg.normalize_obs:
        env = NormalizeObservationWrapper(env)
    if cfg.normalize_reward:
        env = NormalizeRewardWrapper(env)
    return env, env_params


def evaluate_random_policy(cfg: RandomPolicyHopperConfig, logger=DummyLogger()):
    """Run independent random-policy evaluations and log episode returns."""
    env, env_params = make_env(cfg)
    action_space = env.action_space(env_params)
    logger.log_params(asdict(cfg))

    @jax.jit
    def evaluate(key):
        reset_key, rollout_key = jax.random.split(key)
        _, env_state = env.reset(reset_key, env_params)

        def step(state, _):
            env_state, key = state
            key, action_key, step_key = jax.random.split(key, 3)
            action = action_space.sample(action_key)
            _, env_state, _, _, info = env.step(
                step_key, env_state, action, env_params
            )
            return (env_state, key), info

        (_, _), info = jax.lax.scan(
            step,
            (env_state, rollout_key),
            xs=None,
            length=cfg.eval_steps,
        )
        completed = info["returned_episode"]
        return (
            jnp.mean(info["returned_episode_returns"], where=completed),
            jnp.sum(completed),
        )

    key = jax.random.key(cfg.seed)
    rewards = []
    try:
        for index in range(cfg.num_evals):
            key, eval_key = jax.random.split(key)
            reward, episodes = evaluate(eval_key)
            reward = float(reward)
            episodes = int(episodes)
            rewards.append(reward)
            logger.log(
                {
                    "eval/rewards": reward,
                    "eval/episodes": episodes,
                },
                step=(index + 1) * cfg.comparison_step_interval,
            )
            print(
                f"Eval {index + 1:02d}: reward={reward:.2f}, "
                f"episodes={episodes}"
            )
    finally:
        logger.finalize()

    mean_reward = sum(rewards) / len(rewards)
    print(f"Random baseline: mean={mean_reward:.2f}, best={max(rewards):.2f}")
    return mean_reward


if __name__ == "__main__":
    config = simple_parsing.parse(
        RandomPolicyHopperConfig,
        add_config_path_arg=True,
    )
    run_experiment(
        evaluate_random_policy,
        config,
        project_name="memorax-rtrl",
    )
