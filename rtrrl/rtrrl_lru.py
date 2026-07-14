"""Online Actor-Critic with eligibility traces — Memorax 版本.

L3 训练调度层:编排 Memorax 的 Stream AC(λ) + RTU(RTRL backbone) + Brax,
仅负责脚本特定逻辑(epochs 循环 / eval / early stopping / logger)。

不依赖 jax_rl,不使用 CTRNN。算法、网络、环境全部来自 Memorax:
- 算法:  memorax.algorithms.StreamAC        (Stream AC(λ) + ObGD + eligibility trace)
- 网络:  memorax.networks.{FeatureExtractor, Network, RNN, RTUCell, heads}
- 环境:  memorax.environments.brax          (Brax via Gymnax 接口)

注意:StreamAC 用 jax.jacobian 做 truncated 1-step 梯度 + eligibility trace 累积
近似 TD(λ),recurrent 参数梯度为单步截断(非完整 RTRL 雅可比迹)。
完整 RTRL/RFLO 路径(RNN.local_jacobian + phantom)由 Memorax 的其他算法启用。
"""

from dataclasses import asdict, dataclass
from functools import partial
from pprint import pprint
import time

import numpy as np
import simple_parsing
import flax.linen as nn
from tqdm import trange

import jax
import jax.numpy as jnp
from jax import random as jrandom

from memorax.algorithms import StreamAC, StreamACConfig
from memorax.environments import environment as mx_env
from memorax.environments.wrappers import RecordEpisodeStatistics
from memorax.networks import (
    FeatureExtractor,
    Network,
    RNN,
    RTUCell,
    RTUConfig,
    heads,
)
from memorax.utils import Timestep

from logging_util import DummyLogger, with_logger

# Uncomment for faster compilation using persistent cache.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 1000000)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)


@dataclass(unsafe_hash=True)
class RTRRLParams:
    """Class representing the parameters for the RTRRL algorithm (Memorax edition)."""

    debug: int | bool = 0
    seed: int | None = None

    # Training
    total_timesteps: int = 500_000
    num_epochs: int = 50
    num_envs: int = 16

    # Validation
    eval_every: int = 10          # evaluate every N epochs
    eval_steps: int = 1000
    patience: int = 100           # early stopping patience (in epochs)

    # Logging
    logging: str | None = None
    log_repo: str | None = None
    run_name: str | None = None
    save_model: bool = False
    log_every: int = 1

    # Environment (Brax)
    env_name: str = "ant"         # one of: ant, halfcheetah, hopper, walker2d
    mode: str = "F"               # F=full obs, P=proprio, V=velocity
    backend: str = "generalized"

    # Stream AC(λ)
    gamma: float = 0.99
    trace_lambda: float = 0.9
    actor_lr: float = 1.0
    critic_lr: float = 1.0
    actor_kappa: float = 3.0
    critic_kappa: float = 2.0
    entropy_coefficient: float = 1e-5
    adaptive: bool = False
    beta2: float = 0.999
    eps: float = 1e-8

    # RTU backbone
    hidden_dim: int = 32          # recurrent state dimension
    features: int = 32            # per-stream feature dim (obs / action / reward)

    # Meta-RL input concatenation [o, a, r]
    meta_rl: bool = True


def build_networks(args: RTRRLParams, action_dim: int):
    """Build actor/critic networks: FeatureExtractor([o,a,r]) -> RNN(RTUCell) -> head."""
    feat = args.features
    observation_extractor = nn.Sequential((nn.Dense(feat), nn.relu))
    action_extractor = (
        nn.Sequential((nn.Dense(feat), nn.relu)) if args.meta_rl else None
    )
    reward_extractor = (
        nn.Sequential((nn.Dense(feat), nn.relu)) if args.meta_rl else None
    )

    feature_extractor = FeatureExtractor(
        observation_extractor=observation_extractor,
        action_extractor=action_extractor,
        reward_extractor=reward_extractor,
    )

    # RTU input dim = concatenated feature streams.
    streams = 3 if args.meta_rl else 1
    rtu_in = feat * streams
    torso = RNN(
        cell=RTUCell(config=RTUConfig(features=rtu_in, hidden_dim=args.hidden_dim))
    )

    # RTU output = concat(real, imaginary) = 2 * hidden_dim
    actor_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.Gaussian(action_dim=action_dim),
    )
    critic_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.VNetwork(),
    )
    return actor_network, critic_network


def make_eval(agent: StreamAC, env, env_params, num_envs: int, num_steps: int):
    """Deterministic-policy eval loop. Accumulates per-step reward across envs.

    StreamAC.evaluate does not surface episode statistics, so we replicate its
    reset + scan(_step, policy=_deterministic_action) and sum transition rewards
    as the eval metric (sum of per-step rewards == sum of episode returns).
    """
    action_space = env.action_space(env_params)
    act_shape = action_space.shape
    act_dtype = action_space.dtype

    def eval_fn(key, state):
        reset_key, eval_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, num_envs)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)

        eval_state = state.replace(
            timestep=Timestep(
                obs=obs,
                action=jnp.zeros((num_envs, *act_shape), dtype=act_dtype),
                reward=jnp.zeros((num_envs,), dtype=jnp.float32),
                done=jnp.ones((num_envs,), dtype=jnp.bool_),
            ),
            env_state=env_state,
            actor_carry=agent.actor_network.initialize_carry((num_envs, None)),
            critic_carry=agent.critic_network.initialize_carry((num_envs, None)),
        )

        step_keys = jax.random.split(eval_key, num_steps)

        def acc(carry, k):
            carry, transition = agent._step(carry, k, policy=agent._deterministic_action)
            return carry, transition.second.reward

        eval_state, rewards = jax.lax.scan(acc, eval_state, step_keys)
        # rewards: (num_steps, num_envs) -> mean over envs of total return
        return rewards.sum(axis=0).mean()

    return eval_fn


def train_rtrrl(args: RTRRLParams, logger=DummyLogger()):
    """Online Actor-Critic with eligibility traces (Memorax Stream AC(λ) + RTU + Brax)."""
    pprint(args, width=1)

    # ENVIRONMENT --------------------------------------------------------------
    env, env_params = mx_env.make(
        f"brax::{args.env_name}", mode=args.mode, backend=args.backend
    )
    env = RecordEpisodeStatistics(env)
    action_dim = env.action_space(env_params).shape[0]

    args.seed = args.seed or np.random.randint(1e6)
    logger.log_params(asdict(args))
    key = jrandom.key(args.seed)

    # NETWORKS + AGENT ---------------------------------------------------------
    actor_network, critic_network = build_networks(args, action_dim)

    config = StreamACConfig(
        num_envs=args.num_envs,
        gamma=args.gamma,
        trace_lambda=args.trace_lambda,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        actor_kappa=args.actor_kappa,
        critic_kappa=args.critic_kappa,
        entropy_coefficient=args.entropy_coefficient,
        adaptive=args.adaptive,
        beta2=args.beta2,
        eps=args.eps,
    )
    agent = StreamAC(config, env, env_params, actor_network, critic_network)

    # JIT'd entrypoints --------------------------------------------------------
    init = jax.jit(agent.init)
    train = jax.jit(agent.train, static_argnames=["num_steps"])
    eval_fn = jax.jit(make_eval(agent, env, env_params, args.num_envs, args.eval_steps))

    key, init_key = jrandom.split(key)
    state = init(init_key)

    num_steps = args.total_timesteps // args.num_epochs

    # Loop misc ---------------------------------------------------------------
    logger["best_eval_reward"] = -jnp.inf
    steps_since_best = 0
    pbar = trange(args.num_epochs, mininterval=1)

    # MAIN LOOP ---------------------------------------------------------------
    try:
        for i in pbar:
            key, train_key, eval_key = jrandom.split(key, 3)
            start = time.perf_counter()
            state, logs = train(train_key, state, num_steps)
            jax.block_until_ready(state)
            sps = num_steps / (time.perf_counter() - start)

            # Episode statistics from RecordEpisodeStatistics (via env.step info).
            info = logs.pop("info", {})
            mask = info.get("returned_episode", jnp.zeros((), dtype=jnp.bool_))
            ep_returns = info.get("returned_episode_returns", jnp.zeros(()))
            ep_lengths = info.get("returned_episode_lengths", jnp.zeros(()))
            avg_r = float(jnp.mean(ep_returns, where=mask))

            metrics = {
                "train/SPS": sps,
                "train/episode_returns": float(jnp.mean(ep_returns, where=mask)),
                "train/episode_lengths": float(jnp.mean(ep_lengths, where=mask)),
            }
            # Forward remaining scalar logs (td_error, entropy, value, ...).
            for k, v in logs.items():
                if isinstance(v, jnp.ndarray) and v.ndim == 0:
                    metrics[f"train/{k}"] = float(v)

            pbar.set_description(f"ep{i} R={avg_r:.2f}", refresh=False)

            # EVAL ----------------------------------------------------------------
            if args.eval_every and (
                i % args.eval_every == 0 or i == args.num_epochs - 1
            ):
                eval_avg = float(eval_fn(eval_key, state))
                metrics["eval/rewards"] = eval_avg
                pbar.write(f"Eval reward: {eval_avg:.2f}")

                if eval_avg > logger["best_eval_reward"]:
                    steps_since_best = 0
                    logger["best_eval_reward"] = eval_avg
                    metrics["eval/best_eval_reward"] = eval_avg
                else:
                    steps_since_best += 1

            logger.log(metrics, step=int(state.step.item()))

            # Early stopping
            if args.patience and steps_since_best >= args.patience:
                print(f"Early stopping patience {args.patience}")
                break
    except Exception as e:
        print("Exception in training loop!")
        raise e
    finally:
        logger.finalize()

    return logger["best_eval_reward"]


if __name__ == "__main__":
    hparams: RTRRLParams = simple_parsing.parse(RTRRLParams, add_config_path_arg=True)
    run_name = hparams.run_name or hparams.env_name
    with_logger(
        train_rtrrl,
        hparams,
        logger_name=hparams.logging,
        project_name="RTRRL",
        aim_repo=hparams.log_repo,
        run_name=run_name,
        hparams_type=RTRRLParams,
    )
