"""MemoryChain diagnostic: stream AC(lambda) + RTU-RTRL vs RTU-TBPTT(1).

Reproduces the logic of arxiv 2605.24709 Figure 1(b): sweep chain length L,
expect RTU-RTRL to sustain episodic return past L=16 while RTU under TBPTT(1)
collapses. Uses stream AC(lambda) (policy-based) instead of QRC(lambda) for
the diagnostic — the RTRL-vs-truncation contrast is the operative signal.

Run: python examples/stream_ac_rtrl_memorychain.py
"""
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox

from memorax.algorithms import StreamAC, StreamACConfig, StreamACRtrl
from memorax.environments.memory_chain import MemoryChain
from memorax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
    RecordEpisodeStatistics,
)
from memorax.networks import (
    FeatureExtractor,
    Network,
    RNN,
    RTUCell,
    RTUConfig,
    heads,
)

SEED = 0
NUM_ENVS = 16
TOTAL_STEPS = 200_000
HIDDEN = 192          # RTU hidden dim (paper Appendix A)
ENCODER = 64          # encoder width (paper Appendix A)
L_SWEEP = [2, 4, 8, 16, 32, 48, 64]


def build_networks(num_actions: int):
    feature_extractor = FeatureExtractor(
        observation_extractor=nn.Sequential((nn.Dense(ENCODER), nn.leaky_relu))
    )
    torso = RNN(
        cell=RTUCell(config=RTUConfig(features=ENCODER, hidden_dim=HIDDEN))
    )
    actor = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.Categorical(action_dim=num_actions),
    )
    critic = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.VNetwork(),
    )
    return actor, critic


def build_config() -> StreamACConfig:
    return StreamACConfig(
        num_envs=NUM_ENVS,
        gamma=0.99,
        trace_lambda=0.9,
        actor_lr=1.0,
        critic_lr=1.0,
        actor_kappa=3.0,
        critic_kappa=2.0,
        entropy_coefficient=0.01,
    )


def make_env(L: int):
    env = MemoryChain(L=L)
    env = RecordEpisodeStatistics(env)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env)
    return env, env.default_params


def run_one(L: int, use_rtrl: bool, seed: int = SEED) -> float:
    env, env_params = make_env(L)
    num_actions = env.action_space(env_params).n

    actor, critic = build_networks(num_actions)
    cfg = build_config()
    agent_cls = StreamACRtrl if use_rtrl else StreamAC
    agent = agent_cls(cfg, env, env_params, actor, critic)

    init = jax.jit(agent.init)
    train = lox.spool(jax.jit(agent.train, static_argnames=["num_steps"]))

    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    state = init(init_key)

    start = time.perf_counter()
    state, logs = train(key, state, TOTAL_STEPS)
    jax.block_until_ready(state)
    elapsed = time.perf_counter() - start

    info = logs.get("info", {})
    mask = info.get("returned_episode", jnp.zeros((), dtype=jnp.bool_))
    returns = info.get("returned_episode_returns", jnp.zeros(()))
    final_return = float(jnp.mean(returns, where=mask))
    label = "RTRL" if use_rtrl else "TBPTT(1)"
    print(f"  [{label}] L={L:>3} return={final_return:.3f} ({elapsed:.1f}s)")
    return final_return


def main():
    print(f"Sweeping L over {L_SWEEP} (seed={SEED}, steps={TOTAL_STEPS})")
    print(f"{'L':>4} | {'TBPTT(1)':>10} | {'RTRL':>10}")
    print("-" * 32)
    for L in L_SWEEP:
        r_tbptt = run_one(L, use_rtrl=False)
        r_rtrl = run_one(L, use_rtrl=True)
        print(f"{L:>4} | {r_tbptt:>10.3f} | {r_rtrl:>10.3f}")


if __name__ == "__main__":
    main()
