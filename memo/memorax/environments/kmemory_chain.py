"""KMemoryChain diagnostic env (arXiv 2605.24709 §4.4), gymnax-style.

Generalises memory_chain.MemoryChain from a single cue bit to K cue bits that
must all be recalled after L uninformative steps. Sweeping (K, L) probes
staleness of the RTRL credit-assignment signal as both the memory load (K) and
the delay (L) grow. The exact reward shaping is inferred from the paper's
description (partial credit = fraction of correctly recalled bits); confirm
against §4.4 once the paper text is available.

Observation: [bit_0, ..., bit_{K-1}, is_first] float32 (bits informative only
  at t=0; zero otherwise).
Action: Discrete(2**K) — a K-bit prediction (integer 0..2**K-1).
Reward: at the final step, fraction of recalled bits that match the cue
  (0..1); 0 on all non-final steps.
"""
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces

from memorax.utils.typing import Array, Key


@struct.dataclass(frozen=True)
class KMemoryChainState:
    bits: Array        # (K,) int32 cue bits
    t: Array
    L: Array


@struct.dataclass(frozen=True)
class KMemoryChainParams:
    L: int
    K: int


class KMemoryChain:
    """gymnax-interface KMemoryChain. Episode length = L steps."""

    def __init__(self, L: int = 16, K: int = 4):
        self.L = L
        self.K = K

    @property
    def default_params(self) -> KMemoryChainParams:
        return KMemoryChainParams(L=self.L, K=self.K)

    def reset(self, key: Key, params: KMemoryChainParams) -> tuple[Array, KMemoryChainState]:
        bits = jax.random.bernoulli(key, shape=(params.K,)).astype(jnp.int32)
        obs = jnp.concatenate([bits.astype(jnp.float32), jnp.array([1.0], dtype=jnp.float32)])
        state = KMemoryChainState(bits=bits, t=jnp.int32(0), L=jnp.int32(params.L))
        return obs, state

    def step(
        self, key: Key, state: KMemoryChainState, action: Array, params: KMemoryChainParams
    ) -> tuple[Array, KMemoryChainState, Array, Array, dict]:
        L = state.L
        K = params.K
        t = state.t
        predict_step = t == (L - 1)
        # Decode the K-bit prediction from the integer action.
        action_bits = jax.vmap(lambda i: (action >> i) & 1)(jnp.arange(K))
        correct_bits = jnp.sum((action_bits == state.bits).astype(jnp.int32))
        reward = jnp.where(predict_step, correct_bits.astype(jnp.float32) / K, jnp.float32(0.0))
        done = predict_step
        new_t = jnp.where(predict_step, t, t + 1)
        obs = jnp.concatenate([
            jnp.zeros(K, dtype=jnp.float32),
            jnp.array([0.0], dtype=jnp.float32),
        ])
        new_state = KMemoryChainState(bits=state.bits, t=new_t, L=L)
        return obs, new_state, reward, done, {}

    def observation_space(self, params: KMemoryChainParams) -> spaces.Box:
        return spaces.Box(low=-1.0, high=1.0, shape=(params.K + 1,))

    def action_space(self, params: KMemoryChainParams) -> spaces.Discrete:
        return spaces.Discrete(2 ** params.K)
