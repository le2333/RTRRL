"""MemoryChain diagnostic env (bsuite, osband2020bsuite), gymnax-style.

Presents an informative binary cue at t=0, uninformative observations for
L-1 steps, and rewards the agent for reproducing the cue at t=L. Sweeping L
isolates the temporal credit-assignment problem from representation learning
(arxiv 2605.24709, Section 4.1).

Observation: [bit, is_first] float32 (bit only informative at t=0).
Action: Discrete(2) — predict the cue bit.
Reward: +1 at the final step if action == cue, else 0.
"""
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces

from memorax.utils.typing import Array, Key


@struct.dataclass(frozen=True)
class MemoryChainState:
    bit: Array
    t: Array
    L: Array


@struct.dataclass(frozen=True)
class MemoryChainParams:
    L: int


class MemoryChain:
    """gymnax-interface MemoryChain. Episode length = L steps."""

    def __init__(self, L: int = 16):
        self.L = L

    @property
    def default_params(self) -> MemoryChainParams:
        return MemoryChainParams(L=self.L)

    def reset(self, key: Key, params: MemoryChainParams) -> tuple[Array, MemoryChainState]:
        bit = jax.random.bernoulli(key).astype(jnp.int32)
        obs = jnp.stack([bit.astype(jnp.float32), jnp.float32(1.0)])
        state = MemoryChainState(
            bit=bit, t=jnp.int32(0), L=jnp.int32(params.L)
        )
        return obs, state

    def step(
        self, key: Key, state: MemoryChainState, action: Array, params: MemoryChainParams
    ) -> tuple[Array, MemoryChainState, Array, Array, dict]:
        obs, new_state, reward, terminated, _, info = self.trace_step(
            key, state, action, params
        )
        return obs, new_state, reward, terminated, info

    def trace_step(
        self, key: Key, state: MemoryChainState, action: Array, params: MemoryChainParams
    ) -> tuple[Array, MemoryChainState, Array, Array, Array, dict]:
        """Step without auto-reset and expose natural termination explicitly."""

        del key, params
        L = state.L
        t = state.t
        predict_step = t == (L - 1)
        correct = (action == state.bit).astype(jnp.float32)
        reward = jnp.where(predict_step, correct, jnp.float32(0.0))
        done = predict_step
        new_t = jnp.where(predict_step, t, t + 1)
        obs = jnp.stack([jnp.float32(0.0), jnp.float32(0.0)])
        new_state = MemoryChainState(bit=state.bit, t=new_t, L=L)
        return obs, new_state, reward, done, jnp.asarray(False), {}

    def observation_space(self, params: MemoryChainParams) -> spaces.Box:
        return spaces.Box(low=-1.0, high=1.0, shape=(2,))

    def action_space(self, params: MemoryChainParams) -> spaces.Discrete:
        return spaces.Discrete(2)
