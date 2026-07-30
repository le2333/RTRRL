from __future__ import annotations

from typing import Union

import jax.numpy as jnp
from gymnax.environments import environment, spaces
from gymnax.wrappers.purerl import GymnaxWrapper

from memorax.utils.typing import Array, Key


class SelectObservationWrapper(GymnaxWrapper):
    def __init__(self, env, observed):
        super().__init__(env)
        self.observed = jnp.asarray(observed, dtype=jnp.int32)

    def observation_space(self, params=None) -> spaces.Box:
        inner = self._env.observation_space(params)
        return spaces.Box(
            low=inner.low,
            high=inner.high,
            shape=(int(self.observed.size),),
            dtype=inner.dtype,
        )

    def reset(
        self, key: Key, params: environment.EnvParams | None = None
    ) -> tuple[Array, environment.EnvState]:
        observation, state = self._env.reset(key, params)
        return observation[..., self.observed], state

    def step(
        self,
        key: Key,
        state: environment.EnvState,
        action: Union[int, float],
        params: environment.EnvParams | None = None,
    ) -> tuple[Array, environment.EnvState, float, bool, dict]:
        observation, state, reward, done, info = self._env.step(
            key, state, action, params
        )
        return observation[..., self.observed], state, reward, done, info
