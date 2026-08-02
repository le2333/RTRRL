from typing import Any

import jax.numpy as jnp
from gymnax.environments import EnvParams, spaces

from memorax.environments.wrappers import GymnaxWrapper, SelectObservationWrapper
from memorax.utils.typing import Array, Key


class BraxGymnaxWrapper(GymnaxWrapper):
    """Brax behind the gymnax interface, resetting nothing.

    Brax's ``AutoResetWrapper`` hands back the state the next episode starts
    from and destroys the one it ended in, which is the one a bootstrap has to
    value. Restarting belongs to whoever is about to act, not to the step that
    just ended, so it happens in the algorithm and this stays an adapter.
    """

    def __init__(self, env, episode_length: int = 1000):
        super().__init__(env)
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)
        self.episode_length = episode_length

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=self.episode_length)

    def reset(self, key: Key, params) -> tuple[Array, Any]:
        state = self._env.reset(key)
        return state.obs, state

    def step(
        self, key: Key, state, action: Array, params
    ) -> tuple[Array, Any, Array, Array, dict]:
        next_state = self._env.step(state, action)
        done = next_state.done.astype(jnp.bool)
        truncation = jnp.asarray(next_state.info["truncation"]).astype(jnp.bool)
        return (
            next_state.obs,
            next_state,
            next_state.reward,
            done,
            {"terminal": done & ~truncation, "truncation": truncation},
        )

    def observation_space(self, params) -> spaces.Box:
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size,),
        )

    def action_space(self, params) -> spaces.Box:
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )


def make(
    env_id: str,
    observed=None,
    backend="generalized",
    episode_length: int = 1000,
    **kwargs,
) -> tuple:
    from brax import envs
    from brax.envs.wrappers.training import EpisodeWrapper

    env = envs.get_environment(env_name=env_id, backend=backend, **kwargs)
    env = EpisodeWrapper(env, episode_length=episode_length, action_repeat=1)
    env = BraxGymnaxWrapper(env, episode_length=episode_length)
    if observed is not None:
        env = SelectObservationWrapper(env, observed)

    env_params = env.default_params
    return env, env_params
