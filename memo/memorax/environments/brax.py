from typing import Any

import jax
import jax.numpy as jnp
from gymnax.environments import EnvParams, spaces

from memorax.environments.wrappers import GymnaxWrapper, SelectObservationWrapper
from memorax.utils.typing import Array, Key


class BraxGymnaxWrapper(GymnaxWrapper):
    """Brax behind the gymnax interface, keeping what its auto-reset discards.

    Brax's own ``AutoResetWrapper`` overwrites the observation the episode ended
    in with the one the next episode starts from, and our old wrapper threw away
    the ``truncation`` flag underneath it. Both are needed: an episode cut off at
    its step limit was about to go on earning, so the value of where it stopped
    is the best statement anyone has about the rest, and that value is a question
    about an observation that no longer exists by the time the caller sees one.

    So the auto-reset is done here rather than below, on the same stored first
    state brax uses, and both observations come back.
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
        state.info["first_pipeline_state"] = state.pipeline_state
        state.info["first_obs"] = state.obs
        return state.obs, state

    def step(
        self, key: Key, state, action: Array, params
    ) -> tuple[Array, Any, Array, Array, dict]:
        if "steps" in state.info:
            steps = state.info["steps"]
            state.info.update(
                steps=jnp.where(state.done, jnp.zeros_like(steps), steps)
            )
        next_state = self._env.step(state.replace(done=jnp.zeros_like(state.done)), action)

        done = next_state.done.astype(jnp.bool)
        truncation = jnp.asarray(next_state.info["truncation"]).astype(jnp.bool)
        ended_in = next_state.obs

        def restored(first, running):
            mask = done
            if mask.shape:
                mask = jnp.reshape(mask, [running.shape[0]] + [1] * (running.ndim - 1))
            return jnp.where(mask, first, running)

        next_state = next_state.replace(
            pipeline_state=jax.tree.map(
                restored,
                next_state.info["first_pipeline_state"],
                next_state.pipeline_state,
            ),
            obs=restored(next_state.info["first_obs"], next_state.obs),
        )
        return (
            next_state.obs,
            next_state,
            next_state.reward,
            done,
            {
                "terminal": done & ~truncation,
                "truncation": truncation,
                "next_observation": ended_in,
            },
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
