"""Gymnax behind the deployment's environment vocabulary.

Assembly always supplies ``observed``, ``backend`` and ``episode_length``,
because a run document always declares them. Only the first and third mean
anything here: gymnax ships one implementation per environment, so there is no
backend to choose between.
"""

import gymnax
from gymnax.environments import environment

from memorax.environments.wrappers import BSuiteWrapper, SelectObservationWrapper


def make(
    env_id: str,
    observed=None,
    backend=None,
    episode_length: int = 1000,
    **kwargs,
) -> tuple[environment.Environment, environment.EnvParams]:
    del backend

    env, env_params = gymnax.make(env_id, **kwargs)

    if "bsuite" in env_id:
        env = BSuiteWrapper(env)

    if observed is not None:
        env = SelectObservationWrapper(env, observed)

    return env, env_params.replace(max_steps_in_episode=episode_length)
