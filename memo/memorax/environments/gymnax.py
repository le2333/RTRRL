"""Gymnax behind the deployment's environment vocabulary.

Assembly always supplies ``observed``, ``backend`` and ``episode_length``,
because a run document always declares them. Only the first and third mean
anything here: gymnax ships one implementation per environment, so there is no
backend to choose between.

Anything else a run names reaches this module as ``kwargs``, and gymnax splits
it in two places that a caller cannot see from outside. Some arguments belong
to the environment's constructor -- UmbrellaChain's ``n_distractor``,
DiscountingChain's ``mapping_seed`` -- and some belong to its ``EnvParams`` --
UmbrellaChain's ``chain_length``. A run document should not have to know which
is which, so the split is made here from the parameter dataclass's own fields.
"""

import dataclasses

import gymnax
import jax
from gymnax.environments import environment

from memorax.environments.wrappers import (
    EpisodeEndingWrapper,
    SelectObservationWrapper,
    bsuite_wrapper_for,
)

# The environments whose ``max_steps_in_episode`` is a clock imposed on the task
# rather than the horizon the task is defined over. Classic control is the whole
# of the list: a pole falls, a car reaches its flag, a pendulum never finishes at
# all, and in each of them the step limit is somebody's convenience rather than a
# statement about the problem. A bsuite chain is the other case and says so for
# itself -- ``bsuite_wrapper_for`` hands the two tasks that know their own ending
# a wrapper that reports it.
#
# Declared rather than inferred, because from inside a step the two cases are the
# same comparison against the same field. An environment whose only ending is its
# clock is indistinguishable from one whose clock cut a task short, and guessing
# wrong in that direction has TD bootstrap past an ending with nothing after it.
# So the default is to say nothing and leave ``terminal_of`` reading ``done``,
# which is what every gymnax run did before this list existed.
EXTERNAL_EPISODE_LIMIT = frozenset(
    {
        "Acrobot-v1",
        "CartPole-v1",
        "MountainCar-v0",
        "MountainCarContinuous-v0",
        "Pendulum-v1",
    }
)


def _split(kwargs: dict, parameters) -> tuple[dict, dict]:
    """Separate what builds the environment from what parameterises it.

    The parameter dataclass is the authority on the second half, so a name it
    declares is a parameter and everything else is handed to the constructor.
    A name that is neither raises from the constructor, which is the error
    that names the environment that would not take it.
    """

    declared = {field.name for field in dataclasses.fields(parameters)}
    return (
        {name: value for name, value in kwargs.items() if name not in declared},
        {name: value for name, value in kwargs.items() if name in declared},
    )


def make(
    env_id: str,
    observed=None,
    backend=None,
    episode_length: int = 1000,
    **kwargs,
) -> tuple[environment.Environment, environment.EnvParams]:
    del backend

    if "max_steps_in_episode" in kwargs:
        raise ValueError(
            "max_steps_in_episode is episode_length under another name; a run "
            "declares the horizon once, in the environment's episode_length"
        )

    # Built once to learn the split, and again only if a constructor argument
    # was named -- a constructor argument cannot be applied any later than
    # construction. Nothing in this repository's environments makes that
    # second build expensive; MNISTBandit, which reads a dataset, would.
    env, env_params = gymnax.make(env_id)
    constructor, declared = _split(kwargs, env_params)
    if constructor:
        env, env_params = gymnax.make(env_id, **constructor)
    if declared:
        env_params = env_params.replace(**declared)

    env_params = env_params.replace(max_steps_in_episode=episode_length)

    # Underneath everything else: the wrappers above read what a step returns,
    # and what a step returns is only the transition once gymnax's reset has
    # been taken off it.
    env = EpisodeEndingWrapper(env, external_limit=env_id in EXTERNAL_EPISODE_LIMIT)

    if "bsuite" in env_id:
        env = bsuite_wrapper_for(env_id)(env)
        _refuse_a_horizon_that_redefines_the_task(env_id, env_params)

    if observed is not None:
        env = SelectObservationWrapper(env, observed)

    return env, env_params


def _refuse_a_horizon_that_redefines_the_task(env_id: str, params) -> None:
    """Refuse an episode limit that would delete part of a bsuite task.

    DiscountingChain pays each action once, at its own timestep, and the whole
    of the task is that the payments are spread far enough apart for a
    discount rate to choose between them. An episode limit shorter than the
    last payment does not shorten the task, it removes actions from it: at
    ``episode_length`` 20 the rewards at t=30 and t=100 never arrive and two
    of the five actions become worth nothing. That is a different task wearing
    this one's name, and it is quieter to refuse it here than to read it out
    of a curve later.
    """

    if "DiscountingChain" not in env_id:
        return
    # The ensemble entry assembles once before ``vmap`` and again while tracing
    # each swept member. The concrete preflight build retains task validation;
    # repeating it over traced parameters only fails compilation.
    if isinstance(params.reward_timestep, jax.core.Tracer):
        return
    last_payment = int(max(params.reward_timestep))
    if params.max_steps_in_episode < last_payment:
        raise ValueError(
            f"episode_length {params.max_steps_in_episode} is shorter than "
            f"DiscountingChain's last reward at t={last_payment}; the actions "
            "paid after it would be unreachable and the task would not be the "
            "one it is named after"
        )
