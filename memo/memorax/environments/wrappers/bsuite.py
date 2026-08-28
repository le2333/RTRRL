"""What a bsuite episode cost, and which of the two ways it ended.

A bsuite task is scored on regret rather than on return, and regret is not a
quantity a wrapper can infer from the outside: it is the return the best
policy would have had, minus this one's, and only the task knows the first
half. The base class keeps the reading that suited the tasks whose failure is
a terminal penalty, and each task that computes it differently says so.

The second thing each task knows is what its own ending means. An episode ends
either because the task finished or because the clock ran out, and only the
first says the future is worth nothing -- ``td0`` gates the bootstrap on
``terminal``, so an agent told that a step limit is a terminal state is being
taught that reaching the limit is as bad as failing. Gymnax reports one flag
for both, so the distinction is drawn here, per task, from what that task's
own parameters say. It is drawn only for the environments below, because only
they can draw it from a task's own numbers; a gymnax environment outside bsuite
is answered by ``EpisodeEndingWrapper``, which reads a step limit as a clock
only where the adapter has named that environment as having one.

Neither reading consults the environment's state after a step, and both are
kept that way now that one could. Gymnax's base ``step`` resets on the
transition that ended the episode, so the state it handed back on that step was
the *next* episode's opening -- a wrapper that read ``time`` there to decide
whether the chain had completed would read zero, and would report every ending
as a truncation without ever raising. ``EpisodeEndingWrapper`` sits underneath
this one and takes the transition from ``step_env``, so that state is no longer
rewound; the counting here stays where it is because it is right either way,
and a reading that does not depend on how it is stacked is the cheaper one to
keep.
"""

import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment
from gymnax.wrappers.purerl import GymnaxWrapper

from memorax.utils.typing import Array, Key


@struct.dataclass
class BSuiteEnvState:
    env_state: environment.EnvState
    episode_return: float
    episode_regret: float
    returned_episode_regret: float
    # Transitions since this episode opened, which the wrapper counts because
    # the environment's own count is rewound by the auto-reset. ``timestep``
    # below counts the whole stream and is never rewound.
    episode_steps: int
    timestep: int


class BSuiteWrapper(GymnaxWrapper):
    """Regret and episode endings for a bsuite task, in the general case."""

    # Whether this task can say which of the two endings it just had. False
    # here, and the ``terminal``/``truncation`` pair is then left out of the
    # step's info rather than filled in with a guess: ``terminal_of`` already
    # falls back to ``done`` when the key is absent, so saying nothing costs
    # nothing, while writing ``truncation=False`` for a task whose step limit
    # really is a truncation would be a claim no one here is able to make.
    distinguishes_truncation = False

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def episode_regret(self, episode_return, reward, terminal, params) -> Array:
        """What this episode cost against the best one, read when it ends.

        The general reading, which suits a task whose only failure is a
        terminal penalty: ending in one costs two units of return against the
        episode that did not.
        """

        del episode_return, terminal, params
        return jnp.where(reward < 0, 2.0, 0.0)

    def terminal(self, episode_steps, done, params) -> Array:
        """Whether the episode finished, as opposed to being cut off.

        Read only where ``distinguishes_truncation`` says the task can tell,
        because gymnax reports one flag and a wrapper that guessed would be
        guessing about every environment at once.
        """

        del episode_steps, params
        return done

    def reset(self, key: Key, params) -> tuple[Array, BSuiteEnvState]:
        obs, env_state = self._env.reset(key, params)

        state = BSuiteEnvState(
            env_state=env_state,
            episode_return=0.0,
            episode_regret=0.0,
            returned_episode_regret=0.0,
            episode_steps=0,
            timestep=0,
        )
        return obs, state

    def step(
        self, key: Key, state, action: Array, params
    ) -> tuple[Array, BSuiteEnvState, Array, Array, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )

        # UmbrellaChain accumulates its reward from integer terms and hands
        # back int32, so the running return is made a float before it is one.
        reward = jnp.asarray(reward, dtype=jnp.float32)
        episode_return = state.episode_return + reward
        episode_steps = state.episode_steps + 1

        # Computed either way: a task that distinguishes its endings uses it
        # for regret as well, and one that does not is handed ``done``, which
        # is what the reward-sign reading would have seen anyway.
        terminal = done & jnp.asarray(
            self.terminal(episode_steps, done, params), dtype=jnp.bool_
        )

        # Regret is an episode's quantity, so it is read at the end of one and
        # is zero on every step that ends nothing.
        step_regret = jnp.where(
            done,
            self.episode_regret(episode_return, reward, terminal, params),
            0.0,
        )

        episode_regret = state.episode_regret + step_regret

        returned_episode_regret = jnp.where(done, episode_regret, 0.0)

        new_state = BSuiteEnvState(
            env_state=env_state,
            episode_return=episode_return * (1 - done),
            episode_regret=episode_regret * (1 - done),
            returned_episode_regret=returned_episode_regret,
            episode_steps=episode_steps * (1 - done),
            timestep=state.timestep + 1,
        )

        info["returned_episode_regret"] = returned_episode_regret
        info["returned_episode"] = done
        info["timestep"] = new_state.timestep

        info["step_regret"] = step_regret
        info["total_regret"] = episode_regret

        if self.distinguishes_truncation:
            info["terminal"] = terminal
            info["truncation"] = done & ~terminal

        return obs, new_state, reward, done, info


class DiscountingChainWrapper(BSuiteWrapper):
    """The chain that is only ever ended by its clock.

    Its five actions each pay once, at t=1, 3, 10, 30 and 100, and the best of
    them pays ``optimal_return`` where the rest pay 1.0. Nothing about the task
    can end an episode early, so every ending is the step limit -- which is why
    ``terminal`` is reported and always false, and why the critic that keeps
    bootstrapping at the limit is the one measuring what this task is for.

    Its regret is the return it gave up rather than a penalty it was charged:
    this task never pays a negative reward, so the base class's reading of one
    would report every episode, optimal or not, as costing nothing.
    """

    distinguishes_truncation = True

    def episode_regret(self, episode_return, reward, terminal, params) -> Array:
        del reward, terminal
        return params.optimal_return - episode_return

    def terminal(self, episode_steps, done, params) -> Array:
        del episode_steps, params
        return jnp.zeros_like(done)


class UmbrellaChainWrapper(BSuiteWrapper):
    """The chain whose ending is its own, and whose regret is one decision's.

    One decision at the first step is paid at the last, with uninformative
    rewards in between; reaching the end of the chain is the task finishing,
    and is a termination. A step limit shorter than the chain would cut it off
    instead, which is a truncation, and the two are told apart by the chain
    length rather than by which flag gymnax raised.

    The environment keeps a ``total_regret`` of its own, which would be the
    obvious thing to read. It was unreadable while gymnax's reset rewound the
    ending step's state to the next episode's zero, and is readable again now
    that ``EpisodeEndingWrapper`` takes the reset off; the decision is still
    scored by the last step's reward, because moving to the environment's
    counter would change what this wrapper reports and is a question about
    regret rather than about endings.
    """

    distinguishes_truncation = True

    def episode_regret(self, episode_return, reward, terminal, params) -> Array:
        del episode_return, params
        # Only the last step of a full chain pays for the umbrella decision --
        # every earlier reward is the distractor coin. An episode the clock cut
        # off never scored the decision, so it has no regret to report rather
        # than the sign of whichever coin it happened to land on.
        return jnp.where(terminal, jnp.where(reward < 0, 2.0, 0.0), 0.0)

    def terminal(self, episode_steps, done, params) -> Array:
        del done
        return episode_steps >= params.chain_length


_BY_ENVIRONMENT = {
    "DiscountingChain": DiscountingChainWrapper,
    "UmbrellaChain": UmbrellaChainWrapper,
}


def bsuite_wrapper_for(env_id: str) -> type[BSuiteWrapper]:
    """The wrapper that knows this task, or the one that assumes the least."""

    for name, wrapper in _BY_ENVIRONMENT.items():
        if name in env_id:
            return wrapper
    return BSuiteWrapper
