"""What an ending reports: the state it reached, and why it happened.

Gymnax's ``Environment.step`` answers an ending with the *next* episode's first
observation -- it resets on top of the transition it just took -- and folds the
step limit into the same ``done`` as the environment's own failure. Both are the
right shape for a loop that keeps no successor and bootstraps nothing, and the
wrong one for a learner whose target is ``r + gamma (1 - terminal) V(s')``.

Two facts are lost there and this wrapper restores them.

**The state the transition reached.** ``step_env`` is the transition; the reset
gymnax performs after it belongs to the next episode. A learner that stores
``next_observation`` and values it stores the reset instead, and is saved from
reading it only by the second fact also being wrong -- ``terminal`` masking the
bootstrap everywhere. Repair one without the other and the successor of a
cut-off ending becomes a state the transition never entered, which is worse than
either alone. Nothing here resets: every algorithm in this repository opens the
next episode itself, at the top of its own step, so gymnax's reset was redundant
as well as destructive.

**Why the episode ended.** An episode that failed has no future to value; one
stopped by a clock has the future it was about to go on earning, and TD has to
tell the two apart. Gymnax publishes no truncation flag, so the environment's
own ending is read by asking ``is_terminal`` again with the clock taken out of
it -- the limit moved past anything a run can reach. What answers is the
condition the task declares, and an ending the task did not declare is a
truncation.

That reading is only correct where the limit is *external* to the task, which is
why it is asked for by name rather than applied to everything gymnax registers.
``UmbrellaChain`` ends at ``time == chain_length`` and ``DiscountingChain`` at
its horizon: those limits are the tasks, not a clock imposed on one, and calling
either a truncation would have TD bootstrap past an ending that has no past.
Where ``external_limit`` is not declared this reports no ``terminal`` at all,
which leaves ``terminal_of`` reading ``done`` -- the same answer as before this
wrapper existed, and the same answer the fixed-horizon tasks want.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.wrappers.purerl import GymnaxWrapper

from memorax.utils.typing import Array, Key

# Past anything a run can reach, so that ``is_terminal`` asked with this limit
# answers with the task's own ending and nothing else. An integer rather than an
# infinity, because the field it replaces is compared against a step count.
NO_LIMIT = jnp.iinfo(jnp.int32).max


class EpisodeEndingWrapper(GymnaxWrapper):
    """Report the observation an ending reached, and whether it was a truncation.

    ``external_limit`` says that ``max_steps_in_episode`` is a limit imposed on
    the task rather than the horizon the task is defined over. Only an adapter
    that knows the environment may say so; see the module docstring for why the
    two cases cannot be told apart from inside a step.
    """

    def __init__(self, env, *, external_limit: bool = False) -> None:
        super().__init__(env)
        self.external_limit = external_limit

    def step(
        self,
        key: Key,
        state: environment.EnvState,
        action: Array,
        params: environment.EnvParams | None = None,
    ) -> tuple[Array, environment.EnvState, Array, Array, dict[str, Any]]:
        # Gymnax's own signature, where an unnamed parameter set means the
        # environment's default one; ``is_terminal`` below has to be asked
        # against the same parameters the step was taken under.
        limits = self._env.default_params if params is None else params
        observation, env_state, reward, done, info = self._env.step_env(
            key, state, action, limits
        )
        if not self.external_limit:
            return observation, env_state, reward, done, info

        terminal = jnp.asarray(
            self._env.is_terminal(
                env_state,
                dataclasses.replace(limits, max_steps_in_episode=NO_LIMIT),
            ),
            dtype=jnp.bool_,
        )
        return (
            observation,
            env_state,
            reward,
            done,
            {
                **info,
                "terminal": terminal,
                "truncation": jnp.asarray(done, dtype=jnp.bool_) & ~terminal,
            },
        )
