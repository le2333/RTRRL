"""Two endings, told apart, and the observation the episode actually ended in.

An episode ends either because the task failed or because the clock ran out.
Both end it, and only one says the future is worth nothing: a policy cut off at
its step limit was about to go on earning. Conflating them teaches the critic
that reaching the limit is as bad as falling over.

The wiring is what is under test here rather than any environment's dynamics,
so most of it runs against a scripted stand-in. One test runs against brax,
because whether its ``truncation`` flag survives our wrapper is a fact about
brax and cannot be asserted against a stand-in.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp

from memorax.environments import make
from memorax.environments.brax import BraxGymnaxWrapper

RESET = jnp.asarray([0.0, 0.0])
ENDED_IN = jnp.asarray([7.0, 9.0])


@dataclass
class Stepped:
    """One transition an environment is scripted to produce."""

    obs: Any
    done: bool
    truncation: bool


@dataclass
class FakeState:
    obs: Any
    reward: Any
    done: Any
    pipeline_state: Any
    info: dict = field(default_factory=dict)

    def replace(self, **changes):
        return replace(self, **changes)


class FakeEpisodes:
    """A brax ``EpisodeWrapper``'s shape, scripted rather than simulated."""

    action_size = 1
    observation_size = 2

    def __init__(self, script: list[Stepped]) -> None:
        self._script = list(script)
        self._taken = 0

    def reset(self, key):
        del key
        return FakeState(
            obs=RESET,
            reward=jnp.asarray(0.0),
            done=jnp.asarray(False),
            pipeline_state=RESET,
            info={"steps": jnp.asarray(0), "truncation": jnp.asarray(False)},
        )

    def step(self, state, action):
        del action
        taken = self._script[self._taken]
        self._taken += 1
        return state.replace(
            obs=taken.obs,
            reward=jnp.asarray(1.0),
            done=jnp.asarray(taken.done),
            pipeline_state=taken.obs,
            info={**state.info, "truncation": jnp.asarray(taken.truncation)},
        )


def one_step(*, done: bool, truncation: bool):
    env = BraxGymnaxWrapper(
        FakeEpisodes([Stepped(obs=ENDED_IN, done=done, truncation=truncation)])
    )
    params = env.default_params
    _, state = env.reset(jax.random.key(0), params)
    return env.step(jax.random.key(1), state, jnp.zeros((1,)), params)


def test_the_clock_running_out_is_a_truncation_and_not_a_termination():
    _, _, _, done, info = one_step(done=True, truncation=True)

    assert bool(done)
    assert bool(info["truncation"])
    assert not bool(info["terminal"])


def test_the_task_failing_is_a_termination_and_not_a_truncation():
    _, _, _, done, info = one_step(done=True, truncation=False)

    assert bool(done)
    assert bool(info["terminal"])
    assert not bool(info["truncation"])


def test_a_step_that_ends_nothing_is_neither():
    _, _, _, done, info = one_step(done=False, truncation=False)

    assert not bool(done)
    assert not bool(info["terminal"])
    assert not bool(info["truncation"])


def test_the_step_hands_back_the_state_its_episode_ended_in():
    """The wrapper resets nothing, so there is only one observation to hand back.

    Brax's auto-reset would have replaced this with the next episode's opening,
    which is a different state and valuing it is a different question. Starting
    the next episode belongs to whoever is about to act.
    """

    obs, _, _, done, _ = one_step(done=True, truncation=True)

    assert bool(done)
    assert obs.tolist() == ENDED_IN.tolist()


def test_brax_reports_its_own_step_limit_as_a_truncation():
    """The limit is the task's, so it comes from the environment section.

    The same policy's return under a limit of 500 and of 1000 is not the same
    number, which is why a literal in a wrapper was the wrong place for it.
    """

    env, params = make("brax::hopper", backend="spring", episode_length=3)
    obs, state = jax.jit(env.reset)(jax.random.key(0), params)
    action = jnp.zeros(env.action_space(params).shape)
    step = jax.jit(env.step)
    for _ in range(2):
        obs, state, _, done, info = step(jax.random.key(1), state, action, params)
        assert not bool(done)

    _, _, _, done, info = step(jax.random.key(1), state, action, params)
    assert bool(done)
    assert bool(info["truncation"])
    assert not bool(info["terminal"])


def test_the_mask_reaches_what_the_step_hands_back():
    """A masked task is masked everywhere or the critic sees what the actor cannot."""

    env, params = make(
        "brax::hopper", backend="spring", observed=(0, 2), episode_length=3
    )
    _, state = jax.jit(env.reset)(jax.random.key(0), params)
    action = jnp.zeros(env.action_space(params).shape)
    obs, _, _, _, _ = jax.jit(env.step)(jax.random.key(1), state, action, params)

    assert obs.shape == (2,)
