"""What a gymnax ending reports, for the two things a learner reads off it.

An ending carries two facts and gymnax publishes neither: the state the
transition reached, which its base ``step`` overwrites with the next episode's
opening, and whether the episode failed or merely ran out of clock, which it
folds into one ``done``. A TD target needs both -- it values the successor, and
it gates the bootstrap on ``terminal`` -- so both are held here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from memorax.environments import make
from memorax.environments.gymnax import EXTERNAL_EPISODE_LIMIT
from memorax.rl.interaction import terminal_of

pytest.importorskip("gymnax")

CARTPOLE = "gymnax::CartPole-v1"
UMBRELLA = "gymnax::UmbrellaChain-bsuite"

# Gymnax opens CartPole uniformly on (-0.05, 0.05) in every coordinate, so a
# state outside that box on any coordinate cannot be a reset.
RESET_BOUND = 0.05


def balancing(observation):
    """Push the cart the way the pole is leaning, which holds it up for a while.

    A policy is needed here only so that the episode reaches its step limit with
    the pole still up; that is the ending under test, and a random one would
    reach it only sometimes.
    """

    return int(np.asarray(observation)[2] > 0)


def toppling(observation):
    """One direction, forever, which drops the pole in about ten steps."""

    del observation
    return 0


def ending(env_id, *, episode_length, action, limit=None):
    """Run one episode and report the step that ended it, and how."""

    environment, parameters = make(
        env_id, observed=None, backend=None, episode_length=episode_length
    )
    observation, state = environment.reset(jax.random.key(0), parameters)
    for step in range(1, (limit or episode_length) + 1):
        observation, state, reward, done, info = environment.step(
            jax.random.key(step), state, jnp.int32(action(observation)), parameters
        )
        if done:
            return step, observation, info
    raise AssertionError(f"{env_id} did not end within {limit or episode_length} steps")


def test_a_pole_still_up_at_the_step_limit_is_a_truncation():
    """The clock is not the task, so the future it stopped is worth valuing."""

    steps, _, info = ending(CARTPOLE, episode_length=32, action=balancing)

    assert steps == 32
    assert not bool(terminal_of(info, True))
    assert bool(info["truncation"])


def test_a_fallen_pole_is_a_termination_however_much_clock_was_left():
    steps, _, info = ending(CARTPOLE, episode_length=500, action=toppling, limit=64)

    assert steps < 500
    assert bool(terminal_of(info, True))
    assert not bool(info["truncation"])


def test_the_ending_reports_the_state_the_transition_reached():
    """Not the next episode's opening, which is what gymnax's reset put there.

    A learner stores this as ``next_observation`` and values it under the target
    network. Reading the reset instead would value a state the transition never
    entered, which is only invisible while ``terminal`` masks every bootstrap.
    """

    _, reached, _ = ending(CARTPOLE, episode_length=32, action=balancing)

    assert np.max(np.abs(np.asarray(reached))) > RESET_BOUND


def test_a_terminal_ending_also_reports_the_state_it_reached():
    """The pole that fell is past its threshold, and a reset never is."""

    _, reached, _ = ending(CARTPOLE, episode_length=500, action=toppling, limit=64)

    # Gymnax fails CartPole at |theta| > 12 degrees, which is the third
    # coordinate and an order of magnitude outside the reset box.
    assert abs(float(np.asarray(reached)[2])) > RESET_BOUND


def test_an_environment_whose_limit_is_its_task_reports_no_truncation_here():
    """UmbrellaChain ends when its chain does, and says so for itself.

    The chain's own wrapper answers this ending; nothing in the episode-ending
    layer may overrule it, because a task whose horizon *is* the task would be
    taught by a truncation reading to bootstrap past an ending with no past.
    """

    steps, _, info = ending(UMBRELLA, episode_length=100, action=lambda observation: 0)

    assert steps == 10
    assert bool(terminal_of(info, True))
    assert not bool(info["truncation"])


def test_the_environments_that_read_their_limit_as_a_clock_are_named():
    """Declared rather than inferred; see the list's own comment for why."""

    assert "CartPole-v1" in EXTERNAL_EPISODE_LIMIT
    assert not any("bsuite" in name for name in EXTERNAL_EPISODE_LIMIT)
