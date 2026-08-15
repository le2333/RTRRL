"""The Gymnax adapter against the three fields assembly always supplies."""

from __future__ import annotations

import jax
import pytest
from gymnax.environments import spaces

from memorax.environments import make

pytest.importorskip("gymnax")

CARTPOLE = "gymnax::CartPole-v1"


def test_the_factory_accepts_every_deployment_field_a_run_declares():
    """Assembly passes all three whether or not this backend has use for them."""

    environment, parameters = make(
        CARTPOLE,
        observed=None,
        backend=None,
        episode_length=32,
    )

    assert isinstance(environment.action_space(parameters), spaces.Discrete)
    assert environment.action_space(parameters).n == 2
    assert parameters.max_steps_in_episode == 32


def test_the_declared_episode_length_replaces_the_environment_default():
    _, default = make(CARTPOLE, observed=None, backend=None)
    _, shortened = make(CARTPOLE, observed=None, backend=None, episode_length=7)

    assert default.max_steps_in_episode != 7
    assert shortened.max_steps_in_episode == 7


def test_an_observed_selection_narrows_what_the_environment_returns():
    full, full_parameters = make(CARTPOLE, observed=None, backend=None)
    narrowed, narrowed_parameters = make(CARTPOLE, observed=[0, 2], backend=None)

    assert full.observation_space(full_parameters).shape == (4,)
    assert narrowed.observation_space(narrowed_parameters).shape == (2,)

    observation, _ = narrowed.reset(jax.random.key(0), narrowed_parameters)
    assert observation.shape == (2,)


def test_the_backend_a_gymnax_run_declares_reaches_no_constructor():
    """One implementation per environment, so naming a backend changes nothing."""

    named, _ = make(CARTPOLE, observed=None, backend="spring", episode_length=9)
    unnamed, _ = make(CARTPOLE, observed=None, backend=None, episode_length=9)

    assert type(named) is type(unnamed)
