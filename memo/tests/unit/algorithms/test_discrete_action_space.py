"""RTRRL and StreamAC against an action space that names its actions.

R2D2 needs a discrete action space and these two needed a continuous one, so
until the categorical head was registered there was no environment any two of
the three could both be built against -- and the comparison the platform exists
to make had nothing to run on.

What actually differed was one shape, not one distribution. A continuous action
carries a feature axis and a discrete one does not, so the only sites that had
to change are the two that read a width off the space and the two that
concatenate the previous action into the observation. Everything downstream --
``sample_and_log_prob``, ``log_prob``, ``entropy`` -- names no distribution
family and needed nothing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from memorax.algorithms import rtrrl_aaai as rtrrl
from memorax.algorithms import stream_ac
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble
from memorax.parameters import expand
from tests.support.environments import TinyContinuousEnv, TinyDiscreteEnv

ACTIONS = 2
OBSERVATION_DIM = 2
EPISODE_LENGTH = 8


def environment_factory(environment):
    def make(identifier, **options):
        del identifier, options
        return environment, environment.default_params

    return make


def rtrrl_parameters(head, *, meta_rl):
    return expand(
        rtrrl.PARAMETERS,
        {
            "torso.backbone.kind": "lru",
            "torso.backbone.lru.feature_dim": 4,
            "torso.backbone.lru.hidden_dim": 2,
            "torso.backbone.lru.differentiation.kind": "exact_rtrl",
            "torso.optimizer.kind": "adam",
            "torso.optimizer.adam.lr": 1e-3,
            "torso.grad_clip": 1.0,
            "torso.follow": 0.25,
            "actor.optimizer.kind": "adam",
            "actor.optimizer.adam.lr": 5e-4,
            "critic.optimizer.kind": "adam",
            "critic.optimizer.adam.lr": 5e-4,
            "actor.head.kind": head,
            "critic.head.kind": "value",
            "normalization.observation.kind": "none",
            "normalization.reward.kind": "none",
            "meta_rl": meta_rl,
        },
    )


def stream_ac_parameters(head, *, meta_rl):
    return expand(
        stream_ac.PARAMETERS,
        {
            "actor.head.kind": head,
            "actor.optimizer.bound.kind": "none",
            "actor.optimizer.base.kind": "sgd",
            "critic.head.kind": "value",
            "critic.optimizer.bound.kind": "none",
            "critic.optimizer.base.kind": "sgd",
            "normalization.observation.kind": "none",
            "normalization.reward.kind": "none",
            "backbone.kind": "mlp",
            "meta_rl": meta_rl,
        },
    )


ALGORITHMS = {
    "rtrrl": (rtrrl.RTRRL, rtrrl_parameters, rtrrl.OBSERVATIONS),
    "stream_ac": (stream_ac.StreamAC, stream_ac_parameters, stream_ac.OBSERVATIONS),
}


def assembled(name, *, head="categorical", meta_rl=False, environment=None, num_envs=1):
    definition, parameters, observations = ALGORITHMS[name]
    environment = TinyDiscreteEnv() if environment is None else environment
    return assemble(
        definition,
        BuildRequest(
            parameters=parameters(head, meta_rl=meta_rl),
            environment=EnvironmentSpec(
                id="tiny",
                backend=None,
                observed=None,
                episode_length=EPISODE_LENGTH,
            ),
            num_envs=num_envs,
            record=observations.trajectory_fields,
        ),
        environment_factory=environment_factory(environment),
    )


@pytest.mark.parametrize("name", tuple(ALGORITHMS))
def test_a_discrete_space_no_longer_fails_assembly_on_a_missing_feature_axis(name):
    """``Discrete.shape`` is ``()``; reading ``shape[0]`` raised an IndexError."""

    built = assembled(name)

    assert built.program is not None


@pytest.mark.parametrize("name", tuple(ALGORITHMS))
@pytest.mark.parametrize("meta_rl", (False, True))
def test_a_discrete_graph_trains_and_evaluates_through_the_program_contract(
    name, meta_rl
):
    built = assembled(name, meta_rl=meta_rl, num_envs=2)

    state = built.program.init(jax.random.key(0))
    trained, metrics = built.program.train(jax.random.key(1), state, 4)
    opened = built.program.open_evaluation(jax.random.key(2), trained)
    _, evaluated = built.program.evaluate(jax.random.key(3), opened, 4)

    assert int(trained.step) == 4
    assert metrics.interaction.reward.shape == (2, 2)
    assert evaluated.interaction.reward.shape == (2, 2)


@pytest.mark.parametrize("name", tuple(ALGORITHMS))
def test_a_discrete_action_is_one_integer_per_stream_and_a_legal_one(name):
    """No feature axis, and inside the space -- so the environment can step it."""

    built = assembled(name, num_envs=3)

    state = built.program.init(jax.random.key(0))
    _, metrics = built.program.train(jax.random.key(1), state, 6)
    actions = np.asarray(metrics.interaction.action)

    assert actions.shape == (2, 3)
    assert np.issubdtype(actions.dtype, np.integer)
    assert actions.min() >= 0 and actions.max() < ACTIONS


def graph_of(built):
    """The algorithm object the program's three arrows are bound to.

    Through ``getattr`` because ``Program`` declares plain callables, and a
    type checker reading the declaration rather than what assembly puts there
    has no reason to believe they are bound methods.
    """

    return getattr(built.program.init, "__self__")


def feedback_input(name, graph, timestep):
    """The one vector the sequence sees, however its owner spells the call."""

    obs, _, action, reward = timestep
    if name == "rtrrl":
        return graph.core.torso._input(timestep)
    return graph.core.actor.block._input(obs, action, reward)


@pytest.mark.parametrize("name", tuple(ALGORITHMS))
def test_the_feedback_input_carries_the_one_hot_and_not_the_integer(name):
    """An integer has no axis to concatenate on, and no metric meaning if it had.

    The width the graph declared and the width the encoding produces have to be
    the same number or the first forward pass would not build; reading the
    vector itself says which number it is and where the action sits in it.
    """

    built = assembled(name, meta_rl=True, num_envs=2)
    graph = graph_of(built)
    state = built.program.init(jax.random.key(0))
    timestep = state.timestep.replace(
        action=jnp.array([0, 1], dtype=jnp.int32),
        done=jnp.array([False, False]),
    ).to_sequence()

    encoded = feedback_input(name, graph, timestep)

    assert encoded.shape == (2, 1, OBSERVATION_DIM + ACTIONS + 1)
    np.testing.assert_array_equal(
        np.asarray(encoded[:, 0, OBSERVATION_DIM : OBSERVATION_DIM + ACTIONS]),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )


def test_rtrrl_carries_no_action_at_all_across_an_episode_boundary():
    """The zero vector, not the one-hot of whichever action is numbered zero.

    RTRRL clears the feedback of an ended stream, and an ended stream has to
    stay distinguishable from one that chose action 0 -- so the clearing has to
    happen after the widening, not before it.
    """

    built = assembled("rtrrl", meta_rl=True, num_envs=2)
    graph = graph_of(built)
    state = built.program.init(jax.random.key(0))
    timestep = state.timestep.replace(
        action=jnp.array([0, 0], dtype=jnp.int32),
        done=jnp.array([True, False]),
    ).to_sequence()

    encoded = feedback_input("rtrrl", graph, timestep)

    ended, running = np.asarray(
        encoded[:, 0, OBSERVATION_DIM : OBSERVATION_DIM + ACTIONS]
    )
    np.testing.assert_array_equal(ended, np.zeros(ACTIONS, dtype=np.float32))
    np.testing.assert_array_equal(running, np.array([1.0, 0.0], dtype=np.float32))


@pytest.mark.parametrize("name", tuple(ALGORITHMS))
def test_a_gaussian_head_is_refused_by_a_discrete_space_and_the_reverse(name):
    with pytest.raises(ValueError, match="cannot be built against a discrete"):
        assembled(name, head="global_std")

    with pytest.raises(ValueError, match="cannot be built against a continuous"):
        assembled(name, head="categorical", environment=TinyContinuousEnv())


@pytest.mark.parametrize("name", tuple(ALGORITHMS))
def test_the_entropy_term_is_still_one_scalar_per_step(name):
    """What the entropy coefficient multiplies has the shape it always had.

    Only its range changed: a Gaussian's is differential entropy and unbounded
    below, a Categorical's is Shannon entropy in ``[0, ln n]``. Same shape, so
    no call site moves; different scale, so a tuned coefficient does not carry
    across from one to the other.
    """

    built = assembled(name, num_envs=2)

    state = built.program.init(jax.random.key(0))
    _, metrics = built.program.train(jax.random.key(1), state, 4)
    entropy = np.asarray(metrics.forward.actor.entropy)

    assert entropy.shape == (2, 2)
    assert np.all(entropy >= 0.0) and np.all(entropy <= np.log(ACTIONS) + 1e-5)


def test_a_continuous_space_still_reaches_the_gaussian_heads_it_always_did():
    """The branch is new; the path through it for a Box is the old one."""

    for name in ALGORITHMS:
        built = assembled(name, head="state_std", environment=TinyContinuousEnv())
        state = built.program.init(jax.random.key(0))
        _, metrics = built.program.train(jax.random.key(1), state, 2)

        assert np.asarray(metrics.interaction.action).shape == (2, 1, 2)
