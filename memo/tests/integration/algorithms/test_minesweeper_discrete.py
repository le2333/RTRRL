"""Both online learners on a discrete task that is not two actions wide.

CartPole has two actions and is fully observed, so it answers whether the
discrete path holds at all and nothing else. POPJym's Minesweeper is the task
the comparison is actually for: sixteen actions, a two-dimensional observation
that reports only the neighbourhood of the cell just probed, and an episode
that ends the moment a mine is hit. A policy has to remember where it has been,
which is what a recurrent online learner is claimed to be for.

``popjym`` is an optional extra rather than a base dependency, so this skips
where it is not installed -- including the CI job that runs the algorithm
suite, which syncs no extras. Where it does run, it runs the same assembly path
a deployed run does.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from memorax.algorithms import rtrrl_aaai as rtrrl
from memorax.algorithms import stream_ac
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble
from memorax.parameters import expand

pytest.importorskip("popjym", reason="the popjym extra is not installed")

ENVIRONMENT = "popjym::MinesweeperEasy"
ACTIONS = 16
EPISODE_LENGTH = 14
NUM_ENVS = 4
STEPS = 64


def rtrrl_parameters():
    return expand(
        rtrrl.PARAMETERS,
        {
            "torso.backbone.kind": "lru",
            "torso.backbone.lru.feature_dim": 16,
            "torso.backbone.lru.hidden_dim": 8,
            "torso.backbone.lru.differentiation.kind": "exact_rtrl",
            "torso.optimizer.kind": "adam",
            "torso.optimizer.adam.lr": 1e-3,
            "torso.grad_clip": 1.0,
            "torso.follow": 1.0,
            "actor.optimizer.kind": "adam",
            "actor.optimizer.adam.lr": 1e-3,
            "critic.optimizer.kind": "adam",
            "critic.optimizer.adam.lr": 1e-3,
            "actor.head.kind": "categorical",
            "critic.head.kind": "value",
            "normalization.observation.kind": "none",
            "normalization.reward.kind": "none",
            "meta_rl": True,
        },
    )


def stream_ac_parameters():
    return expand(
        stream_ac.PARAMETERS,
        {
            "actor.head.kind": "categorical",
            "actor.optimizer.bound.kind": "none",
            "actor.optimizer.base.kind": "sgd",
            "critic.head.kind": "value",
            "critic.optimizer.bound.kind": "none",
            "critic.optimizer.base.kind": "sgd",
            "normalization.observation.kind": "none",
            "normalization.reward.kind": "none",
            "backbone.kind": "mlp",
            "meta_rl": True,
        },
    )


ALGORITHMS = {
    "rtrrl": (rtrrl.RTRRL, rtrrl_parameters, rtrrl.OBSERVATIONS),
    "stream_ac": (stream_ac.StreamAC, stream_ac_parameters, stream_ac.OBSERVATIONS),
}


def assembled(name):
    definition, parameters, observations = ALGORITHMS[name]
    return assemble(
        definition,
        BuildRequest(
            parameters=parameters(),
            environment=EnvironmentSpec(
                id=ENVIRONMENT,
                backend=None,
                observed=None,
                episode_length=EPISODE_LENGTH,
            ),
            num_envs=NUM_ENVS,
            record=observations.trajectory_fields,
        ),
    )


def test_popjym_resolves_a_registered_name_through_the_deployment_vocabulary():
    """The name POPJym registers, which is the name the authors' fork resolves.

    ``MinesweeperEasy`` rather than ``Minesweeper`` plus a difficulty: the
    adapter used to take the difficulty as a second positional argument, which
    assembly has nothing to pass and so could never call.
    """

    from memorax.environments import make

    environment, parameters = make(
        ENVIRONMENT, observed=None, backend=None, episode_length=EPISODE_LENGTH
    )
    space = environment.action_space(parameters)

    assert space.n == ACTIONS
    assert space.shape == ()


@pytest.mark.parametrize("name", tuple(ALGORITHMS))
def test_both_learners_train_and_evaluate_on_sixteen_actions(name):
    built = assembled(name)

    state = built.program.init(jax.random.key(0))
    trained, metrics = built.program.train(jax.random.key(1), state, STEPS)
    opened = built.program.open_evaluation(jax.random.key(2), trained)
    _, evaluated = built.program.evaluate(jax.random.key(3), opened, STEPS)

    rounds = STEPS // NUM_ENVS
    actions = np.asarray(metrics.interaction.action)

    assert int(trained.step) == STEPS
    assert actions.shape == (rounds, NUM_ENVS)
    assert actions.min() >= 0 and actions.max() < ACTIONS
    for reading in (metrics, evaluated):
        assert np.isfinite(np.asarray(reading.interaction.reward)).all()
    # The task ends episodes on its own, well inside this budget, so the reset
    # path and the cleared feedback are both exercised rather than assumed.
    assert np.asarray(metrics.interaction.done).sum() > 0


@pytest.mark.parametrize("name", tuple(ALGORITHMS))
def test_the_policy_starts_near_uniform_over_the_sixteen(name):
    """A Categorical over n actions starts at ln(n) nats, not at a Gaussian's scale.

    This is the number the entropy coefficient multiplies, and it is why a
    coefficient tuned against a continuous arm does not carry over.
    """

    built = assembled(name)

    state = built.program.init(jax.random.key(0))
    _, metrics = built.program.train(jax.random.key(1), state, STEPS)
    entropy = np.asarray(metrics.forward.actor.entropy)

    assert entropy.shape == (STEPS // NUM_ENVS, NUM_ENVS)
    assert np.all(entropy <= np.log(ACTIONS) + 1e-5)
    assert entropy[0].mean() > 0.5 * np.log(ACTIONS)
