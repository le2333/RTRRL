"""The comparison the platform exists to make, on one real discrete task.

An online recurrent learner and a replayed sequence learner can only be read
against each other on a task both can be built against. R2D2 requires a
discrete action space -- a Q function reads one value per action, so an
infinite action set has no head, and that is a precondition rather than a gap.
Until RTRRL and StreamAC could be built against one too, no such task existed
and their curves had no axis to share.

``gymnax::CartPole-v1`` is that task. This asserts only that all three assemble
against it and produce comparable per-step readings; how well any of them
learns is not a unit test's business.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from memorax.algorithms import r2d2
from memorax.algorithms import rtrrl_aaai as rtrrl
from memorax.algorithms import stream_ac
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble
from memorax.parameters import expand

ENVIRONMENT = "gymnax::CartPole-v1"
EPISODE_LENGTH = 20
NUM_ENVS = 2
STEPS = 4


def rtrrl_parameters():
    return expand(
        rtrrl.PARAMETERS,
        {
            "torso.backbone.kind": "lru",
            "torso.backbone.lru.feature_dim": 8,
            "torso.backbone.lru.hidden_dim": 4,
            "torso.backbone.lru.differentiation.kind": "exact_rtrl",
            "torso.optimizer.kind": "adam",
            "torso.optimizer.adam.lr": 1e-3,
            "torso.grad_clip": 1.0,
            "torso.follow": 1.0,
            "heads.optimizer.kind": "adam",
            "heads.optimizer.adam.lr": 1e-3,
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


def r2d2_parameters():
    return expand(
        r2d2.PARAMETERS,
        {
            "backbone.kind": "lru",
            "backbone.lru.feature_dim": 8,
            "backbone.lru.hidden_dim": 4,
            "head.kind": "dueling",
            "learning.kind": "tbptt",
            "learning.tbptt.burn_in_length": 1,
            "learning.tbptt.unroll_length": 2,
            "optimizer.kind": "adam",
            "optimizer.adam.lr": 1e-3,
            "replay.capacity": 64,
            "replay.minimum_size": 8,
            "replay.batch_size": 2,
            "replay.priority_exponent": 0.9,
            "replay.importance_sampling_exponent": 0.6,
            "replay.max_priority_weight": 0.9,
            "target.update_period": 2,
            "returns.n_step": 2,
            "returns.value_transform.kind": "identity",
            "gamma": 0.99,
            "exploration.epsilon_start": 0.2,
            "exploration.epsilon_end": 0.05,
            "exploration.epsilon_decay_steps": 100,
            "exploration.evaluation_epsilon": 0.0,
        },
    )


ALGORITHMS = {
    "rtrrl": (rtrrl.RTRRL, rtrrl_parameters),
    "stream_ac": (stream_ac.StreamAC, stream_ac_parameters),
    "r2d2": (r2d2.R2D2, r2d2_parameters),
}


def assembled(name):
    definition, parameters = ALGORITHMS[name]
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
        ),
    )


@pytest.mark.parametrize("name", tuple(ALGORITHMS))
def test_every_algorithm_assembles_against_the_same_discrete_environment(name):
    built = assembled(name)

    state = built.program.init(jax.random.key(0))
    trained, metrics = built.program.train(jax.random.key(1), state, STEPS)
    evaluated = built.program.evaluate(jax.random.key(2), trained, STEPS)

    rounds = STEPS // NUM_ENVS
    assert int(trained.step) == STEPS
    # The readings a comparison is drawn from have the same shape whichever
    # algorithm produced them, which is what putting them on one axis needs.
    assert metrics.interaction.reward.shape == (rounds, NUM_ENVS)
    assert np.isfinite(np.asarray(metrics.interaction.reward)).all()
    assert evaluated.interaction.reward.shape == (rounds, NUM_ENVS)
