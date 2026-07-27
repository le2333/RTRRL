"""The class computes what the closures did, leaf for leaf.

RTRRL is six hundred lines of numerics that no reachable assertion pins down:
a trace decayed in the wrong order or a gradient routed to the wrong domain
still runs, still stays finite, and still moves the parameters. Turning it
inside out is only safe if the old text is still around to answer against, so
``_rtrrl_frozen`` is a copy of it and this file is the answer. Both go once
this has run.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest
from _rtrrl_frozen import RTRRLConfig as FrozenConfig
from _rtrrl_frozen import RTRRLParts as FrozenParts
from _rtrrl_frozen import build_rtrrl
from conftest import TinyContinuousEnv

from memorax.algorithms.rtrrl import RTRRL, RTRRLConfig
from memorax.networks import (
    FeatureExtractor,
    LRUCell,
    LRUConfig,
    Memoroid,
    heads,
)
from memorax.rl import NormalizationConfig

# Each one reaches a branch that the others leave alone.
CONFIGURATIONS = {
    "adam": {},
    "obgd": {"update_rule": "obgd", "kappa": 2.0},
    "gates_shut": {"actor_to_recurrent": False, "critic_to_recurrent": False},
    "actor_gate_shut": {"actor_to_recurrent": False},
    "frozen_gamma": {"freeze_gamma": True},
    "stale_trace": {"update_trace_before_td": False},
    "clipped_action": {"act_clip": 0.5},
    "whole_period": {"update_period": 1.0},
}

BASE = {
    "num_envs": 2,
    "gamma": 0.91,
    "lambda_pi": 0.73,
    "lambda_v": 0.67,
    "lambda_rnn": 0.61,
    "td_lr": 2e-4,
    "rnn_lr": 3e-5,
    "eta_pi": 0.4,
    "eta_f": 0.6,
    "entropy_rate": 1e-4,
    "update_period": 0.2,
}


def modules():
    """Fresh module instances, identical in every way that reaches a number."""

    def encoder():
        return nn.Sequential((nn.Dense(3), nn.tanh))

    return {
        "feature_extractor": FeatureExtractor(
            observation_extractor=encoder(),
            action_extractor=encoder(),
            reward_extractor=encoder(),
        ),
        "torso": Memoroid(
            cell=LRUCell(config=LRUConfig(features=9, hidden_dim=2, output_dim=3))
        ),
        "actor_head": heads.Gaussian(action_dim=2),
        "critic_head": heads.VNetwork(),
    }


def frozen_side(normalization, record_trajectory, overrides):
    env = TinyContinuousEnv()
    program = build_rtrrl(
        FrozenConfig(**{**BASE, **overrides}),
        FrozenParts(
            env=env,
            env_params=env.default_params,
            normalization=normalization,
            record_trajectory=record_trajectory,
            **modules(),
        ),
    )
    return program.init_fn, program.train_epoch_fn, program.evaluate_fn


def class_side(normalization, record_trajectory, overrides):
    env = TinyContinuousEnv()
    agent = RTRRL(
        RTRRLConfig(**{**BASE, **overrides}),
        env,
        env.default_params,
        normalization=normalization,
        record_trajectory=record_trajectory,
        **modules(),
    )
    return agent.init, agent.train, agent.evaluate


def compared(old, new, path=""):
    """Every leaf of both trees, paired, with the path that reached it."""

    old_leaves, old_structure = jax.tree.flatten(old)
    new_leaves, new_structure = jax.tree.flatten(new)
    assert old_structure == new_structure, f"{path}: the two differ in shape"
    return list(zip(old_leaves, new_leaves))


def assert_identical(old, new, what):
    pairs = compared(old, new, what)
    assert pairs, f"{what}: nothing was compared"
    for index, (before, after) in enumerate(pairs):
        before = jnp.asarray(before)
        after = jnp.asarray(after)
        assert before.dtype == after.dtype, f"{what}[{index}]: dtype changed"
        if jnp.issubdtype(before.dtype, jnp.inexact):
            assert jnp.array_equal(
                before, after, equal_nan=True
            ), f"{what}[{index}]: {before} became {after}"
        else:
            assert jnp.array_equal(before, after), f"{what}[{index}]"
    return len(pairs)


@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
def test_the_class_reproduces_the_closures(name):
    overrides = CONFIGURATIONS[name]
    normalization = NormalizationConfig(
        normalize_observation=True, normalize_reward=True
    )
    record_trajectory = name == "adam"

    old_init, old_train, old_evaluate = frozen_side(
        normalization, record_trajectory, overrides
    )
    new_init, new_train, new_evaluate = class_side(
        normalization, record_trajectory, overrides
    )

    old_state = jax.jit(old_init)(jax.random.key(0))
    new_state = jax.jit(new_init)(jax.random.key(0))
    compared_leaves = assert_identical(old_state, new_state, "init")

    old_trained, old_metrics = jax.jit(old_train, static_argnums=2)(
        jax.random.key(1), old_state, 8
    )
    new_trained, new_metrics = jax.jit(new_train, static_argnums=2)(
        jax.random.key(1), new_state, 8
    )
    compared_leaves += assert_identical(old_trained, new_trained, "trained")
    compared_leaves += assert_identical(old_metrics, new_metrics, "metrics")

    _, old_summary = jax.jit(old_evaluate, static_argnums=2)(
        jax.random.key(2), old_trained, 4
    )
    _, new_summary = jax.jit(new_evaluate, static_argnums=2)(
        jax.random.key(2), new_trained, 4
    )
    compared_leaves += assert_identical(old_summary, new_summary, "evaluation")

    # A comparison that reached almost nothing would pass while saying nothing.
    assert compared_leaves > 100, f"only {compared_leaves} leaves were compared"


def test_the_comparison_would_notice_a_changed_number():
    """The assertion above is only worth running if it can fail."""

    left = {"a": jnp.zeros((2, 3)), "b": jnp.ones((4,))}
    right = {"a": jnp.zeros((2, 3)), "b": jnp.ones((4,)).at[2].set(1 + 1e-7)}
    with pytest.raises(AssertionError):
        assert_identical(left, right, "sanity")
