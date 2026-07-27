"""Hold the StreamAC-RTRL rewrite to the kernel it replaces, leaf by leaf.

The rewrite turns nested closures into methods so that a step can be reached
and inspected. That is a change of shape, and the only way to say it was not
also a change of arithmetic is to run both and compare everything that comes
out.

Right now this passes trivially, because the frozen copy is byte-identical to
the kernel. That is deliberate: it says the comparison itself works before
anything has moved. Delete both this file and the frozen copy once the rewrite
has landed and passed.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import TinyContinuousEnv

from memorax.algorithms._stream_ac_rtrl_frozen import (
    build_stream_ac_rtrl as build_frozen,
)
from memorax.algorithms.stream_ac_rtrl import (
    StreamACRTRLConfig,
    StreamACRTRLParts,
    build_stream_ac_rtrl,
)
from memorax.networks import RNN, FeatureExtractor, Network, RTUCell, RTUConfig, heads
from memorax.rl import NormalizationConfig

NUM_ENVS = 2
# Four scanned steps against a three step horizon, so a termination lands
# inside the epoch and the reset paths are compared rather than skipped.
TRAIN_STEPS = 8
EVAL_STEPS = 6


def _pieces(normalization, record_trajectory):
    """One config and one set of parts, handed to both builders unchanged."""

    env = TinyContinuousEnv()

    def network(head):
        return Network(
            feature_extractor=FeatureExtractor(
                observation_extractor=nn.Sequential((nn.Dense(3), nn.tanh))
            ),
            torso=RNN(cell=RTUCell(config=RTUConfig(features=3, hidden_dim=2))),
            head=head,
        )

    config = StreamACRTRLConfig(
        num_envs=NUM_ENVS,
        gamma=0.89,
        trace_lambda=0.71,
        actor_lr=0.15,
        critic_lr=0.12,
        entropy_coefficient=0.02,
        beta2=0.95,
        eps=1e-6,
    )
    parts = StreamACRTRLParts(
        env=env,
        env_params=env.default_params,
        actor_network=network(heads.Gaussian(action_dim=2)),
        critic_network=network(heads.VNetwork()),
        normalization=normalization,
        record_trajectory=record_trajectory,
    )
    return config, parts


def _comparable(leaf) -> np.ndarray:
    array = jnp.asarray(leaf)
    if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
        array = jax.random.key_data(array)
    return np.asarray(array)


def _same(actual, expected, label: str) -> None:
    left = jax.tree.leaves(actual)
    right = jax.tree.leaves(expected)
    assert len(left) == len(right), f"{label}: {len(left)} leaves against {len(right)}"
    for index, (one, other) in enumerate(zip(left, right)):
        np.testing.assert_array_equal(
            _comparable(one), _comparable(other), err_msg=f"{label}, leaf {index}"
        )


CASES = [
    pytest.param(None, False, id="plain"),
    pytest.param(
        NormalizationConfig(
            normalize_observation=True,
            normalize_reward=True,
            reward_gamma=0.89,
        ),
        True,
        id="normalized",
    ),
]


@pytest.mark.parametrize("normalization, record_trajectory", CASES)
def test_the_rewrite_changed_no_arithmetic(normalization, record_trajectory):
    config, parts = _pieces(normalization, record_trajectory)
    rewritten = build_stream_ac_rtrl(config, parts)
    frozen = build_frozen(config, parts)

    init_key, train_key, eval_key = jax.random.split(jax.random.key(0), 3)

    new_state = rewritten.init_fn(init_key)
    old_state = frozen.init_fn(init_key)
    _same(new_state, old_state, "initial state")

    new_state, new_metrics = rewritten.train_epoch_fn(train_key, new_state, TRAIN_STEPS)
    old_state, old_metrics = frozen.train_epoch_fn(train_key, old_state, TRAIN_STEPS)
    _same(new_metrics, old_metrics, "epoch metrics")
    _same(new_state, old_state, "state after an epoch")

    _, new_summary = rewritten.evaluate_fn(eval_key, new_state, EVAL_STEPS)
    _, old_summary = frozen.evaluate_fn(eval_key, old_state, EVAL_STEPS)
    _same(new_summary, old_summary, "evaluation summary")

    # Two kernels that both diverged would compare equal, since equality here
    # counts NaN as a match. The learning signal has to be a number for the
    # comparison above to have said anything.
    for name in ("td_error", "value", "log_prob"):
        values = _comparable(getattr(new_metrics, name))
        assert np.all(np.isfinite(values)), f"{name} was not finite"
