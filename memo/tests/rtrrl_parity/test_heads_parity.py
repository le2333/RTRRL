"""Numerical parity tests for the isolated AAAI25 RTRRL heads."""

from typing import cast

import distrax
import flax.linen as nn
from flax.core import freeze
import jax
import jax.numpy as jnp
import numpy as np

from memorax.algorithms.rtrrl.heads import (
    FADense,
    RTRRLTDHead,
    make_action_distribution,
)

from .assertions import assert_tree_close, flatten_with_paths
from .oracle_capture import load_oracle


def _explicit_oracle_variables():
    return freeze(
        {
            "params": {
                "actor": {
                    "kernel": jnp.array(
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.6186411, 0.581442, -0.35133776, -0.5162917],
                        ],
                        dtype=jnp.float32,
                    )
                },
                "critic": {
                    "kernel": jnp.zeros((2, 1), dtype=jnp.float32),
                    "bias": jnp.array([2.234071], dtype=jnp.float32),
                },
            }
        }
    )


def test_strict_parameter_paths_shapes_dtypes_and_initializer():
    arrays, _ = load_oracle()
    model = RTRRLTDHead(action_dim=2, discrete=False)
    variables = model.init(
        jax.random.PRNGKey(0), jnp.asarray(arrays["heads/input"])
    )
    leaves = flatten_with_paths(variables)

    assert sorted(leaves) == [
        "params/actor/kernel",
        "params/critic/bias",
        "params/critic/kernel",
    ]
    assert leaves["params/actor/kernel"].shape == (2, 4)
    assert leaves["params/critic/kernel"].shape == (2, 1)
    assert leaves["params/critic/bias"].shape == (1,)
    assert {leaf.dtype for leaf in leaves.values()} == {jnp.dtype(jnp.float32)}

    key = jax.random.PRNGKey(11)
    shape = (2, 4)
    expected = nn.initializers.glorot_normal(in_axis=-1, out_axis=-2)(
        key, shape, jnp.float32
    )
    actual = FADense(features=4).kernel_init(key, shape, jnp.float32)
    np.testing.assert_array_equal(actual, expected)


def test_fixture_forward_distribution_metrics_and_sample():
    arrays, _ = load_oracle()
    inputs = jnp.asarray(arrays["heads/input"])
    actor_output, value = RTRRLTDHead(
        action_dim=2, discrete=False
    ).apply(_explicit_oracle_variables(), inputs)
    distribution = make_action_distribution(actor_output, discrete=False)
    action = jnp.asarray(arrays["init/action"])

    assert isinstance(distribution, distrax.Normal)
    assert distribution.log_prob(action).shape == action.shape
    with jax.threefry_partitionable(False):
        action_key = jax.random.split(jax.random.PRNGKey(7), 3)[2]
        sampled_action = distribution.sample(seed=action_key)
    assert_tree_close(
        {
            "loc": distribution.loc,
            "scale": distribution.scale,
            "value": value,
            "sample": sampled_action,
            "log_prob": distribution.log_prob(action),
            "entropy": distribution.entropy(),
        },
        {
            "loc": arrays["heads/actor_loc"],
            "scale": arrays["heads/actor_scale"],
            "value": arrays["heads/value"],
            "sample": arrays["init/action"],
            "log_prob": np.array([[-0.9792221, -0.40963465]], dtype=np.float32),
            "entropy": np.array([[0.64159447, 0.452748]], dtype=np.float32),
        },
        (2e-6, 2e-7),
    )
    np.testing.assert_allclose(
        distribution.log_prob(action).mean(),
        np.float32(-0.6944284),
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        distribution.entropy().mean(),
        np.float32(0.54717124),
        rtol=2e-6,
        atol=2e-7,
    )


def test_discrete_distribution_is_categorical():
    logits = jnp.array([[0.25, -0.5, 1.0]], dtype=jnp.float32)

    distribution = make_action_distribution(logits, discrete=True)

    assert isinstance(distribution, distrax.Categorical)
    np.testing.assert_allclose(
        distribution.probs,
        jax.nn.softmax(logits),
        rtol=1e-7,
        atol=1e-7,
    )


def test_fadense_vjp_uses_feedback_matrix_without_changing_forward():
    arrays, _ = load_oracle()
    layer = FADense(features=2, f_align=True, use_bias=False)
    variables = freeze(
        {
            "params": {
                "kernel": jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
            },
            "falign": {
                "B": jnp.array([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32)
            },
        }
    )
    inputs = jnp.asarray(arrays["heads/input"])

    raw_output, pullback = jax.vjp(lambda x: layer.apply(variables, x), inputs)
    output = cast(jax.Array, raw_output)
    (input_vjp,) = pullback(jnp.ones_like(output))

    np.testing.assert_array_equal(
        output, jnp.array([[4.534738, 5.9250712]], jnp.float32)
    )
    np.testing.assert_array_equal(input_vjp, jnp.array([[11.0, 15.0]], jnp.float32))
