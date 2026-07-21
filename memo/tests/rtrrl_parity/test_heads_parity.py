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


def _oracle_variables(arrays):
    return freeze(
        {
            "params": {
                "actor": {
                    "kernel": jnp.asarray(arrays["heads/params/actor/kernel"])
                },
                "critic": {
                    "kernel": jnp.asarray(arrays["heads/params/critic/kernel"]),
                    "bias": jnp.asarray(arrays["heads/params/critic/bias"]),
                },
            },
            "falign": {
                "actor": {"B": jnp.asarray(arrays["heads/falign/actor/B"])},
                "critic": {"B": jnp.asarray(arrays["heads/falign/critic/B"])},
            },
        }
    )


def test_strict_parameter_paths_shapes_dtypes_and_initializer():
    arrays, _ = load_oracle()
    model = RTRRLTDHead(action_dim=2, discrete=False, f_align=True)
    variables = model.init(
        jax.random.PRNGKey(0), jnp.asarray(arrays["heads/input"])
    )
    leaves = flatten_with_paths(variables)
    oracle_leaves = flatten_with_paths(_oracle_variables(arrays))

    assert sorted(leaves) == [
        "falign/actor/B",
        "falign/critic/B",
        "params/actor/kernel",
        "params/critic/bias",
        "params/critic/kernel",
    ]
    assert leaves.keys() == oracle_leaves.keys()
    for path in leaves:
        assert leaves[path].shape == oracle_leaves[path].shape, path
        assert leaves[path].dtype == oracle_leaves[path].dtype, path

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
        action_dim=2, discrete=False, f_align=True
    ).apply(_oracle_variables(arrays), inputs)
    distribution = make_action_distribution(actor_output, discrete=False)
    action = jnp.asarray(arrays["heads/sampled_action"])

    assert isinstance(distribution, distrax.Normal)
    assert distribution.log_prob(action).shape == action.shape
    with jax.threefry_partitionable(False):
        sampled_action = distribution.sample(
            seed=jnp.asarray(arrays["heads/sample_key"])
        )
    assert_tree_close(
        {
            "actor_output": actor_output,
            "loc": distribution.loc,
            "scale": distribution.scale,
            "value": value,
            "sampled_action": sampled_action,
            "log_prob": distribution.log_prob(action),
            "entropy": distribution.entropy(),
            "log_prob_mean": distribution.log_prob(action).mean(),
            "entropy_mean": distribution.entropy().mean(),
        },
        {
            "actor_output": arrays["heads/actor_output"],
            "loc": arrays["heads/actor_loc"],
            "scale": arrays["heads/actor_scale"],
            "value": arrays["heads/value"],
            "sampled_action": arrays["heads/sampled_action"],
            "log_prob": arrays["heads/log_prob"],
            "entropy": arrays["heads/entropy"],
            "log_prob_mean": arrays["heads/log_prob_mean"],
            "entropy_mean": arrays["heads/entropy_mean"],
        },
        "exact",
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


def test_fixture_nontrivial_vjp_matches_complete_head_cotangent_tree():
    arrays, _ = load_oracle()
    model = RTRRLTDHead(action_dim=2, discrete=False, f_align=True)
    variables = _oracle_variables(arrays)
    inputs = jnp.asarray(arrays["heads/input"])
    cotangent = (
        jnp.asarray(arrays["heads/vjp/cotangent/actor"]),
        jnp.asarray(arrays["heads/vjp/cotangent/value"]),
    )
    assert np.count_nonzero(np.asarray(cotangent[0])) == cotangent[0].size
    assert np.count_nonzero(np.asarray(cotangent[1])) == cotangent[1].size
    assert np.any(arrays["heads/vjp/input"] != 0)
    assert np.any(arrays["heads/vjp/params/actor/kernel"] != 0)
    assert np.any(arrays["heads/vjp/params/critic/kernel"] != 0)
    assert np.any(arrays["heads/vjp/params/critic/bias"] != 0)
    np.testing.assert_array_equal(
        arrays["heads/vjp/falign/actor/B"],
        np.zeros_like(arrays["heads/vjp/falign/actor/B"]),
    )
    np.testing.assert_array_equal(
        arrays["heads/vjp/falign/critic/B"],
        np.zeros_like(arrays["heads/vjp/falign/critic/B"]),
    )

    raw_output, pullback = jax.vjp(lambda v, x: model.apply(v, x), variables, inputs)
    output = cast(tuple[jax.Array, jax.Array], raw_output)
    variables_vjp, input_vjp = pullback(cotangent)
    expected_variables_vjp = freeze(
        {
            "params": {
                "actor": {
                    "kernel": jnp.asarray(
                        arrays["heads/vjp/params/actor/kernel"]
                    )
                },
                "critic": {
                    "kernel": jnp.asarray(
                        arrays["heads/vjp/params/critic/kernel"]
                    ),
                    "bias": jnp.asarray(arrays["heads/vjp/params/critic/bias"]),
                },
            },
            "falign": {
                "actor": {
                    "B": jnp.asarray(arrays["heads/vjp/falign/actor/B"])
                },
                "critic": {
                    "B": jnp.asarray(arrays["heads/vjp/falign/critic/B"])
                },
            },
        }
    )
    assert_tree_close(
        {"actor": output[0], "value": output[1]},
        {
            "actor": arrays["heads/actor_output"],
            "value": arrays["heads/value"],
        },
        "exact",
    )
    assert_tree_close(
        variables_vjp,
        expected_variables_vjp,
        "exact",
    )
    assert_tree_close(
        {"input": input_vjp},
        {"input": arrays["heads/vjp/input"]},
        "exact",
    )
