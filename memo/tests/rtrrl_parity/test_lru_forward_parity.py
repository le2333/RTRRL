"""Strict forward parity tests for the AAAI25 LRU component."""

import inspect
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from memorax.algorithms.rtrrl.components import RecurrentComponent
from memorax.algorithms.rtrrl.lru import AAAI25LRU, LRUCarry
from memorax.networks.sequence_models.lru import (
    LRUCarry as GenericLRUCarry,
    LRUCell,
    LRUConfig,
)

from .assertions import assert_tree_close, flatten_with_paths
from .oracle_capture import load_oracle


def test_generic_memorax_lru_documents_strict_aaai25_differences():
    generic = LRUCell(LRUConfig(features=4, hidden_dim=2, output_dim=2))
    inputs = jnp.ones((1, 1, 4), dtype=jnp.float32)

    variables = generic.init(jax.random.PRNGKey(7), inputs)
    paths = set(flatten_with_paths(variables))
    carry = cast(
        GenericLRUCarry,
        generic.initialize_carry(jax.random.PRNGKey(0), (1, 4)),
    )
    readout = cast(
        jax.Array,
        generic.apply(
            variables,
            GenericLRUCarry(
                state=jnp.ones((1, 1, 2), dtype=jnp.complex64),
                decay=jnp.ones((1, 1, 2), dtype=jnp.complex64),
            ),
            inputs,
            method=generic.read,
        ),
    )

    assert "params/B_imag" in paths
    assert "params/B_img" not in paths
    assert carry.state.shape == (1, 1, 2)
    assert carry.decay.shape == (1, 1, 2)
    assert readout.shape == (1, 1, 2)
    assert generic.config.dtype is None


def _oracle_params(arrays):
    return {
        name: jnp.asarray(arrays[f"lru/params/{name}"])
        for name in (
            "nu_log",
            "theta_log",
            "gamma_log",
            "B_real",
            "B_img",
            "C_real",
            "C_img",
            "D",
        )
    }


def test_component_protocol_keeps_dynamic_values_explicit():
    component = AAAI25LRU(input_dim=4, hidden_dim=2, output_dim=2)

    assert isinstance(component, RecurrentComponent)
    assert tuple(inspect.signature(component.forward).parameters) == (
        "params",
        "carry",
        "inputs",
        "reset",
    )
    assert vars(component) == {
        "input_dim": 4,
        "hidden_dim": 2,
        "output_dim": 2,
        "activation": "silu",
    }


def test_fixture_initialization_parameter_paths_shapes_dtypes_and_values():
    arrays, _ = load_oracle()
    component = AAAI25LRU(input_dim=4, hidden_dim=2, output_dim=2)

    params, carry = component.initialize(
        jnp.asarray(arrays["lru/init_key"]), arrays["lru/input"].shape
    )

    assert_tree_close(params, _oracle_params(arrays), "exact")
    assert_tree_close(
        carry,
        LRUCarry(hidden=jnp.asarray(arrays["lru/carry_before"])),
        "exact",
    )


def test_fixture_forward_matches_every_recurrence_and_readout_leaf():
    arrays, _ = load_oracle()
    component = AAAI25LRU(input_dim=4, hidden_dim=2, output_dim=2)
    params = _oracle_params(arrays)
    carry = LRUCarry(hidden=jnp.asarray(arrays["lru/carry_before"]))
    inputs = jnp.asarray(arrays["lru/input"])

    next_carry, output = component.forward(params, carry, inputs, reset=False)
    lam = component.complex_lambda(params)
    normalized_B = component.normalized_B(params)
    projection, skip, preactivation = component.readout_parts(
        params, next_carry, inputs
    )

    assert_tree_close(
        {
            "nu_log": params["nu_log"],
            "theta_log": params["theta_log"],
            "gamma_log": params["gamma_log"],
            "lambda": lam,
            "normalized_B": normalized_B,
            "next_hidden": next_carry.hidden,
            "projection": projection,
            "skip": skip,
            "preactivation": preactivation,
            "output": output,
        },
        {
            "nu_log": arrays["lru/params/nu_log"],
            "theta_log": arrays["lru/params/theta_log"],
            "gamma_log": arrays["lru/params/gamma_log"],
            "lambda": arrays["lru/lambda"],
            "normalized_B": arrays["lru/normalized_B"],
            "next_hidden": arrays["lru/carry_after"],
            "projection": arrays["lru/projection"],
            "skip": arrays["lru/skip"],
            "preactivation": arrays["lru/preactivation"],
            "output": arrays["lru/output"],
        },
        "exact",
    )


def test_fixture_reset_discards_previous_hidden_state():
    arrays, _ = load_oracle()
    component = AAAI25LRU(input_dim=4, hidden_dim=2, output_dim=2)
    params = _oracle_params(arrays)
    carry = LRUCarry(hidden=jnp.asarray(arrays["lru/carry_after"]))
    reset_input = jnp.asarray(arrays["lru/reset/input"])

    reset_carry, reset_output = component.forward(
        params, carry, reset_input, reset=jnp.array([True])
    )
    zero_carry_output = component.forward(
        params,
        LRUCarry(hidden=jnp.zeros_like(carry.hidden)),
        reset_input,
        reset=False,
    )

    assert_tree_close(
        {"hidden": reset_carry.hidden, "output": reset_output},
        {
            "hidden": arrays["lru/reset/carry_after"],
            "output": arrays["lru/reset/output"],
        },
        "exact",
    )
    assert_tree_close(
        (reset_carry, reset_output),
        zero_carry_output,
        "exact",
    )


def test_fixture_second_transition_matches_oracle_carry_handling():
    arrays, _ = load_oracle()
    component = AAAI25LRU(input_dim=4, hidden_dim=2, output_dim=2)

    next_carry, output = component.forward(
        _oracle_params(arrays),
        LRUCarry(hidden=jnp.asarray(arrays["lru/carry_after"])),
        jnp.asarray(arrays["lru/next/input"]),
        reset=False,
    )

    assert_tree_close(
        {"hidden": next_carry.hidden, "output": output},
        {
            "hidden": arrays["lru/next/carry_after"],
            "output": arrays["lru/next/output"],
        },
        "exact",
    )


def test_fixture_forward_eager_and_jit_have_measured_parity():
    arrays, _ = load_oracle()
    component = AAAI25LRU(input_dim=4, hidden_dim=2, output_dim=2)
    args = (
        _oracle_params(arrays),
        LRUCarry(hidden=jnp.asarray(arrays["lru/carry_before"])),
        jnp.asarray(arrays["lru/input"]),
        jnp.array(False),
    )

    eager = component.forward(*args)
    compiled = jax.jit(component.forward)(*args)

    assert_tree_close(eager, compiled, "exact")
    assert np.isfinite(np.asarray(compiled[1])).all()
