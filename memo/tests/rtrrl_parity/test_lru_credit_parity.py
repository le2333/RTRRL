"""Oracle and finite-difference tests for strict AAAI25 LRU online credit."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from memorax.algorithms.rtrrl import lru as lru_module
from memorax.algorithms.rtrrl.lru import AAAI25LRU, LRUCarry

from .assertions import assert_tree_close
from .oracle_capture import load_oracle


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


def _zero_credit(batch_shape: tuple[int, ...] = ()):
    return lru_module.LRUCreditState(
        lambda_sensitivity=jnp.zeros((*batch_shape, 2), dtype=jnp.complex64),
        gamma_sensitivity=jnp.zeros((*batch_shape, 2), dtype=jnp.complex64),
        B_sensitivity=jnp.zeros((*batch_shape, 2, 4), dtype=jnp.complex64),
    )


def _oracle_credit_state(arrays, step: int):
    prefix = f"credit/step_{step}"
    return lru_module.LRUCreditState(
        lambda_sensitivity=jnp.asarray(arrays[f"{prefix}/lambda_sensitivity"]),
        gamma_sensitivity=jnp.asarray(arrays[f"{prefix}/gamma_sensitivity"]),
        B_sensitivity=jnp.asarray(arrays[f"{prefix}/B_sensitivity"]),
    )


def _oracle_recurrent_gradients(arrays, step: int) -> dict[str, jax.Array]:
    return {
        name: jnp.asarray(arrays[f"credit/step_{step}/grad/{name}"])
        for name in ("nu_log", "theta_log", "gamma_log", "B_real", "B_img")
    }


def _oracle_ordinary_gradients(arrays, step: int) -> dict[str, jax.Array]:
    return {
        name: jnp.asarray(arrays[f"credit/step_{step}/grad/{name}"])
        for name in ("C_real", "C_img", "D")
    }


def _oracle_state_gradient(arrays, step: int):
    prefix = f"credit/step_{step}/carry_gradient"
    return lru_module.LRUCreditState(
        lambda_sensitivity=jnp.asarray(
            arrays[f"{prefix}/lambda_sensitivity"]
        ),
        gamma_sensitivity=jnp.asarray(
            arrays[f"{prefix}/gamma_sensitivity"]
        ),
        B_sensitivity=jnp.asarray(arrays[f"{prefix}/B_sensitivity"]),
    )


def _max_abs_difference(actual, expected) -> float:
    differences = [
        np.max(np.abs(np.asarray(left) - np.asarray(right)), initial=0.0)
        for left, right in zip(
            jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
        )
    ]
    return float(max(differences, default=0.0))


def test_batched_credit_is_rejected_before_calculation():
    arrays, _ = load_oracle()
    component = AAAI25LRU(input_dim=4, hidden_dim=2, output_dim=2)

    with pytest.raises(ValueError, match="credit requires unbatched"):
        component.credit(
            _oracle_params(arrays),
            _zero_credit((1,)),
            LRUCarry(hidden=jnp.asarray(arrays["lru/carry_before"])),
            jnp.asarray(arrays["lru/input"]),
            jnp.asarray(arrays["credit/step_1/cotangent"])[None, :],
        )


def test_batched_custom_vjp_is_rejected_before_calculation():
    arrays, _ = load_oracle()
    component = AAAI25LRU(input_dim=4, hidden_dim=2, output_dim=2)

    with pytest.raises(ValueError, match="credit requires unbatched"):
        component.forward_with_credit(
            _oracle_params(arrays),
            _zero_credit((1,)),
            LRUCarry(hidden=jnp.asarray(arrays["lru/carry_before"])),
            jnp.asarray(arrays["lru/input"]),
            jnp.array(False),
        )


def test_two_step_credit_states_and_recurrent_gradients_match_oracle():
    arrays, _ = load_oracle()
    component = AAAI25LRU(input_dim=4, hidden_dim=2, output_dim=2)
    params = _oracle_params(arrays)
    carry_0 = LRUCarry(hidden=jnp.asarray(arrays["lru/unbatched/carry_before"]))

    state_1, gradients_1 = component.credit(
        params,
        _zero_credit(),
        carry_0,
        jnp.asarray(arrays["lru/unbatched/input"]),
        jnp.asarray(arrays["credit/step_1/cotangent"]),
    )
    carry_1, _ = component.forward(
        params, carry_0, jnp.asarray(arrays["lru/unbatched/input"]), reset=False
    )
    state_2, gradients_2 = component.credit(
        params,
        state_1,
        carry_1,
        jnp.asarray(arrays["lru/next/input"])[0],
        jnp.asarray(arrays["credit/step_2/cotangent"]),
    )

    assert_tree_close(state_1, _oracle_credit_state(arrays, 1), (2e-6, 2e-7))
    assert_tree_close(state_2, _oracle_credit_state(arrays, 2), (2e-6, 2e-7))
    assert_tree_close(
        gradients_1, _oracle_recurrent_gradients(arrays, 1), (2e-6, 2e-7)
    )
    assert_tree_close(
        gradients_2, _oracle_recurrent_gradients(arrays, 2), (2e-6, 2e-7)
    )
    for step, state, gradients in (
        (1, state_1, gradients_1),
        (2, state_2, gradients_2),
    ):
        print(
            f"CREDIT_MAX step={step} "
            f"state={_max_abs_difference(state, _oracle_credit_state(arrays, step)):.9g} "
            f"gradient={_max_abs_difference(gradients, _oracle_recurrent_gradients(arrays, step)):.9g}"
        )


@pytest.mark.parametrize("step", [1, 2])
def test_B_img_gradient_uses_negative_imaginary_contraction(step):
    arrays, _ = load_oracle()
    hidden_cotangent = np.asarray(
        arrays[f"credit/step_{step}/hidden_cotangent"]
    )
    B_sensitivity = np.asarray(arrays[f"credit/step_{step}/B_sensitivity"])
    complex_contraction = hidden_cotangent * B_sensitivity

    np.testing.assert_allclose(
        arrays[f"credit/step_{step}/grad/B_real"],
        np.real(complex_contraction),
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        arrays[f"credit/step_{step}/grad/B_img"],
        -np.imag(complex_contraction),
        rtol=2e-6,
        atol=2e-7,
    )


def test_custom_vjp_preserves_forward_values_and_uses_online_credit():
    arrays, _ = load_oracle()
    component = AAAI25LRU(input_dim=4, hidden_dim=2, output_dim=2)
    params = _oracle_params(arrays)
    carry = LRUCarry(hidden=jnp.asarray(arrays["lru/unbatched/carry_after"]))
    state = _oracle_credit_state(arrays, 1)
    inputs = jnp.asarray(arrays["lru/next/input"])[0]

    expected_forward = component.forward(params, carry, inputs, reset=False)
    (actual_state, actual_carry, actual_output), pullback = jax.vjp(
        component.forward_with_credit,
        params,
        state,
        carry,
        inputs,
        jnp.array(False),
    )
    output_cotangent = jnp.asarray(arrays["credit/step_2/cotangent"])
    zero_state = jax.tree.map(jnp.zeros_like, actual_state)
    zero_carry = jax.tree.map(jnp.zeros_like, actual_carry)
    params_grad, *_ = pullback((zero_state, zero_carry, output_cotangent))

    assert_tree_close((actual_carry, actual_output), expected_forward, "exact")
    assert_tree_close(actual_state, _oracle_credit_state(arrays, 2), (2e-6, 2e-7))
    assert_tree_close(
        {name: params_grad[name] for name in _oracle_recurrent_gradients(arrays, 2)},
        _oracle_recurrent_gradients(arrays, 2),
        (2e-6, 2e-7),
    )


def test_jitted_custom_vjp_backward_preserves_ordinary_cotangents():
    arrays, _ = load_oracle()
    component = AAAI25LRU(input_dim=4, hidden_dim=2, output_dim=2)
    params = _oracle_params(arrays)
    state = _oracle_credit_state(arrays, 1)
    carry = LRUCarry(hidden=jnp.asarray(arrays["lru/unbatched/carry_after"]))
    inputs = jnp.asarray(arrays["lru/next/input"])[0]
    output_cotangent = jnp.asarray(arrays["credit/step_2/cotangent"])

    def ordinary_forward(ordinary_params, ordinary_carry, ordinary_inputs):
        return component.forward(
            ordinary_params,
            ordinary_carry,
            ordinary_inputs,
            reset=jnp.array(False),
        )

    ordinary_output, ordinary_pullback = jax.vjp(
        ordinary_forward, params, carry, inputs
    )
    zero_carry = jax.tree.map(jnp.zeros_like, ordinary_output[0])
    expected_params, expected_carry, expected_inputs = ordinary_pullback(
        (zero_carry, output_cotangent)
    )

    @jax.jit
    def compiled_backward(compiled_params, compiled_state, compiled_carry, x, dy):
        primal, pullback = jax.vjp(
            component.forward_with_credit,
            compiled_params,
            compiled_state,
            compiled_carry,
            x,
            jnp.array(False),
        )
        next_state, next_carry, _ = primal
        cotangents = (
            jax.tree.map(jnp.zeros_like, next_state),
            jax.tree.map(jnp.zeros_like, next_carry),
            dy,
        )
        return primal, pullback(cotangents)

    (next_state, next_carry, output), gradients = compiled_backward(
        params, state, carry, inputs, output_cotangent
    )
    (
        params_gradient,
        state_gradient,
        carry_gradient,
        input_gradient,
        reset_gradient,
    ) = gradients

    assert isinstance(next_state, lru_module.LRUCreditState)
    assert isinstance(next_carry, LRUCarry)
    assert isinstance(state_gradient, lru_module.LRUCreditState)
    assert isinstance(carry_gradient, LRUCarry)
    assert set(params_gradient) == set(params)
    assert reset_gradient.dtype == jax.dtypes.float0
    assert_tree_close(
        (next_carry, output),
        component.forward(params, carry, inputs, reset=False),
        (2e-6, 2e-7),
    )
    assert_tree_close(
        {
            name: params_gradient[name]
            for name in ("nu_log", "theta_log", "gamma_log", "B_real", "B_img")
        },
        _oracle_recurrent_gradients(arrays, 2),
        (2e-6, 2e-7),
    )
    ordinary_params_gradient = {
        name: params_gradient[name] for name in ("C_real", "C_img", "D")
    }
    oracle_ordinary_gradient = _oracle_ordinary_gradients(arrays, 2)
    standard_ordinary_gradient = {
        name: expected_params[name] for name in ("C_real", "C_img", "D")
    }
    assert_tree_close(
        ordinary_params_gradient, oracle_ordinary_gradient, (2e-6, 2e-7)
    )
    assert_tree_close(
        ordinary_params_gradient,
        standard_ordinary_gradient,
        (2e-6, 2e-7),
    )
    assert_tree_close(
        carry_gradient,
        LRUCarry(
            hidden=jnp.asarray(
                arrays["credit/step_2/carry_gradient/hidden"]
            )
        ),
        "exact",
    )
    assert_tree_close(carry_gradient, expected_carry, "exact")
    assert_tree_close(
        input_gradient,
        jnp.asarray(arrays["credit/step_2/input_gradient"]),
        (2e-6, 2e-7),
    )
    assert_tree_close(input_gradient, expected_inputs, (2e-6, 2e-7))
    assert_tree_close(
        state_gradient,
        _oracle_state_gradient(arrays, 2),
        "exact",
    )
    assert_tree_close(
        state_gradient,
        jax.tree.map(jnp.zeros_like, state_gradient),
        "exact",
    )
    print(
        "CUSTOM_VJP_JIT_MAX "
        f"ordinary_oracle={_max_abs_difference(ordinary_params_gradient, oracle_ordinary_gradient):.9g} "
        f"ordinary_standard={_max_abs_difference(ordinary_params_gradient, standard_ordinary_gradient):.9g} "
        f"carry={_max_abs_difference(carry_gradient, expected_carry):.9g} "
        f"input={_max_abs_difference(input_gradient, expected_inputs):.9g}"
    )


def test_credit_eager_and_jit_match():
    arrays, _ = load_oracle()
    component = AAAI25LRU(input_dim=4, hidden_dim=2, output_dim=2)
    args = (
        _oracle_params(arrays),
        _oracle_credit_state(arrays, 1),
        LRUCarry(hidden=jnp.asarray(arrays["lru/unbatched/carry_after"])),
        jnp.asarray(arrays["lru/next/input"])[0],
        jnp.asarray(arrays["credit/step_2/cotangent"]),
    )

    eager = component.credit(*args)
    compiled = jax.jit(component.credit)(*args)

    assert_tree_close(eager, compiled, (2e-6, 2e-7))
    print(f"EAGER_JIT_MAX {_max_abs_difference(eager, compiled):.9g}")


@pytest.mark.skipif(
    os.environ.get("RTRRL_RUN_ACCELERATED_NUMERICS") != "1",
    reason="directional finite differences run only on authorized Batch workers",
)
@pytest.mark.parametrize(
    "group", ["nu_log", "theta_log", "gamma_log", "B_real", "B_img"]
)
def test_two_step_credit_directional_finite_differences(group):
    arrays, _ = load_oracle()
    component = AAAI25LRU(input_dim=4, hidden_dim=2, output_dim=2)
    params = _oracle_params(arrays)
    inputs_1 = jnp.asarray(arrays["lru/unbatched/input"])
    inputs_2 = jnp.asarray(arrays["lru/next/input"])[0]
    hidden_cotangent = jnp.asarray(arrays["credit/step_2/hidden_cotangent"])
    carry_0 = LRUCarry(hidden=jnp.asarray(arrays["lru/unbatched/carry_before"]))
    state_1, _ = component.credit(
        params,
        _zero_credit(),
        carry_0,
        inputs_1,
        jnp.asarray(arrays["credit/step_1/cotangent"]),
    )
    carry_1, _ = component.forward(params, carry_0, inputs_1, reset=False)
    _, gradients_2 = component.credit(
        params,
        state_1,
        carry_1,
        inputs_2,
        jnp.asarray(arrays["credit/step_2/cotangent"]),
    )
    analytical = np.asarray(gradients_2[group])
    epsilon = jnp.float32(1e-3)

    def objective(candidate):
        lam = component.complex_lambda(candidate)
        normalized_B = component.normalized_B(candidate)
        hidden_1 = jnp.einsum("hi,...i->...h", normalized_B, inputs_1)
        hidden_2 = lam * hidden_1 + jnp.einsum(
            "hi,...i->...h", normalized_B, inputs_2
        )
        return jnp.real(jnp.sum(hidden_cotangent * hidden_2))

    directions = np.eye(analytical.size, dtype=np.float32).reshape(
        (analytical.size, *analytical.shape)
    )
    numerical_directions = []
    analytical_directions = []
    for direction in directions:
        plus = {**params, group: params[group] + epsilon * direction}
        minus = {**params, group: params[group] - epsilon * direction}
        numerical_directions.append(
            np.asarray((objective(plus) - objective(minus)) / (2 * epsilon))
        )
        analytical_directions.append(np.sum(analytical * direction))

    numerical = np.asarray(numerical_directions, dtype=np.float32)
    exact = np.asarray(analytical_directions, dtype=np.float32)
    assert np.isfinite(numerical).all()
    assert np.isfinite(exact).all()
    cosine = float(
        np.dot(exact, numerical)
        / (np.linalg.norm(exact) * np.linalg.norm(numerical))
    )
    relative_error = float(
        np.linalg.norm(exact - numerical) / max(np.linalg.norm(numerical), 1e-12)
    )
    print(
        f"FD_METRIC group={group} cosine={cosine:.9g} "
        f"relative_error={relative_error:.9g}"
    )
    assert cosine >= 0.999
    assert relative_error <= 1e-2
