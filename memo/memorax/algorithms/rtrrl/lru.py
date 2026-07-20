"""Strict AAAI25 LRU initialization, recurrence, and readout."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Mapping, cast

import flax.linen as nn
from flax import struct
import jax
import jax.numpy as jnp

from memorax.utils.typing import Array


def _nu_log_init(
    key: Array,
    shape: tuple[int, ...],
    r_max: float = 1.0,
    r_min: float = 0.0,
) -> Array:
    uniform = jax.random.uniform(key, shape=shape)
    return jnp.log(
        -0.5 * jnp.log(uniform * (r_max**2 - r_min**2) + r_min**2)
    )


def _theta_log_init(
    key: Array, shape: tuple[int, ...], max_phase: float = 6.28
) -> Array:
    return jnp.log(max_phase * jax.random.uniform(key, shape=shape))


def _gamma_log_init(
    key: Array,
    shape: tuple[int, ...],
    nu_log: Array,
    theta_log: Array,
) -> Array:
    del key, shape
    lam = jnp.exp(-jnp.exp(nu_log) + 1j * jnp.exp(theta_log))
    return jnp.log(jnp.sqrt(1 - jnp.abs(lam) ** 2))


def _matrix_init(
    key: Array,
    shape: tuple[int, ...],
    dtype: Any = jnp.float32,
    normalization: float | Array = 1,
) -> Array:
    return jax.random.normal(key=key, shape=shape, dtype=dtype) / normalization


class _InitializerCell(nn.Module):
    hidden_dim: int
    input_dim: int

    def setup(self) -> None:
        self.nu_log = self.param(
            "nu_log", _nu_log_init, (self.hidden_dim,), 1.0, 0.0
        )
        self.theta_log = self.param(
            "theta_log", _theta_log_init, (self.hidden_dim,), 6.28
        )
        self.gamma_log = self.param(
            "gamma_log",
            _gamma_log_init,
            (self.hidden_dim,),
            self.nu_log,
            self.theta_log,
        )

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        B_real = self.param(
            "B_real",
            partial(_matrix_init, normalization=jnp.sqrt(2 * self.input_dim)),
            (self.hidden_dim, self.input_dim),
        )
        B_img = self.param(
            "B_img",
            partial(_matrix_init, normalization=jnp.sqrt(2 * self.input_dim)),
            (self.hidden_dim, self.input_dim),
        )
        input_vector = inputs if inputs.ndim == 1 else inputs[0]
        return (B_real + 1j * B_img) @ input_vector


class _InitializerOnlineCell(nn.Module):
    hidden_dim: int
    input_dim: int

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        return _InitializerCell(
            self.hidden_dim,
            self.input_dim,
            name="LRUCell_0",
        )(inputs)


class _InitializerLayer(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        C_real = self.param(
            "C_real",
            partial(_matrix_init, normalization=jnp.sqrt(self.hidden_dim)),
            (self.output_dim, self.hidden_dim),
        )
        C_img = self.param(
            "C_img",
            partial(_matrix_init, normalization=jnp.sqrt(self.hidden_dim)),
            (self.output_dim, self.hidden_dim),
        )
        D = self.param("D", _matrix_init, (self.output_dim, self.input_dim))
        hidden = _InitializerOnlineCell(
            self.hidden_dim,
            self.input_dim,
            name="OnlineLRUCell_0",
        )(inputs)
        input_vector = inputs if inputs.ndim == 1 else inputs[0]
        return ((C_real + 1j * C_img) @ hidden).real + D @ input_vector


@struct.dataclass
class LRUCarry:
    """The AAAI25 complex hidden state."""

    hidden: Array


@struct.dataclass
class LRUCreditState:
    """Compact AAAI25 sensitivities for lambda, gamma, and complex B."""

    lambda_sensitivity: Array
    gamma_sensitivity: Array
    B_sensitivity: Array


@dataclass(frozen=True)
class AAAI25LRU:
    """The strict one-step AAAI25 LRU with explicit parameters and carry."""

    input_dim: int
    hidden_dim: int
    output_dim: int
    activation: str | None = "silu"

    def initialize(
        self, key: Array, input_shape: tuple[int, ...]
    ) -> tuple[dict[str, Array], LRUCarry]:
        if input_shape[-1] != self.input_dim:
            raise ValueError(
                f"input width {input_shape[-1]} does not match {self.input_dim}"
            )
        dummy = jnp.zeros(input_shape, dtype=jnp.float32)
        with jax.threefry_partitionable(False):
            initialized = _InitializerLayer(
                self.input_dim,
                self.hidden_dim,
                self.output_dim,
            ).init(key, dummy)
        layer_params = cast(Mapping[str, Any], initialized["params"])
        online_params = cast(
            Mapping[str, Any], layer_params["OnlineLRUCell_0"]
        )
        cell_params = cast(Mapping[str, Array], online_params["LRUCell_0"])
        params = {
            "nu_log": cell_params["nu_log"],
            "theta_log": cell_params["theta_log"],
            "gamma_log": cell_params["gamma_log"],
            "B_real": cell_params["B_real"],
            "B_img": cell_params["B_img"],
            "C_real": layer_params["C_real"],
            "C_img": layer_params["C_img"],
            "D": layer_params["D"],
        }
        carry = LRUCarry(
            hidden=jnp.zeros(
                (*input_shape[:-1], self.hidden_dim), dtype=jnp.complex64
            )
        )
        return params, carry

    def complex_lambda(self, params: Mapping[str, Array]) -> Array:
        return jnp.exp(
            -jnp.exp(params["nu_log"]) + 1j * jnp.exp(params["theta_log"])
        )

    def normalized_B(self, params: Mapping[str, Array]) -> Array:
        return (params["B_real"] + 1j * params["B_img"]) * jnp.exp(
            params["gamma_log"][:, None]
        )

    def readout_parts(
        self,
        params: Mapping[str, Array],
        carry: LRUCarry,
        inputs: Array,
    ) -> tuple[Array, Array, Array]:
        C = params["C_real"] + 1j * params["C_img"]
        projection = (carry.hidden @ C.transpose()).real
        skip = inputs @ params["D"].transpose()
        return projection, skip, projection + skip

    def forward(
        self,
        params: Mapping[str, Array],
        carry: LRUCarry,
        inputs: Array,
        reset: Array | bool,
    ) -> tuple[LRUCarry, Array]:
        del carry, reset
        normalized_B = self.normalized_B(params)
        projected_input = (
            normalized_B @ inputs
            if inputs.ndim == 1
            else jax.vmap(lambda value: normalized_B @ value)(inputs)
        )
        next_carry = LRUCarry(hidden=projected_input)
        _, _, preactivation = self.readout_parts(params, next_carry, inputs)
        output = (
            getattr(jax.nn, self.activation)(preactivation)
            if self.activation is not None
            else preactivation
        )
        return next_carry, output

    def _update_credit(
        self,
        params: Mapping[str, Array],
        credit_state: LRUCreditState,
        carry: LRUCarry,
        inputs: Array,
    ) -> LRUCreditState:
        lam = self.complex_lambda(params)
        B = params["B_real"] + 1j * params["B_img"]
        gamma = jnp.exp(params["gamma_log"])
        return LRUCreditState(
            lambda_sensitivity=(
                lam * credit_state.lambda_sensitivity + carry.hidden
            ),
            gamma_sensitivity=(
                lam * credit_state.gamma_sensitivity
                + jnp.einsum("hi,...i->...h", B, inputs)
            ),
            B_sensitivity=(
                lam[:, None] * credit_state.B_sensitivity
                + jnp.einsum("h,...i->...hi", gamma, inputs)
            ),
        )

    def _validate_credit_path(
        self,
        credit_state: LRUCreditState,
        carry: LRUCarry,
        inputs: Array,
        cotangent: Array | None = None,
    ) -> None:
        expected_shapes = {
            "credit_state.lambda_sensitivity": (self.hidden_dim,),
            "credit_state.gamma_sensitivity": (self.hidden_dim,),
            "credit_state.B_sensitivity": (self.hidden_dim, self.input_dim),
            "carry.hidden": (self.hidden_dim,),
            "inputs": (self.input_dim,),
        }
        actual_shapes = {
            "credit_state.lambda_sensitivity": credit_state.lambda_sensitivity.shape,
            "credit_state.gamma_sensitivity": credit_state.gamma_sensitivity.shape,
            "credit_state.B_sensitivity": credit_state.B_sensitivity.shape,
            "carry.hidden": carry.hidden.shape,
            "inputs": inputs.shape,
        }
        if cotangent is not None:
            expected_shapes["cotangent"] = (self.output_dim,)
            actual_shapes["cotangent"] = cotangent.shape
        mismatches = [
            f"{name}={actual_shapes[name]} (expected {expected})"
            for name, expected in expected_shapes.items()
            if actual_shapes[name] != expected
        ]
        if mismatches:
            raise ValueError(
                "online LRU credit requires unbatched pinned shapes; "
                + ", ".join(mismatches)
            )

    def credit(
        self,
        params: Mapping[str, Array],
        credit_state: LRUCreditState,
        carry: LRUCarry,
        inputs: Array,
        cotangent: Array,
    ) -> tuple[LRUCreditState, dict[str, Array]]:
        self._validate_credit_path(credit_state, carry, inputs, cotangent)
        next_credit = self._update_credit(params, credit_state, carry, inputs)
        next_carry, _ = self.forward(params, carry, inputs, reset=False)

        def output_from_hidden(hidden: Array) -> Array:
            hidden_carry = LRUCarry(hidden=hidden)
            _, _, preactivation = self.readout_parts(
                params, hidden_carry, inputs
            )
            return (
                getattr(jax.nn, self.activation)(preactivation)
                if self.activation is not None
                else preactivation
            )

        _, hidden_pullback = jax.vjp(output_from_hidden, next_carry.hidden)
        hidden_cotangent = hidden_pullback(cotangent)[0][0]

        lambda_cotangent = (
            hidden_cotangent * next_credit.lambda_sensitivity
        )
        while lambda_cotangent.ndim > 1:
            lambda_cotangent = lambda_cotangent.sum(axis=0)
        _, lambda_pullback = jax.vjp(
            lambda nu_log, theta_log: jnp.exp(
                -jnp.exp(nu_log) + 1j * jnp.exp(theta_log)
            ),
            params["nu_log"],
            params["theta_log"],
        )
        nu_log_grad, theta_log_grad = lambda_pullback(lambda_cotangent)

        gamma_contraction = (
            hidden_cotangent * next_credit.gamma_sensitivity
        ).real
        while gamma_contraction.ndim > 1:
            gamma_contraction = gamma_contraction.sum(axis=0)
        gamma_log_grad = gamma_contraction * jnp.exp(params["gamma_log"])

        B_contraction = hidden_cotangent * next_credit.B_sensitivity
        while B_contraction.ndim > 2:
            B_contraction = B_contraction.sum(axis=0)
        gradients = {
            "nu_log": nu_log_grad,
            "theta_log": theta_log_grad,
            "gamma_log": gamma_log_grad,
            "B_real": B_contraction.real,
            "B_img": -B_contraction.imag,
        }
        return next_credit, gradients

    def forward_with_credit(
        self,
        params: Mapping[str, Array],
        credit_state: LRUCreditState,
        carry: LRUCarry,
        inputs: Array,
        reset: Array | bool,
    ) -> tuple[LRUCreditState, LRUCarry, Array]:
        """Run the verified forward under the compact online-credit VJP."""
        self._validate_credit_path(credit_state, carry, inputs)
        return _forward_with_credit(
            self, params, credit_state, carry, inputs, reset
        )


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _forward_with_credit(
    component: AAAI25LRU,
    params: Mapping[str, Array],
    credit_state: LRUCreditState,
    carry: LRUCarry,
    inputs: Array,
    reset: Array | bool,
) -> tuple[LRUCreditState, LRUCarry, Array]:
    next_carry, output = component.forward(params, carry, inputs, reset)
    next_credit = component._update_credit(params, credit_state, carry, inputs)
    return next_credit, next_carry, output


def _forward_with_credit_fwd(
    component: AAAI25LRU,
    params: Mapping[str, Array],
    credit_state: LRUCreditState,
    carry: LRUCarry,
    inputs: Array,
    reset: Array | bool,
) -> tuple[
    tuple[LRUCreditState, LRUCarry, Array],
    tuple[Mapping[str, Array], LRUCreditState, LRUCarry, Array, Array | bool],
]:
    next_carry, output = component.forward(params, carry, inputs, reset)
    next_credit = component._update_credit(
        params, credit_state, carry, inputs
    )
    result = (next_credit, next_carry, output)
    return result, (params, credit_state, carry, inputs, reset)


def _forward_with_credit_bwd(
    component: AAAI25LRU,
    residual: tuple[
        Mapping[str, Array], LRUCreditState, LRUCarry, Array, Array | bool
    ],
    output_cotangent: tuple[LRUCreditState, LRUCarry, Array],
) -> tuple[Any, LRUCreditState, LRUCarry, Array, None]:
    params, credit_state, carry, inputs, reset = residual
    _, carry_cotangent, value_cotangent = output_cotangent

    def ordinary_forward(
        ordinary_params: Mapping[str, Array],
        ordinary_carry: LRUCarry,
        ordinary_inputs: Array,
    ) -> tuple[LRUCarry, Array]:
        return component.forward(
            ordinary_params, ordinary_carry, ordinary_inputs, reset
        )

    _, ordinary_pullback = jax.vjp(ordinary_forward, params, carry, inputs)
    params_cotangent, carry_input_cotangent, inputs_cotangent = (
        ordinary_pullback((carry_cotangent, value_cotangent))
    )
    _, recurrent_cotangent = component.credit(
        params, credit_state, carry, inputs, value_cotangent
    )
    params_cotangent = {
        **params_cotangent,
        **recurrent_cotangent,
    }
    credit_input_cotangent = jax.tree.map(jnp.zeros_like, credit_state)
    return (
        params_cotangent,
        credit_input_cotangent,
        carry_input_cotangent,
        inputs_cotangent,
        None,
    )


_forward_with_credit.defvjp(
    _forward_with_credit_fwd,
    _forward_with_credit_bwd,
)


__all__ = ["AAAI25LRU", "LRUCarry", "LRUCreditState"]
