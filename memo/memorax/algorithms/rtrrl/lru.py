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

    def credit(
        self,
        params: Any,
        credit_state: Any,
        carry: Any,
        inputs: Any,
        cotangent: Any,
    ) -> Any:
        del params, credit_state, carry, inputs, cotangent
        raise NotImplementedError("online LRU credit is implemented in Task 6")


__all__ = ["AAAI25LRU", "LRUCarry"]
