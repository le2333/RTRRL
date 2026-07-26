"""Expose a recurrent network through explicit dynamic arguments.

Flax modules keep carry and sensitivity implicit in how they are called.
Online algorithms differentiate through those quantities and carry them in
their own state, so they need them as ordinary arguments and results. This
adapter is the translation, and nothing about it is specific to one algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

import jax
import jax.numpy as jnp

from .sequence_models.memoroid import Memoroid
from .sequence_models.rnn import RNN


@runtime_checkable
class RecurrentComponent(Protocol):
    """A recurrent component whose changing values are explicit arguments."""

    def initialize(self, key: Any, input_shape: tuple[int, ...]) -> Any: ...

    def forward(self, params: Any, carry: Any, inputs: Any, reset: Any) -> Any: ...

    def credit(
        self,
        params: Any,
        credit_state: Any,
        carry: Any,
        inputs: Any,
        cotangent: Any,
    ) -> Any: ...


@dataclass(frozen=True)
class MemoraxRecurrentAdapter:
    """Expose an existing Memorax torso through explicit dynamic arguments."""

    module: Memoroid | RNN

    def init(self, *args: Any, **kwargs: Any) -> Any:
        return self.module.init(*args, **kwargs)

    def apply(self, *args: Any, **kwargs: Any) -> Any:
        return self.module.apply(*args, **kwargs)

    def initialize_carry(self, key: Any, input_shape: Any) -> Any:
        return self.module.initialize_carry(key, input_shape)

    def initialize_sensitivity(self, key: Any, input_shape: Any) -> Any:
        return self.module.initialize_sensitivity(key, input_shape)

    def initialize(self, key: Any, input_shape: tuple[int, ...]) -> Any:
        carry_shape = cast(Any, (*input_shape[:-1], None))
        carry_key, params_key, sensitivity_key = jax.random.split(key, 3)
        carry = self.module.initialize_carry(carry_key, carry_shape)
        inputs = jnp.zeros(input_shape, dtype=jnp.float32)
        reset = jnp.ones(input_shape[:-1], dtype=jnp.bool_)
        variables = self.module.init(
            {"params": params_key}, inputs, reset, initial_carry=carry
        )
        sensitivity = self.module.initialize_sensitivity(sensitivity_key, carry_shape)
        return variables["params"], carry, sensitivity

    def forward(self, params: Any, carry: Any, inputs: Any, reset: Any) -> Any:
        return self.module.apply({"params": params}, inputs, reset, initial_carry=carry)

    def credit(
        self,
        params: Any,
        credit_state: Any,
        carry: Any,
        inputs: Any,
        cotangent: Any,
    ) -> Any:
        del cotangent
        return self.module.apply(
            {"params": params},
            inputs,
            False,
            carry,
            sensitivity=credit_state,
            method="local_jacobian",
        )


__all__ = [
    "MemoraxRecurrentAdapter",
    "RecurrentComponent",
]
