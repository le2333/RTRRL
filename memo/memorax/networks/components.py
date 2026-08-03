"""The leaves a sequence is built from: one operation per component.

``reads`` is what a component asks for beyond its input. Nothing here asks for
anything, so nothing here takes a second argument.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from memorax.utils.typing import Array, Carry, Key


class Stateless(nn.Module):
    """A component with nothing to carry, which hands back what it was given."""

    recurrent = False
    reads = ()

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple) -> Carry:
        del key, input_shape


class FFN(Stateless):
    """One affine map."""

    features: int
    use_bias: bool = True
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: Array, initial_carry: Carry = None):
        return initial_carry, nn.Dense(
            self.features,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)


class LayerNorm(Stateless):
    @nn.compact
    def __call__(self, x: Array, initial_carry: Carry = None):
        return initial_carry, nn.LayerNorm()(x)


class Tanh(Stateless):
    @nn.compact
    def __call__(self, x: Array, initial_carry: Carry = None):
        return initial_carry, jnp.tanh(x)


class ReLU(Stateless):
    @nn.compact
    def __call__(self, x: Array, initial_carry: Carry = None):
        return initial_carry, jax.nn.relu(x)


class LeakyReLU(Stateless):
    negative_slope: float = 0.01

    @nn.compact
    def __call__(self, x: Array, initial_carry: Carry = None):
        return initial_carry, jax.nn.leaky_relu(x, self.negative_slope)


class SiLU(Stateless):
    @nn.compact
    def __call__(self, x: Array, initial_carry: Carry = None):
        return initial_carry, jax.nn.silu(x)


class Readout(Stateless):
    """Adapts a head, which returns ``(output, aux)``, to the carry protocol."""

    module: nn.Module

    @nn.compact
    def __call__(self, x: Array, initial_carry: Carry = None):
        return initial_carry, self.module(x)
