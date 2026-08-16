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
    """Normalise the feature axis, learning the scale and shift or not.

    Both default to on, which is Flax's default and what every sequence that
    already held one was getting. RTRRL turns both off: the published cell
    normalises its output through ``nn.LayerNorm(use_bias=False,
    use_scale=False)``, and a learnable affine there would be parameters the
    algorithm it is a rebuild of does not have -- each of which RTRRL would
    then carry an eligibility trace for.
    """

    use_scale: bool = True
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: Array, initial_carry: Carry = None):
        normalized = nn.LayerNorm(use_scale=self.use_scale, use_bias=self.use_bias)(x)
        return initial_carry, normalized


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
