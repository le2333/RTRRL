"""The leaves a sequence is built from, each doing one thing.

A layer, a normalisation and an activation are three components rather than one
block that does all three, because the three published backbones order them
differently and any block that fused two of them would fit at most one of the
three. It is also what makes a network describable as data later: a fixed
arrangement cannot be written down as a list.

``reads`` is what a component asks for beyond its input. Nothing here asks for
anything, which is why nothing here has a second argument to accept: a component
that is handed an ending it cannot use is a component that had to be written to
ignore one.
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
    """One affine map. The nonlinearity, if any, is the next component."""

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
    """The last component, whose output is whatever its module answers with.

    Value and policy heads predate this protocol and hand back a pair of their
    own; wrapping one is cheaper than teaching fifteen of them to carry a carry
    they have no use for.
    """

    module: nn.Module

    @nn.compact
    def __call__(self, x: Array, initial_carry: Carry = None):
        return initial_carry, self.module(x)
