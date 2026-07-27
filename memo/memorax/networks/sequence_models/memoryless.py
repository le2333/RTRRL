"""A torso with nothing to carry, so that memory can be ablated.

Every other torso in this package remembers something, which makes "how much of
the score is memory" a question none of them can answer. This one fills the same
slot and keeps no state: ``done`` and the incoming carry are accepted and
ignored, because there is nothing to reset and nothing to hand on.

Two dense layers, each followed by a normalisation and an activation, which is
the torso StreamAC as published uses on MuJoCo. Under either credit setting it
behaves the same, and that is not a special case being handled: exact recurrent
credit and truncated credit differ only in what a carry contributes, and there
is no carry.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
from flax import linen as nn
from flax import struct

from memorax.utils.typing import Array, Carry, Key

from .sequence_model import SequenceModel


@struct.dataclass
class MemorylessConfig:
    features: int
    hidden_dim: int
    activation_fn: Callable = struct.field(pytree_node=False, default=jax.nn.leaky_relu)


class Memoryless(SequenceModel):

    config: MemorylessConfig

    @nn.compact
    def __call__(
        self,
        inputs: Array,
        done: Array | None = None,
        initial_carry: Carry | None = None,
        **kwargs,
    ) -> tuple[Carry, Array]:
        del done, initial_carry, kwargs
        x = nn.Dense(self.config.hidden_dim)(inputs)
        x = nn.LayerNorm()(x)
        x = self.config.activation_fn(x)
        x = nn.Dense(self.config.features)(x)
        x = nn.LayerNorm()(x)
        x = self.config.activation_fn(x)
        return None, x

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple) -> None:
        del key, input_shape
