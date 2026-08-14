from abc import abstractmethod
import jax
import jax.numpy as jnp
from flax import linen as nn

from memorax.utils.axes import (
    add_feature_axis,
    broadcast_done,
    get_input_shape,
    last,
    tail,
)
from memorax.utils.typing import Array, Carry

from .sequence_model import SequenceModel


class MemoroidCellBase(nn.Module):
    @abstractmethod
    def __call__(self, x: Array, **kwargs) -> Carry: ...

    @abstractmethod
    def binary_operator(self, a: Carry, b: Carry) -> Carry: ...

    @abstractmethod
    def read(self, h: Carry, x: Array, **kwargs) -> Array: ...

    @abstractmethod
    def initialize_carry(
        self, key: jax.Array, input_shape: tuple[int, ...]
    ) -> Carry: ...



class Memoroid(SequenceModel):
    recurrent = True
    reads = ("done",)

    cell: MemoroidCellBase

    def scan_fn(self, z, initial_carry, done):
        z = jax.tree.map(
            lambda c, e: jnp.concatenate([c, e], axis=1),
            initial_carry,
            z,
        )

        reset = jnp.concatenate([jnp.zeros((done.shape[0], 1)), done], axis=1)
        reset = add_feature_axis(reset)

        cell = self.cell

        @jax.vmap
        def binary_operator(lhs, rhs):
            lhs_carry, lhs_reset = lhs
            rhs_carry, rhs_reset = rhs

            combined = cell.binary_operator(lhs_carry, rhs_carry)

            out = jax.tree.map(
                lambda rc, c: jnp.where(broadcast_done(rhs_reset, rc), rc, c),
                rhs_carry,
                combined,
            )

            return out, jnp.maximum(lhs_reset, rhs_reset)

        h, _ = jax.lax.associative_scan(binary_operator, (z, reset), axis=1)

        next_carry = jax.tree.map(last, h)
        h = jax.tree.map(tail, h)
        return h, next_carry

    @nn.compact
    def __call__(
        self,
        inputs: Array,
        done: Array,
        initial_carry: Carry | None = None,
        **kwargs,
    ) -> tuple[Carry, Array]:
        if initial_carry is None:
            input_shape = get_input_shape(inputs)
            initial_carry = self.cell.initialize_carry(jax.random.key(0), input_shape)

        z = self.cell(inputs, **kwargs)
        h, next_carry = self.scan_fn(z, initial_carry, done)
        y = self.cell.read(h, inputs, **kwargs)

        return next_carry, y

    def initialize_carry(self, key: jax.Array, input_shape: tuple[int, ...]) -> Carry:
        return self.cell.initialize_carry(key, input_shape)
