from typing import Mapping

import jax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.core.scope import CollectionFilter, PRNGSequenceFilter
from flax.typing import InOutScanAxis

from memorax.utils.axes import get_time_axis_and_input_shape, reset_carry
from memorax.utils.typing import Array, Carry

from .sequence_model import SequenceModel


class RNNCellBase(nn.recurrent.RNNCellBase):
    pass


class RNN(SequenceModel):
    recurrent = True
    reads = ("done",)

    cell: nn.RNNCellBase
    unroll: int = 1
    variable_axes: Mapping[CollectionFilter, InOutScanAxis] = FrozenDict()
    variable_broadcast: CollectionFilter = "params"
    variable_carry: CollectionFilter = False
    split_rngs: Mapping[PRNGSequenceFilter, bool] = FrozenDict({"params": False})

    def __call__(
        self,
        inputs: Array,
        done: Array,
        initial_carry: Carry | None = None,
        **kwargs,
    ) -> tuple[Carry, Array]:
        time_axis, input_shape = get_time_axis_and_input_shape(inputs)

        if initial_carry is None:
            initial_carry = self.cell.initialize_carry(jax.random.key(0), input_shape)

        carry: Carry = initial_carry

        def scan_fn(cell, carry, x, done):
            carry = reset_carry(
                done, carry, self.cell.initialize_carry(jax.random.key(0), input_shape)
            )
            carry, y = cell(carry, x)
            return carry, y

        scan = nn.transforms.scan(
            scan_fn,
            in_axes=time_axis,
            out_axes=time_axis,
            unroll=self.unroll,
            variable_axes=self.variable_axes,
            variable_broadcast=self.variable_broadcast,
            variable_carry=self.variable_carry,
            split_rngs=self.split_rngs,
        )

        carry, outputs = scan(self.cell, carry, inputs, done)

        return carry, outputs

    @nn.nowrap
    def initialize_carry(self, key: jax.Array, input_shape: tuple[int, ...]) -> Carry:
        return self.cell.initialize_carry(key, input_shape)
