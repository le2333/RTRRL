"""LSTM, and the RFLO-shaped online credit the streaming update spends on it.

The cell is the standard LSTM, written with one packed matrix per gate so that
the input block, the recurrent block and the bias column sit together the way
``ctrnn.py`` writes them::

    v_t = [x_t, h_{t-1}, 1]
    i_t = sigma(W_i v_t)      f_t = sigma(W_f v_t)
    g_t = tanh(W_g v_t)       o_t = sigma(W_o v_t)
    c_t = f_t * c_{t-1} + i_t * g_t
    h_t = o_t * tanh(c_t)

RFLO (Murray 2019) is forward sensitivity with the part of the state Jacobian
that runs through the recurrent nonlinearity dropped, and only the leak kept.
On the CTRNN the leak is ``1 - dt/tau``. Here the state that carries memory
across a transition is ``c``, its leak is the forget gate, and the term dropped
is the whole path by which ``c_{t-1}`` reaches ``c_t`` through ``h_{t-1}`` and
the four gates::

    dc_t/dtheta = f_t * dc_{t-1}/dtheta
                + (dc_t/dh_{t-1}) (dh_{t-1}/dtheta)      <- dropped
                + dc_t/dtheta|_{h,c held}

so the carried trace is

    p^{W_f}_{kj} <- f_{t,k} p^{W_f}_{kj} + c_{t-1,k} sigma'(a^f_{t,k}) v_{t,j}
    p^{W_i}_{kj} <- f_{t,k} p^{W_i}_{kj} + g_{t,k}   sigma'(a^i_{t,k}) v_{t,j}
    p^{W_g}_{kj} <- f_{t,k} p^{W_g}_{kj} + i_{t,k}   tanh'(a^g_{t,k})  v_{t,j}

and ``W_o`` carries no trace at all: ``dc_t/dW_o`` is identically zero, because
the output gate does not enter the cell state. Its whole gradient is the
instantaneous one through ``h_t = o_t * tanh(c_t)``, which ordinary autodiff
through the step already produces -- exactly, not approximately, under the same
approximation the other three are taken under. Three traces and not four is a
statement about the equations rather than a saving.

The dropped block is genuinely there: unit ``k``'s next cell state reads every
other unit's previous hidden state through the recurrent columns of the four
matrices. So RFLO here is a real approximation, as on the CTRNN and unlike on
the LRU and the RTU, whose cross-unit block is identically zero and for which
this repository offers exact RTRL instead. See
``docs/exact-recurrent-sensitivity.md``.

This is the same trace e-prop (Bellec et al. 2020) derives for the LSTM, which
is what makes it the LSTM's RFLO rather than an analogy to it: both drop every
path through the recurrent input of the gates and keep the multiplicative
carry, and for this cell that carry is ``f_t``. What the two call it differs;
the recurrence does not. ``docs/rtrrl-lstm-rflo.md`` sets the derivation out and
says where a reader of either paper should expect a different letter.

The approximation is expressed the way the other online cores express theirs.
The sensitivity is carried as state, and a phantom -- a value that is
identically zero and whose derivative is the trace it stands for -- is added to
the carry so that ordinary autodiff through the step produces the online
gradient. What makes it RFLO rather than RTRL is *where*: the gates read
``stop_gradient(h_{t-1})``, so nothing reaches the parameters through them, and
the phantom rides only ``c_{t-1}``, which the step multiplies by exactly
``f_t``. The dropped term is dropped in the forward rather than subtracted back
out afterwards.

Unlike ``ctrnn.py`` this cell has no published RTRRL implementation to answer
to: the paper's LSTM appears as a DRQN core and is differentiated by truncated
backpropagation, so there is nothing to hold a parity suite against and nothing
to correct. What the equations are held to is the equations, in
``tests/test_lstm_rflo.py``.
"""

from __future__ import annotations

import operator
from contextlib import nullcontext
from dataclasses import dataclass
from functools import reduce
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from flax.typing import Dtype

from memorax.building import ComponentFamily
from memorax.networks.differentiation import TruncatedBPTT
from memorax.utils.axes import (
    broadcast_done,
    get_time_axis_and_input_shape,
    reset_carry,
)
from memorax.utils.typing import Array

from .rnn import RNNCellBase

#: The three gates whose weights the cell state carries a trace for, in the
#: order the parameter tree sorts them. ``W_o`` is deliberately absent: the
#: output gate does not enter ``c``, so its trace would be a zero the scan
#: carries and the phantom contracts to nothing.
TRACED: tuple[str, ...] = ("W_f", "W_g", "W_i")


def packed_kernel(bias: float):
    """One gate's ``[input, hidden, bias]`` matrix, drawn as its two parts.

    ``lecun_normal`` over the connection columns, and a constant on the bias
    column -- with the fan-in taken over the connections alone, since the bias
    column is not one. Packing the bias into the matrix is ``ctrnn.py``'s
    layout and is what lets one trace per gate have the shape of one gate's
    parameters; drawing it from the same normal as the connections would make
    the forget gate's initial bias a small random number, which is the one
    place in an LSTM where that choice is known to matter.
    """

    def initialize(key, shape, dtype=jnp.float32):
        rows, width = shape
        connections = nn.initializers.lecun_normal(in_axis=-1, out_axis=-2)(
            key, (rows, width - 1), dtype
        )
        column = jnp.full((rows, 1), bias, dtype)
        return jnp.concatenate([connections, column], axis=-1)

    return initialize


class Preactivations(NamedTuple):
    """``W_* v_t`` for the four gates, before any nonlinearity is applied."""

    input: Array
    forget: Array
    candidate: Array
    output: Array


@struct.dataclass
class LSTMCarry:
    """What one stream carries between transitions.

    Two arrays and not one, and the difference between them is the whole of
    where RFLO's approximation sits: ``cell`` is the state the trace is a
    derivative of and the state the phantom rides, ``hidden`` is what the gates
    read and is stopped before they read it.
    """

    cell: Array
    hidden: Array


@struct.dataclass
class LSTMConfig:
    """Everything the cell reads that does not change during a run.

    ``forget_bias`` is the bias column the forget gate's matrix is drawn with.
    One is the standard choice and the reason for it is this cell's own
    recurrence: the forget gate is the factor the trace is multiplied by every
    transition, so a gate initialised near ``0.5`` halves the trace per step
    and an online method that never unrolls has nothing else to recover the
    horizon with.
    """

    features: int
    hidden_dim: int
    forget_bias: float = 1.0
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32


class LSTMCell(RNNCellBase):
    """One LSTM step, and the RFLO sensitivity of its cell state."""

    config: LSTMConfig

    @property
    def num_feature_axes(self) -> int:
        return 1

    def setup(self):
        shape = (
            self.config.hidden_dim,
            self.config.features + self.config.hidden_dim + 1,
        )
        # Four matrices rather than one stacked ``(4H, width)``: a trace has
        # the shape of the parameters of the unit it belongs to, and unit ``k``
        # owns row ``k`` of each gate. Stacked, the trace would either carry a
        # gate axis it is diagonal in or lose which gate a column credits.
        self.W_i = self.param("W_i", packed_kernel(0.0), shape, self.config.param_dtype)
        self.W_f = self.param(
            "W_f",
            packed_kernel(self.config.forget_bias),
            shape,
            self.config.param_dtype,
        )
        self.W_g = self.param("W_g", packed_kernel(0.0), shape, self.config.param_dtype)
        self.W_o = self.param("W_o", packed_kernel(0.0), shape, self.config.param_dtype)

    # ------------------------------------------------------------- the forward
    def _row(self, hidden: Array, inputs: Array) -> Array:
        """``v_t = [x_t, h_{t-1}, 1]``, the one row every gate reads."""

        ones = jnp.ones(inputs.shape[:-1] + (1,), dtype=inputs.dtype)
        return jnp.concatenate([inputs, hidden, ones], axis=-1)

    def _preactivations(
        self, hidden: Array, inputs: Array
    ) -> tuple[Preactivations, Array]:
        """The four ``W_* v_t``, and the row they were formed from."""

        row = self._row(hidden, inputs)
        return (
            Preactivations(
                input=row @ self.W_i.T,
                forget=row @ self.W_f.T,
                candidate=row @ self.W_g.T,
                output=row @ self.W_o.T,
            ),
            row,
        )

    def _step(
        self, preactivations: Preactivations, cell: Array
    ) -> tuple[LSTMCarry, Array]:
        """The state update, over whichever cell state the leak path reads."""

        forget = jax.nn.sigmoid(preactivations.forget)
        written = jax.nn.sigmoid(preactivations.input) * jnp.tanh(
            preactivations.candidate
        )
        advanced = forget * cell + written
        hidden = jax.nn.sigmoid(preactivations.output) * jnp.tanh(advanced)
        return LSTMCarry(cell=advanced, hidden=hidden), hidden

    @nn.compact
    def __call__(self, carry: LSTMCarry, inputs: Array) -> tuple[LSTMCarry, Array]:
        preactivations, _ = self._preactivations(carry.hidden, inputs)
        return self._step(preactivations, carry.cell)

    @nn.nowrap
    def initialize_carry(
        self, rng: jax.Array, input_shape: tuple[int, ...]
    ) -> LSTMCarry:
        # ``rng`` rather than ``key`` because ``RNNCellBase`` names it that, and
        # every caller passes it positionally. Both states start at zero, so
        # there is nothing to draw.
        del rng
        *batch_dims, _ = input_shape
        empty = jnp.zeros((*batch_dims, self.config.hidden_dim))
        return LSTMCarry(cell=empty, hidden=empty)

    # --------------------------------------------------------- the sensitivity
    @nn.nowrap
    def initialize_sensitivity(
        self, key: jax.Array, input_shape: tuple[int, ...]
    ) -> dict[str, Array]:
        del key
        *batch_dims, _ = input_shape
        hidden = self.config.hidden_dim
        width = self.config.features + hidden + 1
        return {name: jnp.zeros((*batch_dims, hidden, width)) for name in TRACED}

    def compute_phantom(self, sensitivity: dict[str, Array]) -> Array:
        """A zero whose derivative is the carried sensitivity of ``c``.

        ``theta - stop_gradient(theta)`` is zero and differentiates to one, so
        adding this to the cell state adds ``p`` to that state's derivative
        without moving any number the forward computes. Each trace's leading
        axes are the streams and its next axis is the unit the sensitivity
        belongs to, which is the axis that survives; the parameter's own
        remaining axes are summed.
        """

        params = {"W_i": self.W_i, "W_f": self.W_f, "W_g": self.W_g, "W_o": self.W_o}
        contributions = []
        for name in sorted(sensitivity):
            value = params[name]
            trace = sensitivity[name]
            summed = tuple(range(trace.ndim - (value.ndim - 1), trace.ndim))
            contributions.append(
                jnp.sum(trace * (value - jax.lax.stop_gradient(value)), axis=summed)
            )
        return reduce(operator.add, contributions)

    def local_jacobian(
        self,
        carry: LSTMCarry,
        inputs: Array,
        phantom: Array,
        sensitivity: dict[str, Array],
    ) -> tuple[LSTMCarry, Array, dict[str, Array]]:
        """One step, differentiated by RFLO, with the trace advanced beside it.

        The carry's two halves are read differently, and that is the
        approximation. ``hidden`` is stopped before the gates read it, so no
        derivative runs back through ``sigma`` or ``tanh`` and the cross-unit
        block of ``dc/dc`` is gone. ``cell`` is stopped too, and the phantom is
        added to it -- the one path the carried sensitivity survives on, which
        the step multiplies by exactly ``f_t``. The recurrence below is the same
        statement in closed form.

        The inputs are not stopped, so whatever stands in front of the cell
        receives the step's own Jacobian; nor are the four matrices, so this
        transition's explicit derivative is autodiff's, and it enters the trace
        only for the transitions that come after it.
        """

        held_cell = jax.lax.stop_gradient(carry.cell)
        held_hidden = jax.lax.stop_gradient(carry.hidden)
        preactivations, row = self._preactivations(held_hidden, inputs)
        advanced, output = self._step(preactivations, held_cell + phantom)

        held_row = jax.lax.stop_gradient(row)[..., None, :]
        # The activations again, stopped: the trace is carried state and not a
        # thing to be differentiated. Each slope is written from the activation
        # it belongs to -- sigma' = sigma(1 - sigma), tanh' = 1 - tanh^2 --
        # rather than taken by autodiff, because the values are already here.
        forget = jax.lax.stop_gradient(jax.nn.sigmoid(preactivations.forget))
        opened = jax.lax.stop_gradient(jax.nn.sigmoid(preactivations.input))
        candidate = jax.lax.stop_gradient(jnp.tanh(preactivations.candidate))
        decay = forget[..., None]
        immediate = {
            # d c_t / d W_f, with `c_{t-1}` and `h_{t-1}` held.
            "W_f": (held_cell * forget * (1.0 - forget))[..., None] * held_row,
            # d c_t / d W_i.
            "W_i": (candidate * opened * (1.0 - opened))[..., None] * held_row,
            # d c_t / d W_g.
            "W_g": (opened * (1.0 - candidate**2))[..., None] * held_row,
        }
        next_sensitivity = {
            name: decay * sensitivity[name] + immediate[name] for name in sensitivity
        }
        return advanced, output, next_sensitivity


# ------------------------------------------------------------- differentiation
@dataclass(frozen=True)
class LSTMRflo:
    """RFLO: forward sensitivity of ``c`` with the ``h_{t-1}`` path dropped."""

    core: Any

    def initialize(self, key, input_shape):
        return self.core.cell.initialize_sensitivity(key, input_shape)

    def initialization(self):
        return nullcontext()

    def __call__(self, params, inputs, done, carry, state):
        return self.core.apply(
            {"params": params},
            inputs,
            done,
            carry,
            state,
            method=_rflo,
        )


def _construct_differentiation(selection, builder, *, core):
    del builder
    if selection.kind == "rflo":
        return LSTMRflo(core)
    return TruncatedBPTT(core)


LSTM_DIFFERENTIATION_FAMILY = ComponentFamily(
    branches={"rflo": (), "tbptt": ()},
    construct=_construct_differentiation,
)


def _rflo(core, inputs, done, carry, state, **kwargs):
    """Run the LSTM scan and its RFLO sensitivity recurrence beside it."""

    time_axis, input_shape = get_time_axis_and_input_shape(inputs)
    initial_carry = core.cell.initialize_carry(jax.random.key(0), input_shape)

    def scan_fn(cell, recurrent, x, done_t):
        cell_carry, sensitivity = recurrent
        # An ending clears both before anything reads them, so the phantom a
        # restarted stream carries is the zero of a sequence that has not run.
        sensitivity = jax.tree.map(
            lambda value: jnp.where(broadcast_done(done_t, value), 0, value),
            sensitivity,
        )
        cell_carry = reset_carry(done_t, cell_carry, initial_carry)
        phantom = cell.compute_phantom(sensitivity)
        next_carry, output, next_state = cell.local_jacobian(
            cell_carry, x, phantom, sensitivity
        )
        return (next_carry, next_state), output

    scan = nn.transforms.scan(
        scan_fn,
        in_axes=time_axis,
        out_axes=time_axis,
        unroll=core.unroll,
        variable_axes=core.variable_axes,
        variable_broadcast=core.variable_broadcast,
        variable_carry=core.variable_carry,
        split_rngs=core.split_rngs,
    )
    (next_carry, next_state), outputs = scan(core.cell, (carry, state), inputs, done)
    return next_carry, outputs, next_state
