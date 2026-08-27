"""A dense linear state-space core, and RFLO as what its off-diagonal costs.

The cell is the state-space layer before anyone diagonalised it::

    v_t = [x_t, 1]
    h_t = A h_{t-1} + B v_t
    y_t = tanh(C h_t)

``A`` is a full ``(hidden, hidden)`` matrix, ``B`` carries the input block and
the bias column together in ``ctrnn.py``'s layout, and ``C`` is the readout the
heads see.

This exists to make one comparison possible. ``lru.py`` and ``rtu.py`` are the
same recurrence with ``A`` constrained to be diagonal, and
``docs/exact-recurrent-sensitivity.md`` argues that on those two RFLO is not an
approximation at all -- the cross-unit block of ``dh/dh`` is identically zero,
so there is nothing to drop and exact RTRL costs what RFLO would. That argument
is about the *structure*, not about linearity, and this core is the control it
was missing: the same linear step with the structure removed, where the block
is present and RFLO is again an approximation. On this cell

    exact:  S_t = A S_{t-1} + dh_t/dtheta
    RFLO:   S_t = diag(A) * S_{t-1} + dh_t/dtheta

and the difference is the off-diagonal of ``A`` applied to the previous
sensitivity -- literally the cross-unit block, with no nonlinearity in the way
to obscure which term is which. Set the off-diagonal to zero and the two
recurrences are one, which is the LRU's case recovered as a limit rather than
asserted.

``C`` carries no trace: ``dh_t/dC`` is identically zero, so its whole gradient
is the instantaneous one through ``y_t = tanh(C h_t)``, which ordinary autodiff
produces. That is exact rather than approximate, and it is the same statement
``lstm.py`` makes about its output gate.

``A`` is the one parameter with a domain. A linear recurrence with a spectral
radius above one diverges, and unlike the LRU -- whose parameterisation puts
``|lambda| < 1`` in the exponent and cannot leave it -- a free matrix reaches
that domain on an ordinary gradient step. The cell states the set as a bound on
the induced infinity-norm, ``max_i sum_j |A_ij| <= spectral_bound``, which
bounds the spectral radius, is a projection rather than a search, and costs one
row-wise sum. The torso projects onto whatever set the kernel names, so this
is stated here rather than by whichever learner is doing the stepping.

The approximation is expressed the way the other online cores express theirs: a
sensitivity carried as state, and a phantom -- identically zero, carrying the
derivative the trace stands for -- added to the carry so that autodiff through
the step produces the online gradient. What makes it RFLO rather than RTRL is
where the phantom is added. The step is written as its diagonal and
off-diagonal halves; the off-diagonal half reads the stopped carry, so nothing
propagates across units, and the phantom rides only the diagonal, which
multiplies it by exactly ``diag(A)``. The dropped term is dropped in the
forward rather than subtracted back out.

``docs/rtrrl-dense-ssm-rflo.md`` derives it. ``tests/test_dense_ssm_rflo.py``
holds the cell against the equations, against autodiff, and -- on a matrix
whose off-diagonal is zero -- against exact RTRL, which is the LRU's argument
run as a measurement.
"""

from __future__ import annotations

import operator
from contextlib import nullcontext
from dataclasses import dataclass
from functools import reduce
from typing import Any

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

#: The two parameters the state's sensitivity is carried for. ``C`` is absent:
#: it does not enter ``h``, so its trace would be a zero the scan carries.
TRACED: tuple[str, ...] = ("A", "B")


def contract(matrix: Array, bound: float) -> Array:
    """Scale each row of ``A`` until its absolute sum is within ``bound``.

    ``max_i sum_j |A_ij|`` is the induced infinity-norm and bounds the spectral
    radius, so a matrix inside this set has a recurrence that decays. Rows
    already inside are left exactly where they are -- this is the projection
    onto the set and not a normalisation, so a run whose ``A`` never reaches
    the boundary is a run this never touches.
    """

    rows = jnp.sum(jnp.abs(matrix), axis=-1, keepdims=True)
    return matrix * jnp.minimum(1.0, bound / jnp.maximum(rows, 1e-12))


def contracted(bound: float):
    """``lecun_normal``, projected onto the set the recurrence is stable on.

    The draw is scaled because ``lecun_normal``'s row sums grow like the square
    root of the width: at ``hidden_dim = 32`` a drawn row already sums to about
    ``4.5``, so an unprojected initial ``A`` would diverge before the first
    update rather than after some number of them.
    """

    def initialize(key, shape, dtype=jnp.float32):
        drawn = nn.initializers.lecun_normal(in_axis=-1, out_axis=-2)(key, shape, dtype)
        return contract(drawn, bound)

    return initialize


@struct.dataclass
class DenseSSMConfig:
    """Everything the cell reads that does not change during a run.

    ``spectral_bound`` is read by the initial draw and by
    :meth:`DenseSSMCell.constrain`, and by nothing in the forward: it is the
    set the parameter lives in rather than a term in the step.
    """

    features: int
    hidden_dim: int
    spectral_bound: float = 0.9
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32


class DenseSSMCell(RNNCellBase):
    """One linear state step, its readout, and the RFLO trace of the state."""

    config: DenseSSMConfig

    @property
    def num_feature_axes(self) -> int:
        return 1

    def setup(self):
        hidden = self.config.hidden_dim
        self.A = self.param(
            "A",
            contracted(self.config.spectral_bound),
            (hidden, hidden),
            self.config.param_dtype,
        )
        self.B = self.param(
            "B",
            nn.initializers.lecun_normal(in_axis=-1, out_axis=-2),
            (hidden, self.config.features + 1),
            self.config.param_dtype,
        )
        self.C = self.param(
            "C",
            nn.initializers.lecun_normal(in_axis=-1, out_axis=-2),
            (hidden, hidden),
            self.config.param_dtype,
        )

    # ------------------------------------------------------------- the forward
    def _row(self, inputs: Array) -> Array:
        """``v_t = [x_t, 1]``. The state has its own matrix and is not in here."""

        ones = jnp.ones(inputs.shape[:-1] + (1,), dtype=inputs.dtype)
        return jnp.concatenate([inputs, ones], axis=-1)

    def _read(self, state: Array) -> Array:
        return jnp.tanh(state @ self.C.T)

    @nn.compact
    def __call__(self, carry: Array, inputs: Array) -> tuple[Array, Array]:
        advanced = carry @ self.A.T + self._row(inputs) @ self.B.T
        return advanced, self._read(advanced)

    @nn.nowrap
    def initialize_carry(self, rng: jax.Array, input_shape: tuple[int, ...]) -> Array:
        # ``rng`` rather than ``key`` because ``RNNCellBase`` names it that, and
        # every caller passes it positionally. The state starts at zero.
        del rng
        *batch_dims, _ = input_shape
        return jnp.zeros((*batch_dims, self.config.hidden_dim))

    # --------------------------------------------------------- the sensitivity
    @nn.nowrap
    def initialize_sensitivity(
        self, key: jax.Array, input_shape: tuple[int, ...]
    ) -> dict[str, Array]:
        del key
        *batch_dims, _ = input_shape
        hidden = self.config.hidden_dim
        return {
            "A": jnp.zeros((*batch_dims, hidden, hidden)),
            "B": jnp.zeros((*batch_dims, hidden, self.config.features + 1)),
        }

    def compute_phantom(self, sensitivity: dict[str, Array]) -> Array:
        """A zero whose derivative is the carried sensitivity of the state.

        ``theta - stop_gradient(theta)`` is zero and differentiates to one, so
        adding this to the carry adds ``p`` to the carry's derivative without
        moving any number the forward computes. Each trace's leading axes are
        the streams and its next axis is the unit the sensitivity belongs to,
        which is the axis that survives; the parameter's own remaining axes are
        summed.
        """

        params = {"A": self.A, "B": self.B, "C": self.C}
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
        carry: Array,
        inputs: Array,
        phantom: Array,
        sensitivity: dict[str, Array],
    ) -> tuple[Array, Array, dict[str, Array]]:
        """One step, differentiated by RFLO, with the trace advanced beside it.

        The step is written as ``diag(A) h + (A - diag(A)) h + B v``, which is
        ``A h + B v`` and is arranged this way so the two halves can be read
        differently. The off-diagonal half sees the stopped carry, so no
        sensitivity propagates across units; the diagonal half sees the carry
        plus the phantom, so what survives is multiplied by exactly
        ``diag(A)``. That pair is the approximation, and the recurrence below is
        the same statement in closed form.

        The inputs are not stopped, so whatever stands in front of the cell
        receives the step's own Jacobian; nor are the matrices, so this
        transition's explicit derivative is autodiff's and enters the trace only
        for the transitions after it.
        """

        held = jax.lax.stop_gradient(carry)
        row = self._row(inputs)
        diagonal = jnp.diagonal(self.A)
        crossing = self.A - jnp.diag(diagonal)
        advanced = diagonal * (held + phantom) + held @ crossing.T + row @ self.B.T

        # The trace is carried state and not a thing to be differentiated.
        # ``held`` is stopped already; ``row`` is not, because the inputs are
        # what the step's own Jacobian has to reach.
        decay = jax.lax.stop_gradient(diagonal)[..., None]
        held_state = held[..., None, :]
        held_row = jax.lax.stop_gradient(row)[..., None, :]
        next_sensitivity = {
            # d h_i / d A_ij is h_{t-1,j}, for every j including j == i.
            "A": decay * sensitivity["A"] + held_state,
            # d h_i / d B_ij is v_{t,j}.
            "B": decay * sensitivity["B"] + held_row,
        }
        return advanced, self._read(advanced), next_sensitivity

    @nn.nowrap
    def constrain(self, params: dict[str, Any]) -> dict[str, Any]:
        """The set the parameters are projected back onto after a step.

        Only ``A`` has one. ``B`` and ``C`` are read once per transition and a
        large value in either is a large number rather than a divergence; ``A``
        is applied once per transition *for the length of the episode*, so a
        spectral radius above one is an episode whose state grows without
        bound and whose trace grows with it.
        """

        return {**params, "A": contract(params["A"], self.config.spectral_bound)}


# ------------------------------------------------------------- differentiation
@dataclass(frozen=True)
class DenseSSMRflo:
    """RFLO: forward sensitivity with the off-diagonal of ``A`` dropped."""

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
        return DenseSSMRflo(core)
    return TruncatedBPTT(core)


DENSE_SSM_DIFFERENTIATION_FAMILY = ComponentFamily(
    branches={"rflo": (), "tbptt": ()},
    construct=_construct_differentiation,
)


def _rflo(core, inputs, done, carry, state, **kwargs):
    """Run the state-space scan and its RFLO sensitivity recurrence beside it."""

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
