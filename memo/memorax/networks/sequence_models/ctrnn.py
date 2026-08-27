"""CTRNN, and the RFLO approximation the published RTRRL differentiates it with.

The cell is one Euler step of a continuous-time recurrent network::

    v_t = [x_t, h_{t-1}, 1]
    u_t = (M * W) v_t
    h_t = h_{t-1} + (dt / tau) * (tanh(u_t) - h_{t-1})

One weight matrix carries the input block, the recurrent block and the bias
column together, which is how ``RTRRL-AAAI25/models/ctrnn.py`` writes it; ``M``
is the wiring mask, a constant, and ``tau`` is a learned per-unit time constant
the algorithm holds at or above a floor.

RFLO (Murray 2019) is what is left of forward-mode sensitivity when the part of
``dh_t/dh_{t-1}`` that runs through ``tanh`` is dropped and only the leak term
``1 - dt/tau`` is kept::

    p^W_{ij} <- (1 - dt/tau_i) p^W_{ij} + (dt/tau_i) tanh'(u_i) M_ij v_j
    p^tau_i  <- (1 - dt/tau_i) p^tau_i  + (dt/tau_i^2) (h_{t-1,i} - tanh(u_i))

Unlike the LRU and the RTU, this core is *not* structurally diagonal: unit i's
next state reads every other unit's previous state through the recurrent block
of ``W``. So the term RFLO drops is not zero here, and RFLO is a real
approximation rather than a second name for exact RTRL -- which is why this is
the one core in the repository that offers it. See
``docs/exact-recurrent-sensitivity.md``.

The approximation is expressed the way the other online cores express theirs:
the sensitivity is carried as state, and a phantom -- a value that is
identically zero but carries the derivative the trace stands for -- is added to
the carry so that ordinary autodiff through the step produces the online
gradient. What makes it RFLO rather than RTRL is *where* the phantom is added.
``u_t`` reads the stopped carry and nothing else, so no derivative reaches the
parameters through ``tanh``; the phantom rides only on the leak path, which
multiplies it by exactly ``1 - dt/tau``. The dropped term is dropped in one
place, in the forward, rather than being subtracted back out afterwards.

Rebuilt from ``RTRRL-AAAI25/models/ctrnn.py``. What was corrected on the way,
and why, is in ``docs/rtrrl-ctrnn-rflo-corrections.md``.
"""

from __future__ import annotations

import operator
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial, reduce
from typing import Any, Callable

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

#: The wirings this repository carries, and what each masks out.
#:
#: ``fully_connected`` is the published default and every reported CTRNN result
#: ran under it; it is a matrix of ones, so it is represented as no mask at all.
#: ``no_self`` removes each unit's connection to its own previous state and
#: nothing else.
#:
#: The published ``random`` and ``ncp`` wirings are not here. Both draw their
#: mask from the model's ``params`` rng and keep it in a Flax collection of its
#: own, and the boundary between a sequence and its differentiation carries only
#: ``params``. A mask that has to be redrawn at each call site is a mask two
#: call sites will eventually draw differently, so the key-dependent wirings are
#: left out rather than approximated.
WIRINGS: tuple[str, ...] = ("fully_connected", "no_self")


def wiring_mask(name: str, *, features: int, hidden_dim: int) -> Array | None:
    """The constant ``W`` is multiplied by, or ``None`` when it is all ones.

    The column layout is ``[input, hidden, bias]``, so the recurrent block a
    self-connection lives on is columns ``features .. features + hidden_dim``.
    The published ``fully_connected_no_self`` places its identity on the *last*
    ``hidden_dim`` columns instead, which under this layout sits one column to
    the right of the recurrent block: it removes unit ``i``'s reading of unit
    ``i + 1``, and for the last unit it removes the bias, which the mask builder
    then restores -- leaving that unit fully self-connected. The name says which
    connection it means, so the diagonal is placed where the name says.
    """

    if name == "fully_connected":
        return None
    if name != "no_self":
        listed = ", ".join(WIRINGS)
        raise ValueError(f"unknown wiring {name!r}; carried: {listed}")
    mask = jnp.ones((hidden_dim, features + hidden_dim + 1))
    return mask.at[:, features : features + hidden_dim].add(-jnp.eye(hidden_dim))


@struct.dataclass
class CTRNNConfig:
    """Everything the cell reads that does not change during a run.

    ``tau_min`` and ``tau_max`` bound the initial draw. ``tau_floor`` is not
    read by the forward at all: it is the set the parameter is projected back
    onto after a step, which the cell states in :meth:`CTRNNCell.constrain` for
    whichever learner is doing the stepping.
    """

    features: int
    hidden_dim: int
    dt: float = 1.0
    wiring: str = "fully_connected"
    tau_min: float = 1.0
    tau_max: float = 5.0
    tau_floor: float = 1.0
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    activation_fn: Callable = struct.field(pytree_node=False, default=jnp.tanh)


class CTRNNCell(RNNCellBase):
    """One Euler step, its RFLO sensitivity, and the floor ``tau`` stays above."""

    config: CTRNNConfig

    @property
    def num_feature_axes(self) -> int:
        return 1

    def setup(self):
        shape = (
            self.config.hidden_dim,
            self.config.features + self.config.hidden_dim + 1,
        )
        self.W = self.param(
            "W",
            nn.initializers.lecun_normal(in_axis=-1, out_axis=-2),
            shape,
            self.config.param_dtype,
        )
        self.tau = self.param(
            "tau",
            partial(
                jax.random.uniform,
                minval=self.config.tau_min,
                maxval=self.config.tau_max,
            ),
            (self.config.hidden_dim,),
            self.config.param_dtype,
        )

    # ------------------------------------------------------------- the forward
    def _mask(self) -> Array | None:
        return wiring_mask(
            self.config.wiring,
            features=self.config.features,
            hidden_dim=self.config.hidden_dim,
        )

    def _weights(self) -> Array:
        """``W`` as the network uses it, which is masked wherever a mask exists.

        The mask is a constant and carries no gradient of its own, so a masked
        entry of ``W`` receives none either -- in the forward and in the
        sensitivity alike. The published implementation applies it in the
        forward only and computes the RFLO trace from the raw matrix, so a
        masked run there trains about a different network than the one it steps.
        """

        mask = self._mask()
        if mask is None:
            return self.W
        return jax.lax.stop_gradient(mask) * self.W

    def _preactivation(self, carry: Array, inputs: Array) -> tuple[Array, Array]:
        """``u_t``, and the row it was formed from."""

        ones = jnp.ones(inputs.shape[:-1] + (1,), dtype=inputs.dtype)
        row = jnp.concatenate([inputs, carry, ones], axis=-1)
        return row @ self._weights().T, row

    def _integrate(self, leak: Array, activated: Array) -> Array:
        """The Euler step, over whichever carry the leak path is reading."""

        return leak + (self.config.dt / self.tau) * (activated - leak)

    @nn.compact
    def __call__(self, carry: Array, inputs: Array) -> tuple[Array, Array]:
        preactivation, _ = self._preactivation(carry, inputs)
        out = self._integrate(carry, self.config.activation_fn(preactivation))
        return out, out

    @nn.nowrap
    def initialize_carry(self, rng: jax.Array, input_shape: tuple[int, ...]) -> Array:
        # ``rng`` rather than ``key`` because ``RNNCellBase`` names it that, and
        # every caller passes it positionally. The state starts at zero, so
        # there is nothing to draw.
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
        width = self.config.features + hidden + 1
        return {
            "W": jnp.zeros((*batch_dims, hidden, width)),
            "tau": jnp.zeros((*batch_dims, hidden)),
        }

    def compute_phantom(self, sensitivity: dict[str, Array]) -> Array:
        """A zero whose derivative is the carried sensitivity.

        ``theta - stop_gradient(theta)`` is zero and differentiates to one, so
        adding this to the carry adds ``p`` to the carry's derivative without
        moving any number the forward computes. Each trace's leading axes are
        the streams and its next axis is the unit the sensitivity belongs to,
        which is the axis that survives; the parameter's own axes are summed.
        """

        params = {"W": self.W, "tau": self.tau}
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

        The carry enters twice and the two readings are not interchangeable.
        ``held`` is where ``u_t`` reads the previous state: stopped, so no
        gradient runs back through ``tanh`` and the cross-unit block of the
        recurrent Jacobian is gone. ``leak`` is the same value plus the phantom,
        and is the only path the carried sensitivity survives on -- which
        multiplies it by ``1 - dt/tau``. That pair is the approximation, and the
        recurrence below is the same statement in closed form.
        """

        held = jax.lax.stop_gradient(carry)
        leak = held + phantom
        preactivation, row = self._preactivation(held, inputs)
        activated = self.config.activation_fn(preactivation)
        out = self._integrate(leak, activated)

        rate = jax.lax.stop_gradient(self.config.dt / self.tau)
        decay = 1.0 - rate
        slope = jax.grad(lambda x: self.config.activation_fn(x).sum())(
            jax.lax.stop_gradient(preactivation)
        )
        immediate = rate[..., None] * slope[..., None] * row[..., None, :]
        mask = self._mask()
        if mask is not None:
            immediate = immediate * jax.lax.stop_gradient(mask)
        # `rate / constant` is `dt / tau**2`, which is what differentiating
        # `(dt/tau)(phi(u) - h)` by `tau` leaves once `phi(u)` is held.
        constant = jax.lax.stop_gradient(self.tau)
        next_sensitivity = {
            "W": decay[..., None] * sensitivity["W"] + immediate,
            "tau": decay * sensitivity["tau"]
            + (rate / constant) * (held - jax.lax.stop_gradient(activated)),
        }
        return out, out, next_sensitivity

    @nn.nowrap
    def constrain(self, params: dict[str, Any]) -> dict[str, Any]:
        """The set the parameters are projected back onto after a step.

        ``tau`` below ``dt`` makes ``1 - dt/tau`` negative and the Euler step
        overshoot; at zero it divides by zero. The published implementation
        clips it to one after every update, and this is that clip -- stated by
        the component whose parameter it is rather than by whichever learner
        happens to be stepping it.
        """

        return {**params, "tau": jnp.clip(params["tau"], min=self.config.tau_floor)}


# ------------------------------------------------------------- differentiation
@dataclass(frozen=True)
class CTRNNRflo:
    """RFLO: forward sensitivity with the ``tanh`` path of ``dh/dh`` dropped."""

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
        return CTRNNRflo(core)
    return TruncatedBPTT(core)


CTRNN_DIFFERENTIATION_FAMILY = ComponentFamily(
    branches={"rflo": (), "tbptt": ()},
    construct=_construct_differentiation,
)


def _rflo(core, inputs, done, carry, state, **kwargs):
    """Run the CTRNN scan and its RFLO sensitivity recurrence beside it."""

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
