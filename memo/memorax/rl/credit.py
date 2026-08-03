"""How a recurrent parameter learns from a step it can no longer see.

Two answers, behind one interface, so that choosing between them is a setting
rather than a rewrite. Neither computes anything here: the Jacobian of a carry
with respect to its parameters belongs to the cell that owns the recurrence and
every cell in this package already implements it, while truncating the credit is
the absence of a computation. This file only says which of a cell's methods
stands for credit, so a kernel can ask without naming the cell.

``ExactRTRL`` carries a sensitivity forward, so a parameter is credited for its
effect on every carry it ever helped produce. ``TruncatedBPTT`` carries nothing
and lets the gradient stop at the incoming carry, which is one step of
backpropagation through time. Swapping one for the other is the recurrent half
of a gradient ablation: same objective, same update rule, and the recurrent path
either credited exactly or cut to a step.

Neither is the published StreamAC, which is feedforward and so has no carry to
credit either way. Truncating is what the recurrent implementations this
repository inherited do, and they do it by construction rather than by setting.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax

CREDITS = ("rtrl", "tbptt")


def _delegate_rtu_init_forward(next_fun, args, kwargs, context):
    """Initialise an RTU through its Jacobian, since that is what will run.

    Flax builds parameters by tracing whichever method it is handed. Exact RTRL
    never calls the cell's plain forward, so tracing that one at initialisation
    would shape the parameters against a method the run does not use.
    """

    module = context.module
    if context.method_name == "__call__" and _is_rtu(module):
        carry, inputs = args
        sensitivity = module.initialize_sensitivity(jax.random.key(0), inputs.shape)
        if sensitivity is None:
            raise TypeError("RTUCell initialization requires local sensitivity")
        next_carry, output, _ = module.local_jacobian(carry, inputs, sensitivity)
        return next_carry, output
    return next_fun(*args, **kwargs)


def _is_rtu(module) -> bool:
    from memorax.networks.sequence_models.rtu import RTUCell

    return type(module) is RTUCell


@dataclass(frozen=True)
class ExactRTRL:
    """Pure adapter over a core's existing exact RTRL implementation."""

    core: Any

    def initialize(self, key, input_shape):
        return self.core.initialize_sensitivity(key, input_shape)

    def initialization(self):
        return nn.intercept_methods(_delegate_rtu_init_forward)

    def __call__(self, params, inputs, done, carry, credit):
        return self.core.apply(
            {"params": params},
            inputs,
            done,
            carry,
            sensitivity=credit,
            method="local_jacobian",
        )


@dataclass(frozen=True)
class TruncatedBPTT:
    """Credit that stops where the step does: the cell's own forward, once.

    There is no sensitivity to carry, so the kernel carries ``None``, and the
    gradient a kernel takes through this reaches only as far as the incoming
    carry it was handed -- which the kernel has already cut from the graph.
    """

    core: Any

    def initialize(self, key, input_shape):
        """No sensitivity to carry, which is what truncating the credit means."""

        del key, input_shape

    def initialization(self):
        return nullcontext()

    def __call__(self, params, inputs, done, carry, credit):
        del credit
        next_carry, hidden = self.core.apply({"params": params}, inputs, done, carry)
        return next_carry, hidden, None


def make_credit(kind: str, core):
    """Build the recurrent credit a kernel asked for by name.

    A sequence with nothing recurrent in it has no Jacobian to carry anything
    through, so both settings collapse to the one that carries nothing. That is
    not a special case being handled: the two differ only in what a carry
    contributes, and there is no carry.
    """

    if kind not in CREDITS:
        raise ValueError(f"unknown credit {kind!r}; use {', '.join(CREDITS)}")
    if core is None or kind == "tbptt":
        return TruncatedBPTT(core)
    return ExactRTRL(core)


def make_exact_rtrl_credit(core):
    """Build exact recurrent credit without duplicating core Jacobian math."""

    return ExactRTRL(core)
