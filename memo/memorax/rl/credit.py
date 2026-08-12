"""Two recurrent credit adapters behind one interface.

Neither computes anything here. ``ExactRTRL`` calls the cell's own
``local_jacobian`` and carries the sensitivity it returns; ``TruncatedBPTT``
calls the cell's forward and carries nothing. This file only says which of a
cell's methods stands for credit, so a kernel can ask without naming the cell.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax

from memorax.building import ComponentFamily

CREDITS = ("rtrl", "tbptt")
CREDIT_BRANCHES = {name: () for name in CREDITS}


def _construct_credit(selection, builder):
    del builder
    return selection.kind


CREDIT_FAMILY = ComponentFamily(
    branches=CREDIT_BRANCHES,
    construct=_construct_credit,
)


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
