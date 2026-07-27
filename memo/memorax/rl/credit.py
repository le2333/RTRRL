"""Exact online recurrent credit delegated to sequence-model primitives.

Nothing here computes anything, which is deliberate and is why no test compares
it against anything: the Jacobian of a carry with respect to its parameters
belongs to the cell that owns the recurrence, and every cell in this package
already implements it. This file only spells out which of a cell's methods
stands for online credit, so a kernel can ask for it without naming the cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExactRTRL:
    """Pure adapter over a core's existing exact RTRL implementation."""

    core: Any

    def initialize(self, key, input_shape):
        return self.core.initialize_sensitivity(key, input_shape)

    def __call__(self, params, inputs, done, carry, credit):
        return self.core.apply(
            {"params": params},
            inputs,
            done,
            carry,
            sensitivity=credit,
            method="local_jacobian",
        )


def make_exact_rtrl_credit(core):
    """Build exact recurrent credit without duplicating core Jacobian math."""

    return ExactRTRL(core)
