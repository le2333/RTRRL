"""Exact online recurrent credit delegated to sequence-model primitives."""

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
