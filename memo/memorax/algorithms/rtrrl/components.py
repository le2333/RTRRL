"""Protocols shared by modular recurrent RTRRL components."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RecurrentComponent(Protocol):
    """A recurrent component whose changing values are explicit arguments."""

    def initialize(self, key: Any, input_shape: tuple[int, ...]) -> Any: ...

    def forward(
        self, params: Any, carry: Any, inputs: Any, reset: Any
    ) -> Any: ...

    def credit(
        self,
        params: Any,
        credit_state: Any,
        carry: Any,
        inputs: Any,
        cotangent: Any,
    ) -> Any: ...


__all__ = ["RecurrentComponent"]
