"""How a recurrent component carries parameter influence through time."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RecurrentDifferentiation(Protocol):
    """The boundary shared by generic and kernel-specific differentiation."""

    def initialize(self, key: Any, input_shape: tuple[int, ...]) -> Any: ...

    def initialization(self) -> Any: ...

    def __call__(
        self,
        params: Any,
        inputs: Any,
        done: Any,
        carry: Any,
        state: Any,
    ) -> tuple[Any, Any, Any]: ...


@dataclass(frozen=True)
class TruncatedBPTT:
    """Differentiate one recurrent step without carrying a derivative state."""

    core: Any

    def initialize(self, key, input_shape):
        del key, input_shape

    def initialization(self):
        return nullcontext()

    def __call__(self, params, inputs, done, carry, state):
        del state
        next_carry, output = self.core.apply(
            {"params": params}, inputs, done, carry
        )
        return next_carry, output, None
