from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Episode:
    number: int
    phase: str
    start_env_steps: int
    end_env_steps: int
    rewards: Sequence[Any]
    terminals: Sequence[bool]
    truncations: Sequence[bool]
    # Which stream ran it. An episode belongs to one, and a sample step -- the
    # step counter numbers every stream's every step -- names one.
    stream: int = 0
    series: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    observations: Sequence[Any] | None = None
    actions: Sequence[Any] | None = None
    environment_states: Sequence[Any] | None = None

    def __post_init__(self) -> None:
        if type(self.number) is not int:
            raise TypeError("episode number must be an integer")
        if not 1 <= self.number <= 999_999:
            raise ValueError("episode number must be between 1 and 999999")
        for field_name in ("rewards", "terminals", "truncations"):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))
        for field_name in ("observations", "actions", "environment_states"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, tuple(value))
        object.__setattr__(
            self,
            "series",
            {name: tuple(values) for name, values in self.series.items()},
        )

        lengths = {len(self.rewards), len(self.terminals), len(self.truncations)}
        if len(lengths) != 1:
            raise ValueError("episode transition arrays must have equal lengths")
        transition_count = len(self.rewards)
        if transition_count == 0 or not (self.terminals[-1] or self.truncations[-1]):
            raise ValueError("episode must be complete (terminal or truncated)")
        for name, values in self.series.items():
            if len(values) != transition_count:
                raise ValueError(f"series {name} must span the episode")
        if self.actions is not None and len(self.actions) != transition_count:
            raise ValueError("actions must hold one value per transition")
        if self.observations is not None and (
            len(self.observations) != transition_count + 1
        ):
            raise ValueError(
                "episode observations must contain N+1 values for N transitions"
            )
        if self.end_env_steps < self.start_env_steps:
            raise ValueError("end_env_steps must not precede start_env_steps")
