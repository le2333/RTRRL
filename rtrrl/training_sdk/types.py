from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class Episode:
    number: int
    phase: str
    start_env_steps: int
    end_env_steps: int
    observations: Sequence[Any]
    actions: Sequence[Any]
    rewards: Sequence[Any]
    terminals: Sequence[bool]
    truncations: Sequence[bool]
    environment_states: Sequence[Any] | None = None

    def __post_init__(self) -> None:
        if type(self.number) is not int:
            raise TypeError("episode number must be an integer")
        if not 1 <= self.number <= 999_999:
            raise ValueError("episode number must be between 1 and 999999")
        for field_name in (
            "observations",
            "actions",
            "rewards",
            "terminals",
            "truncations",
        ):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))
        if self.environment_states is not None:
            object.__setattr__(
                self, "environment_states", tuple(self.environment_states)
            )

        transition_count = len(self.actions)
        transition_lengths = {
            len(self.actions),
            len(self.rewards),
            len(self.terminals),
            len(self.truncations),
        }
        if len(transition_lengths) != 1:
            raise ValueError("episode transition arrays must have equal lengths")
        if len(self.observations) != transition_count + 1:
            raise ValueError(
                "episode observations must contain N+1 values for N transitions"
            )
        if transition_count == 0 or not (
            self.terminals[-1] or self.truncations[-1]
        ):
            raise ValueError("episode must be complete (terminal or truncated)")
        if self.end_env_steps < self.start_env_steps:
            raise ValueError("end_env_steps must not precede start_env_steps")
