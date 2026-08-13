"""Legacy score policy retained only until its Task 9 move into Infra."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ScoreConfig(_Frozen):
    metric: str
    window_steps: tuple[int, int]
    reduce: Literal["mean", "median", "min", "max", "last"]
    direction: Literal["maximize", "minimize"]
    non_finite: Literal["worst"] | float
    s3: str

    @model_validator(mode="after")
    def _ordered(self) -> "ScoreConfig":
        if self.window_steps[0] > self.window_steps[1]:
            raise ValueError("window_steps must be ordered")
        return self
