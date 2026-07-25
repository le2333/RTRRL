from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTRACT_VERSION = 2

Scalar: TypeAlias = int | float | str | bool


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FloatSpec(_Frozen):
    type: Literal["float"]
    low: float
    high: float
    log: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> "FloatSpec":
        if self.low > self.high:
            raise ValueError("float low must not exceed high")
        return self


class IntSpec(_Frozen):
    type: Literal["int"]
    low: int
    high: int
    step: int = 1
    log: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> "IntSpec":
        if self.low > self.high:
            raise ValueError("int low must not exceed high")
        if self.step < 1:
            raise ValueError("int step must be positive")
        return self


class ChoiceSpec(_Frozen):
    choices: tuple[Scalar, ...]

    @model_validator(mode="before")
    @classmethod
    def _from_list(cls, value: object) -> object:
        if isinstance(value, list):
            return {"choices": value}
        return value

    @model_validator(mode="after")
    def _non_empty(self) -> "ChoiceSpec":
        if not self.choices:
            raise ValueError("choice list must not be empty")
        return self


SpaceEntry: TypeAlias = Annotated[
    FloatSpec | IntSpec | ChoiceSpec, Field(union_mode="left_to_right")
]


class EntryDescriptor(_Frozen):
    command: tuple[str, ...]
    source_hash: str
    metrics: tuple[str, ...]
    space: dict[str, SpaceEntry]

    @model_validator(mode="after")
    def _non_empty(self) -> "EntryDescriptor":
        if not self.command:
            raise ValueError("command must not be empty")
        if not self.metrics:
            raise ValueError("metrics must not be empty")
        return self


class Catalog(_Frozen):
    contract: int
    entries: dict[str, EntryDescriptor]


class LoggingConfig(_Frozen):
    aim: str
    every_steps: int
    rerun_s3: str | None = None
    rerun_every_episodes: int | None = None


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


class RunConfig(_Frozen):
    contract: int
    run_id: str
    experiment: str
    name: str
    launch_id: str
    trial: int
    entry: str
    params: dict[str, Scalar]
    logging: LoggingConfig
    score: ScoreConfig
