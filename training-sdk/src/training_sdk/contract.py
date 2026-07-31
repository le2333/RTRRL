from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTRACT_VERSION = 4

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


class EnvironmentConfig(_Frozen):
    id: str
    backend: str
    num_envs: int
    observed: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def _usable(self) -> "EnvironmentConfig":
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        if self.observed is None:
            return self
        if not self.observed:
            raise ValueError("observed must name at least one index")
        if len(set(self.observed)) != len(self.observed):
            raise ValueError("observed must not repeat an index")
        if any(index < 0 for index in self.observed):
            raise ValueError("observed indices must not be negative")
        return self


class BudgetConfig(_Frozen):
    total_steps: int
    epoch_steps: int
    eval_steps: int

    @model_validator(mode="after")
    def _whole(self) -> "BudgetConfig":
        if self.total_steps < 1:
            raise ValueError("total_steps must be positive")
        if self.epoch_steps < 1:
            raise ValueError("epoch_steps must be positive")
        if self.eval_steps < 0:
            raise ValueError("eval_steps must not be negative")
        if self.total_steps % self.epoch_steps:
            raise ValueError(
                f"total_steps {self.total_steps} is not whole epochs of "
                f"{self.epoch_steps}"
            )
        return self


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
    digest: str
    environment: EnvironmentConfig
    budget: BudgetConfig
    params: dict[str, Scalar]
    logging: LoggingConfig
    score: ScoreConfig

    @model_validator(mode="after")
    def _epochs_hold_whole_rounds_of_streams(self) -> "RunConfig":
        if self.budget.epoch_steps % self.environment.num_envs:
            raise ValueError(
                f"epoch_steps {self.budget.epoch_steps} is not "
                f"{self.environment.num_envs} streams' worth"
            )
        return self
