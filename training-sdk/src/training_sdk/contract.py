from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTRACT_VERSION = 6

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


class FloatValidSpec(_Frozen):
    type: Literal["float"]
    low: float | None = None
    high: float | None = None

    @model_validator(mode="after")
    def _ordered(self) -> "FloatValidSpec":
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError("float valid low must not exceed high")
        return self


class IntValidSpec(_Frozen):
    type: Literal["int"]
    low: int | None = None
    high: int | None = None
    step: int = 1

    @model_validator(mode="after")
    def _ordered(self) -> "IntValidSpec":
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError("int valid low must not exceed high")
        if self.step < 1:
            raise ValueError("int valid step must be positive")
        return self


ValidSpec: TypeAlias = Annotated[
    FloatValidSpec | IntValidSpec | ChoiceSpec, Field(union_mode="left_to_right")
]


class ParameterSpec(_Frozen):
    kind: Literal["param"] = "param"
    value_type: Literal["float", "int", "str", "bool"]
    valid: ValidSpec
    search: SpaceEntry
    placeholder: Scalar


ParameterTree: TypeAlias = dict[str, "ParameterNode"]


class StructureSpec(_Frozen):
    kind: Literal["structure"] = "structure"
    placeholder: Scalar
    branches: dict[str, ParameterTree]

    @model_validator(mode="after")
    def _placeholder_is_a_branch(self) -> "StructureSpec":
        if self.placeholder not in self.branches:
            raise ValueError(
                f"structure placeholder {self.placeholder!r} is not one of its "
                f"branches: {', '.join(sorted(self.branches))}"
            )
        return self


ParameterNode: TypeAlias = Annotated[
    ParameterSpec | StructureSpec, Field(discriminator="kind")
]


class EntryDescriptor(_Frozen):
    command: tuple[str, ...]
    metrics: tuple[str, ...]
    parameters: dict[str, ParameterNode]

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
    seed: int
    observed: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def _usable(self) -> "EnvironmentConfig":
        if self.seed < 0:
            raise ValueError("seed must not be negative")
        if self.observed is None:
            return self
        if not self.observed:
            raise ValueError("observed must name at least one index")
        if len(set(self.observed)) != len(self.observed):
            raise ValueError("observed must not repeat an index")
        if any(index < 0 for index in self.observed):
            raise ValueError("observed indices must not be negative")
        return self


class TrainingConfig(_Frozen):
    num_envs: int
    total_steps: int
    epoch_steps: int
    chunk_steps: int | None = None
    early_stop_patience: int | None = None

    @model_validator(mode="after")
    def _whole(self) -> "TrainingConfig":
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        if self.total_steps < 1:
            raise ValueError("total_steps must be positive")
        if self.epoch_steps < 1:
            raise ValueError("epoch_steps must be positive")
        if self.total_steps % self.epoch_steps:
            raise ValueError(
                f"total_steps {self.total_steps} is not whole epochs of "
                f"{self.epoch_steps}"
            )
        if self.epoch_steps % self.num_envs:
            raise ValueError(
                f"epoch_steps {self.epoch_steps} is not "
                f"{self.num_envs} streams' worth"
            )
        if self.chunk_steps is not None:
            if self.chunk_steps < 1:
                raise ValueError("chunk_steps must be positive")
            per_chunk = self.chunk_steps * self.num_envs
            if self.total_steps % per_chunk or self.epoch_steps % per_chunk:
                raise ValueError(
                    f"chunk_steps {self.chunk_steps} over {self.num_envs} streams "
                    "must divide total_steps and epoch_steps"
                )
        if self.early_stop_patience is not None and self.early_stop_patience < 0:
            raise ValueError("early_stop_patience must not be negative")
        return self


class EvaluationConfig(_Frozen):
    steps: int
    num_envs: int

    @model_validator(mode="after")
    def _usable(self) -> "EvaluationConfig":
        if self.steps < 0:
            raise ValueError("evaluation steps must not be negative")
        if self.num_envs < 1:
            raise ValueError("evaluation num_envs must be positive")
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
    training: TrainingConfig
    evaluation: EvaluationConfig
    params: dict[str, Scalar]
    logging: LoggingConfig
    score: ScoreConfig
