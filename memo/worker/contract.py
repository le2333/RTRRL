"""What a run configuration is, as the worker receives it.

The control plane writes these and this side validates what arrives, refusing
a version it does not implement. Neither side imports the other: they answer to
the shape written down in docs/contract.md and to ``CONTRACT_VERSION``, which is
what makes a mismatch a refusal rather than a misreading.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from memorax.parameters import ParameterNode, Scalar

CONTRACT_VERSION = 6


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


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
    # How long an episode may run before the clock ends it. The same policy's
    # return under a limit of 500 and of 1000 is not the same number, so this
    # is part of what the task is rather than a literal inside a wrapper.
    episode_length: int = 1000
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
    enable_rerun: bool = False
    rerun_s3: str | None = None
    rerun_every_steps: int | None = None


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
