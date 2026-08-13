"""Legacy execution and score models pending Worker transport migration.

The versioned catalog is owned by :mod:`deployment.contract`; the version-8
Worker projection is :mod:`worker.envelope`. Task 8 removes the remaining flat
execution model together with Worker-owned scoring.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from memorax.parameters import Scalar


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


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
    # ``num_envs`` is the one field here the graph reads rather than the
    # schedule: every carry, trace and sensitivity is opened at this width, so
    # it is fixed when the agent is built and cannot be varied afterwards.
    num_envs: int
    total_steps: int
    epoch_steps: int

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
                f"epoch_steps {self.epoch_steps} is not {self.num_envs} streams' worth"
            )
        return self


class EvaluationConfig(_Frozen):
    """How much evaluation to run. Not on how many streams.

    Evaluation reuses the streams the agent was built with, because the widths
    above are baked into the graph. Naming a second count here would describe
    something only a second graph could do.
    """

    steps: int

    @model_validator(mode="after")
    def _usable(self) -> "EvaluationConfig":
        if self.steps < 0:
            raise ValueError("evaluation steps must not be negative")
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
