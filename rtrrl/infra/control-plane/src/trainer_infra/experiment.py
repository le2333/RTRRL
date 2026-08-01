from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, model_validator
from training_sdk.contract import SpaceEntry

Blank = Annotated[str, BeforeValidator(lambda value: "" if value is None else value)]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Compute(_Frozen):
    instance_type: str
    timeout_minutes: int

    @model_validator(mode="after")
    def _positive(self) -> "Compute":
        if self.timeout_minutes < 1:
            raise ValueError("timeout_minutes must be positive")
        return self


class Hpo(_Frozen):
    sampler: Literal["tpe", "random", "grid"]
    rounds: int
    trials_per_round: int
    parallel_jobs: int
    # Where the sampler starts. Two searches meant to be compared -- two entries
    # over one space, say -- have to be asked the same questions first, or part of
    # whatever separates them is which points they happened to try. Left unset the
    # sampler seeds itself, which is right for a search that answers to nobody.
    seed: int | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "Hpo":
        if min(self.rounds, self.trials_per_round, self.parallel_jobs) < 1:
            raise ValueError("rounds, trials_per_round and parallel_jobs must be positive")
        if self.parallel_jobs > self.trials_per_round:
            raise ValueError("parallel_jobs must not exceed trials_per_round")
        return self


class ScoreSpec(_Frozen):
    metric: str
    window_steps: tuple[int, int]
    reduce: Literal["mean", "median", "min", "max", "last"]
    direction: Literal["maximize", "minimize"]
    non_finite: Literal["worst"] | float

    @model_validator(mode="after")
    def _ordered(self) -> "ScoreSpec":
        if self.window_steps[0] > self.window_steps[1]:
            raise ValueError("window_steps must be ordered")
        return self


class LoggingSpec(_Frozen):
    aim: str
    every_steps: int
    enable_rerun: bool = False
    rerun_every_episodes: int | None = None


RESERVED = frozenset(
    {
        "environment",
        "env_mode",
        "env_backend",
        "observed",
        "seed",
        "num_envs",
        "total_steps",
        "epoch_steps",
        "eval_steps",
        "chunk_steps",
        "early_stop_patience",
        "eval_envs",
    }
)


class Environment(_Frozen):
    id: str
    backend: str
    seed: int
    observed: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def _usable(self) -> "Environment":
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


class Training(_Frozen):
    num_envs: int
    total_steps: int
    epoch_steps: int
    chunk_steps: int | None = None
    early_stop_patience: int | None = None

    @model_validator(mode="after")
    def _whole(self) -> "Training":
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


class Evaluation(_Frozen):
    steps: int
    num_envs: int

    @model_validator(mode="after")
    def _usable(self) -> "Evaluation":
        if self.steps < 0:
            raise ValueError("evaluation steps must not be negative")
        if self.num_envs < 1:
            raise ValueError("evaluation num_envs must be positive")
        return self


class Experiment(_Frozen):
    experiment: str
    name: str
    description: Blank = ""
    image: str
    entry: str
    storage: str
    environment: Environment
    training: Training
    evaluation: Evaluation
    compute: Compute
    hpo: Hpo
    space: dict[str, SpaceEntry]
    score: ScoreSpec
    logging: LoggingSpec

    @model_validator(mode="after")
    def _space_is_only_algorithm(self) -> "Experiment":
        taken = sorted(RESERVED & set(self.space))
        if taken:
            raise ValueError(
                f"space names {', '.join(taken)}, which belong to the environment "
                "and training or evaluation sections and are not searched"
            )
        return self


def load_experiment(path: Path) -> Experiment:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Experiment.model_validate(document)
