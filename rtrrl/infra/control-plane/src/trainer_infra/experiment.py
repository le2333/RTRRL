from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator
from training_sdk.contract import SpaceEntry


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
    rerun_every_episodes: int | None = None


class Experiment(_Frozen):
    experiment: str
    name: str
    description: str
    image: str
    entry: str
    storage: str
    compute: Compute
    hpo: Hpo
    space: dict[str, SpaceEntry]
    score: ScoreSpec
    logging: LoggingSpec


def load_experiment(path: Path) -> Experiment:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Experiment.model_validate(document)
