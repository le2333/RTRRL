"""The complete run document validated at the Entry composition boundary."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from deployment.contract import CONTRACT_VERSION
from memorax.parameters import Scalar


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RunIdentity(_Frozen):
    run_id: str
    experiment: str
    launch_id: str
    trial: int
    digest: str


class Artifacts(_Frozen):
    root: str


class EnvironmentSpec(_Frozen):
    id: str
    backend: str
    observed: tuple[int, ...] | None = None
    episode_length: int = 1000

    @model_validator(mode="after")
    def _usable(self) -> "EnvironmentSpec":
        if self.episode_length < 1:
            raise ValueError("episode_length must be positive")
        if self.observed is not None and (
            not self.observed
            or len(set(self.observed)) != len(self.observed)
            or any(index < 0 for index in self.observed)
        ):
            raise ValueError("observed must contain unique non-negative indices")
        return self


class AlgorithmSpec(_Frozen):
    environment: EnvironmentSpec
    num_envs: int
    parameters: dict[str, Scalar]

    @model_validator(mode="after")
    def _usable(self) -> "AlgorithmSpec":
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        return self


class RuntimeSpec(_Frozen):
    seed: int
    total_steps: int
    epoch_steps: int
    evaluation_steps: int

    @model_validator(mode="after")
    def _whole_epochs(self) -> "RuntimeSpec":
        if self.seed < 0:
            raise ValueError("seed must not be negative")
        if self.total_steps < 1 or self.epoch_steps < 1:
            raise ValueError("training step budgets must be positive")
        if self.evaluation_steps < 0:
            raise ValueError("evaluation_steps must not be negative")
        if self.total_steps % self.epoch_steps:
            raise ValueError("total_steps must consist of whole epochs")
        return self


class AimSpec(_Frozen):
    url: str


class RerunSpec(_Frozen):
    every_steps: int

    @model_validator(mode="after")
    def _usable(self) -> "RerunSpec":
        if self.every_steps < 1:
            raise ValueError("rerun every_steps must be positive")
        return self


class LoggingSpec(_Frozen):
    aim: AimSpec
    rerun: RerunSpec | None = None


class RunSpec(_Frozen):
    contract: Literal[CONTRACT_VERSION]
    identity: RunIdentity
    entry: str
    artifacts: Artifacts
    algorithm: AlgorithmSpec
    runtime: RuntimeSpec
    logging: LoggingSpec

    @model_validator(mode="after")
    def _graph_width_matches_schedule(self) -> "RunSpec":
        if self.runtime.epoch_steps % self.algorithm.num_envs:
            raise ValueError("epoch_steps must contain whole environment steps")
        return self
