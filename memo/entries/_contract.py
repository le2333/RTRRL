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
    """What a run says about the environment its graph is built against.

    ``backend`` is null wherever the namespace has only one implementation to
    choose between. Brax names a physics backend; Gymnax has none, and saying
    so is not the same as omitting a field that means something.
    """

    id: str
    backend: str | None = None
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


class TrainingSpec(_Frozen):
    """The budget and the shape of the run, and nothing about reporting.

    ``chunk_steps`` is how much one training call may hold, which is the
    memory a run costs. It is deliberately unrelated to the environment's
    episode limit: how long an episode runs is decided while the run is going
    and changes as the policy improves, so sizing a buffer from it would tie a
    memory budget to a number that has nothing to do with memory.
    """

    seed: int
    total_steps: int
    chunk_steps: int

    @model_validator(mode="after")
    def _usable(self) -> "TrainingSpec":
        if self.seed < 0:
            raise ValueError("seed must not be negative")
        if self.total_steps < 1 or self.chunk_steps < 1:
            raise ValueError("training step budgets must be positive")
        return self


class EvaluationSpec(_Frozen):
    """How often the policy is measured, and for how long each time."""

    every_steps: int
    rollout_steps: int

    @model_validator(mode="after")
    def _usable(self) -> "EvaluationSpec":
        if self.every_steps < 1:
            raise ValueError("evaluation every_steps must be positive")
        if self.rollout_steps < 0:
            raise ValueError("rollout_steps must not be negative")
        return self


class AimTrainingSpec(_Frozen):
    """Present when training metrics are wanted on the dashboard at all."""

    log_every_steps: int

    @model_validator(mode="after")
    def _usable(self) -> "AimTrainingSpec":
        if self.log_every_steps < 1:
            raise ValueError("aim training log_every_steps must be positive")
        return self


class AimSpec(_Frozen):
    """Evaluation always reaches Aim; training reaches it only if asked.

    A run ends one episode every few dozen steps before its policy is any
    good, so recording every one of them is tens of millions of points nobody
    can read. The complete record is the metrics artifact, which is not
    optional and not configurable.
    """

    url: str
    training: AimTrainingSpec | None = None


class RerunSpec(_Frozen):
    log_every_steps: int

    @model_validator(mode="after")
    def _usable(self) -> "RerunSpec":
        if self.log_every_steps < 1:
            raise ValueError("rerun log_every_steps must be positive")
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
    training: TrainingSpec
    evaluation: EvaluationSpec
    logging: LoggingSpec

    @model_validator(mode="after")
    def _graph_width_matches_schedule(self) -> "RunSpec":
        streams = self.algorithm.num_envs
        if self.training.chunk_steps % streams:
            raise ValueError("chunk_steps must contain whole environment steps")
        if self.evaluation.every_steps % streams:
            raise ValueError(
                "evaluation every_steps must contain whole environment steps"
            )
        if self.training.total_steps % self.evaluation.every_steps:
            raise ValueError("total_steps must consist of whole evaluation intervals")
        return self
