from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from pydantic import field_validator, model_validator
from training_sdk import RunContext

from trainer_infra.identities import canonical_json, sha256_text
from trainer_infra.models import (
    ConcreteRun,
    ContractModel,
    ResourceProfileName,
    freeze_json,
    thaw_json,
)


def _require_exact_zero(value: Any) -> Any:
    if type(value) is not int or value != 0:
        raise ValueError("attempt must be the integer zero")
    return value


class CanonicalRecord(ContractModel):
    _HASH_FIELD: ClassVar[str] = "sha256"

    def to_json(self) -> str:
        return canonical_json(self.model_dump(mode="json"))

    @classmethod
    def from_json(cls, value: str) -> Self:
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError(f"{cls.__name__} must be a JSON object")
        return cls.model_validate(payload)

    @property
    def sha256(self) -> str:
        return sha256_text(self.to_json())


class RunBundle(CanonicalRecord):
    run_id: str
    attempt: Literal[0] = 0
    argv: tuple[str, ...]
    config_yaml: str
    config_sha256: str
    run_context: dict[str, Any]
    run_context_sha256: str
    artifact_prefix: str

    _validate_attempt = field_validator("attempt", mode="before")(_require_exact_zero)

    @field_validator("run_id", "artifact_prefix")
    @classmethod
    def require_nonempty(cls, value: str) -> str:
        if not value:
            raise ValueError("value must not be empty")
        return value

    @field_validator("argv")
    @classmethod
    def require_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not item for item in value):
            raise ValueError("argv must contain nonempty arguments")
        return value

    @model_validator(mode="after")
    def verify_input_hashes(self) -> RunBundle:
        if sha256_text(self.config_yaml) != self.config_sha256:
            raise ValueError("config_sha256 does not match config_yaml")
        if sha256_text(canonical_json(self.run_context)) != self.run_context_sha256:
            raise ValueError("run_context_sha256 does not match run_context")
        return self

    def model_post_init(self, __context: Any) -> None:
        object.__setattr__(self, "run_context", freeze_json(dict(self.run_context)))


class JobBundle(CanonicalRecord):
    job_id: str
    image_digest: str
    resource_profile: ResourceProfileName
    runs: tuple[RunBundle, ...]

    @model_validator(mode="after")
    def require_identity_and_runs(self) -> JobBundle:
        if not self.job_id or not self.image_digest:
            raise ValueError("job_id and image_digest must not be empty")
        if not self.runs:
            raise ValueError("runs must not be empty")
        run_ids = [run.run_id for run in self.runs]
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("run IDs must be unique within a job")
        return self


class CompletionMarker(CanonicalRecord):
    run_id: str
    attempt: Literal[0] = 0
    exit_code: int
    started_at: datetime
    finished_at: datetime
    artifacts: tuple[str, ...] = ()

    _validate_attempt = field_validator("attempt", mode="before")(_require_exact_zero)

    @model_validator(mode="after")
    def require_valid_interval_and_artifacts(self) -> CompletionMarker:
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if self.started_at.tzinfo is None or self.finished_at.tzinfo is None:
            raise ValueError("completion timestamps must be timezone-aware")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        if any(not artifact for artifact in self.artifacts):
            raise ValueError("artifact paths must not be empty")
        return self


class JobQuery(CanonicalRecord):
    job_id: str
    status: Literal[
        "SUBMITTED",
        "PENDING",
        "RUNNABLE",
        "STARTING",
        "RUNNING",
        "SUCCEEDED",
        "FAILED",
    ]
    status_reason: str | None = None

    @field_validator("job_id")
    @classmethod
    def require_job_id(cls, value: str) -> str:
        if not value:
            raise ValueError("job_id must not be empty")
        return value


def build_run_context(
    experiment_id: str,
    group: str,
    concrete_run: ConcreteRun,
    artifact_prefix: str | Path,
) -> RunContext:
    seed = concrete_run.final_parameters.get("seed")
    if type(seed) is not int:
        raise ValueError("concrete run requires an integer seed")
    experiment_name = concrete_run.study_key.rsplit(":", 1)[0]
    return RunContext(
        experiment_name=experiment_name,
        experiment_id=experiment_id,
        group=group,
        script=concrete_run.script,
        run_id=concrete_run.run_id,
        run_number=concrete_run.run_number,
        trial_number=concrete_run.trial_number,
        seed=seed,
        metadata=thaw_json(concrete_run.metadata),
        environment=concrete_run.environment.model_dump(mode="json"),
        training_budget=concrete_run.training_budget.model_dump(mode="json"),
        fixed_parameters=thaw_json(concrete_run.fixed_parameters),
        sampled_parameters=thaw_json(concrete_run.sampled_parameters),
        final_parameters=thaw_json(concrete_run.final_parameters),
        image_digest=concrete_run.image,
        resource_profile=concrete_run.resources.profile,
        artifact_directory=Path(artifact_prefix),
        logging=concrete_run.logging.model_dump(mode="json"),
        objective=concrete_run.objective.model_dump(mode="json"),
    )
