from __future__ import annotations

from dataclasses import fields
from datetime import datetime
import json
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from pydantic import field_validator, model_validator
from training_sdk import RunContext
import yaml

from trainer_infra.image_catalog import resolve_image
from trainer_infra.identities import canonical_json, canonical_yaml, sha256_text
from trainer_infra.models import (
    ConcreteRun,
    ContractModel,
    ResourceProfileName,
    _require_json_value,
    freeze_json,
    thaw_json,
)
_RUN_CONTEXT_FIELDS = frozenset(field.name for field in fields(RunContext))


def _require_exact_zero(value: Any) -> Any:
    if type(value) is not int or value != 0:
        raise ValueError("attempt must be the integer zero")
    return value


def _canonical_image_digest(value: str) -> str:
    resolved = resolve_image(value)
    if resolved.reference != value:
        raise ValueError(
            "image_digest must be a canonical immutable reference without a tag"
        )
    return resolved.reference


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
    image_digest: str
    resource_profile: ResourceProfileName
    config_yaml: str
    config_sha256: str
    run_context: dict[str, Any]
    run_context_sha256: str
    artifact_prefix: str

    _validate_attempt = field_validator("attempt", mode="before")(_require_exact_zero)
    _validate_image_digest = field_validator("image_digest")(_canonical_image_digest)

    @field_validator("run_context", mode="before")
    @classmethod
    def require_strict_complete_run_context(cls, value: Any) -> Any:
        if type(value) is not dict:
            raise TypeError("run_context must be a plain JSON object")
        _require_json_value(value, "run_context")
        actual_fields = set(value)
        if actual_fields != _RUN_CONTEXT_FIELDS:
            missing = sorted(_RUN_CONTEXT_FIELDS - actual_fields)
            extra = sorted(actual_fields - _RUN_CONTEXT_FIELDS)
            raise ValueError(
                f"run_context fields are incomplete or unknown; missing={missing}, extra={extra}"
            )
        try:
            RunContext(**value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"run_context is invalid: {error}") from error
        return value

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
        try:
            parsed_config = yaml.safe_load(self.config_yaml)
        except yaml.YAMLError as error:
            raise ValueError(f"config_yaml is not valid YAML: {error}") from error
        if type(parsed_config) is not dict:
            raise ValueError("config_yaml must contain a YAML object")
        if canonical_yaml(parsed_config) != self.config_yaml:
            raise ValueError("config_yaml must use canonical YAML serialization")
        if sha256_text(self.config_yaml) != self.config_sha256:
            raise ValueError("config_sha256 does not match config_yaml")
        if sha256_text(canonical_json(self.run_context)) != self.run_context_sha256:
            raise ValueError("run_context_sha256 does not match run_context")
        context = RunContext(**thaw_json(self.run_context))
        if context.run_id != self.run_id:
            raise ValueError("run_context run_id does not match run bundle")
        if context.image_digest != self.image_digest:
            raise ValueError("run_context image_digest does not match run bundle")
        if context.resource_profile != self.resource_profile:
            raise ValueError("run_context resource_profile does not match run bundle")
        return self

    def model_post_init(self, __context: Any) -> None:
        object.__setattr__(self, "run_context", freeze_json(dict(self.run_context)))


class JobBundle(CanonicalRecord):
    job_id: str
    image_digest: str
    resource_profile: ResourceProfileName
    runs: tuple[RunBundle, ...]

    _validate_image_digest = field_validator("image_digest")(_canonical_image_digest)

    @model_validator(mode="after")
    def require_identity_and_runs(self) -> JobBundle:
        if not self.job_id:
            raise ValueError("job_id must not be empty")
        if not self.runs:
            raise ValueError("runs must not be empty")
        run_ids = [run.run_id for run in self.runs]
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("run IDs must be unique within a job")
        for run in self.runs:
            if run.image_digest != self.image_digest:
                raise ValueError(
                    f"child run {run.run_id!r} image_digest does not match job"
                )
            if run.resource_profile != self.resource_profile:
                raise ValueError(
                    f"child run {run.run_id!r} resource profile does not match job"
                )
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
    try:
        experiment_name, derived_group = concrete_run.study_key.rsplit(":", 1)
    except ValueError as error:
        raise ValueError("study_key must contain experiment and group identity") from error
    if not experiment_name or not derived_group:
        raise ValueError("study_key must contain nonempty experiment and group identity")
    if group != derived_group:
        raise ValueError(
            f"group {group!r} does not match concrete run group {derived_group!r}"
        )
    expected_run_id = f"{concrete_run.study_key}:{concrete_run.run_number:04d}"
    if concrete_run.run_id != expected_run_id:
        raise ValueError(
            f"run_id {concrete_run.run_id!r} does not match {expected_run_id!r}"
        )
    context_group = concrete_run.context.get("group")
    context_run_id = concrete_run.context.get("run_id")
    if context_group != derived_group or context_run_id != concrete_run.run_id:
        raise ValueError("concrete run context identity does not match study_key/run_id")
    seed = concrete_run.final_parameters.get("seed")
    if type(seed) is not int:
        raise ValueError("concrete run requires an integer seed")
    image_digest = _canonical_image_digest(concrete_run.image)
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
        image_digest=image_digest,
        resource_profile=concrete_run.resources.profile,
        artifact_directory=Path(artifact_prefix),
        logging=concrete_run.logging.model_dump(mode="json"),
        objective=concrete_run.objective.model_dump(mode="json"),
    )
