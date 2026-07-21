from __future__ import annotations

from dataclasses import fields
from datetime import datetime
import hashlib
import json
import math
import re
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
import yaml

from .context import RunContext

ResourceProfileName = Literal["c7am", "c7al", "c7ax", "g6x"]
_RUN_CONTEXT_FIELDS = frozenset(field.name for field in fields(RunContext))
_IMAGE = re.compile(r"[^:@\s]+(?:/[^:@\s]+)+@sha256:[0-9a-f]{64}\Z")


def canonical_json(value: Any) -> str:
    return json.dumps(
        thaw_json(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_yaml(value: Any) -> str:
    return yaml.safe_dump(
        thaw_json(value),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=True,
    )


def _require_json(value: Any, path: str) -> None:
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path}: JSON float values must be finite")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _require_json(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path}: JSON object keys must be strings")
            _require_json(item, f"{path}.{key}")
        return
    raise TypeError(f"{path}: unsupported JSON value: {type(value).__name__}")


def _immutable(*_args: Any, **_kwargs: Any) -> None:
    raise TypeError("frozen JSON values cannot be modified")


class FrozenDict(dict[str, Any]):
    __delitem__ = _immutable
    __ior__ = _immutable
    __setitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


class FrozenList(list[Any]):
    __delitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable
    __setitem__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable


def freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return FrozenDict({key: freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return FrozenList(freeze_json(item) for item in value)
    return value


def thaw_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_json(item) for item in value]
    return value


def require_canonical_image_digest(value: str) -> str:
    if "://" in value or _IMAGE.fullmatch(value) is None:
        raise ValueError("image_digest must be a canonical immutable reference without a tag")
    repository = value.rsplit("@", 1)[0]
    if repository.rfind(":") > repository.rfind("/"):
        raise ValueError("image_digest must be a canonical immutable reference without a tag")
    return value


def _require_exact_zero(value: Any) -> Any:
    if type(value) is not int or value != 0:
        raise ValueError("attempt must be the integer zero")
    return value


class CanonicalRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_json(self) -> str:
        return canonical_json(self.model_dump(mode="json"))

    @classmethod
    def from_json(cls, value: str) -> Self:
        try:
            payload = json.loads(value)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(f"{cls.__name__} must be canonical JSON") from error
        if not isinstance(payload, dict):
            raise ValueError(f"{cls.__name__} must be a JSON object")
        record = cls.model_validate(payload)
        if value != record.to_json():
            raise ValueError(f"{cls.__name__} must use canonical JSON serialization")
        return record

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()


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
    _validate_image_digest = field_validator("image_digest")(require_canonical_image_digest)

    @field_validator("run_context", mode="before")
    @classmethod
    def require_strict_complete_run_context(cls, value: Any) -> Any:
        if type(value) is not dict:
            raise TypeError("run_context must be a plain JSON object")
        _require_json(value, "run_context")
        if set(value) != _RUN_CONTEXT_FIELDS:
            raise ValueError("run_context fields are incomplete or unknown")
        RunContext(**value)
        return value

    @field_validator("run_id", "artifact_prefix")
    @classmethod
    def require_nonempty(cls, value: str) -> str:
        if not value:
            raise ValueError("value must not be empty")
        return value

    @field_validator("argv")
    @classmethod
    def require_worker_config_placeholder(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not item for item in value):
            raise ValueError("argv must contain nonempty arguments")
        indexes = [index for index, item in enumerate(value) if item == "{config_path}"]
        if (
            len(indexes) != 1
            or indexes[0] == 0
            or value[indexes[0] - 1] not in {"--config", "--config_path"}
        ):
            raise ValueError("argv must contain one worker-generated {config_path}")
        return value

    @model_validator(mode="after")
    def verify_input_hashes(self) -> RunBundle:
        try:
            parsed_config = yaml.safe_load(self.config_yaml)
        except yaml.YAMLError as error:
            raise ValueError(f"config_yaml is not valid YAML: {error}") from error
        if type(parsed_config) is not dict or canonical_yaml(parsed_config) != self.config_yaml:
            raise ValueError("config_yaml must use canonical YAML serialization")
        if hashlib.sha256(self.config_yaml.encode()).hexdigest() != self.config_sha256:
            raise ValueError("config_sha256 does not match config_yaml")
        if hashlib.sha256(canonical_json(self.run_context).encode()).hexdigest() != (
            self.run_context_sha256
        ):
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

    _validate_image_digest = field_validator("image_digest")(require_canonical_image_digest)

    @model_validator(mode="after")
    def require_identity_and_runs(self) -> JobBundle:
        if not self.job_id:
            raise ValueError("job_id must not be empty")
        if not self.runs:
            raise ValueError("runs must not be empty")
        if len({run.run_id for run in self.runs}) != len(self.runs):
            raise ValueError("run IDs must be unique within a job")
        for run in self.runs:
            if run.image_digest != self.image_digest:
                raise ValueError(f"child run {run.run_id!r} image_digest does not match job")
            if run.resource_profile != self.resource_profile:
                raise ValueError(f"child run {run.run_id!r} resource profile does not match job")
        return self


class CompletionMarker(CanonicalRecord):
    run_id: str
    attempt: Literal[0] = 0
    exit_code: int
    started_at: datetime
    finished_at: datetime
    artifacts: tuple[str, ...] = ()
    error: str | None = None

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
        if self.error == "":
            raise ValueError("error must be nonempty when present")
        return self


class JobQuery(CanonicalRecord):
    job_id: str
    status: Literal[
        "SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING", "SUCCEEDED", "FAILED"
    ]
    status_reason: str | None = None

    @field_validator("job_id")
    @classmethod
    def require_job_id(cls, value: str) -> str:
        if not value:
            raise ValueError("job_id must not be empty")
        return value
