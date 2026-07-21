from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal, Mapping, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    field_validator,
    model_validator,
)
from typing_extensions import TypeAliasType

JsonScalar: TypeAlias = str | int | float | bool | None
ResourceProfileName: TypeAlias = Literal["c7am", "c7al", "c7ax", "g6x"]
JsonValue = TypeAliasType(
    "JsonValue",
    JsonScalar | list["JsonValue"] | dict[str, "JsonValue"],
)


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
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_json(item) for item in value]
    return value


def _require_json_value(value: Any, path: str) -> None:
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path}: JSON float values must be finite")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _require_json_value(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path}: JSON object keys must be strings")
            _require_json_value(item, f"{path}.{key}")
        return
    raise TypeError(f"{path}: unsupported JSON value: {type(value).__name__}")


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ParameterPolicy(str, Enum):
    SCAN_UNFIXED = "scan_unfixed"
    EXPLICIT_SCAN = "explicit_scan"


class DiscreteDomain(ContractModel):
    values: list[JsonScalar]

    @model_validator(mode="after")
    def require_values(self) -> DiscreteDomain:
        if not self.values:
            raise ValueError("values must not be empty")
        return self


class ContinuousDomain(ContractModel):
    min: float
    max: float
    scale: Literal["linear", "log"] = "linear"

    @model_validator(mode="after")
    def require_finite_ordered_bounds(self) -> ContinuousDomain:
        if not math.isfinite(self.min) or not math.isfinite(self.max) or self.min >= self.max:
            raise ValueError("continuous bounds must be finite and min < max")
        if self.scale == "log" and self.min <= 0:
            raise ValueError("log domains require min > 0")
        return self


ParameterDomain: TypeAlias = DiscreteDomain | ContinuousDomain


class HpoSpec(ContractModel):
    total_trials: PositiveInt
    configs_per_batch: PositiveInt
    parameter_policy: ParameterPolicy = ParameterPolicy.SCAN_UNFIXED

    @model_validator(mode="after")
    def batch_fits_budget(self) -> HpoSpec:
        if self.configs_per_batch > self.total_trials:
            raise ValueError("configs_per_batch must not exceed total_trials")
        return self


class ExecutionSpec(ContractModel):
    runs_per_job: PositiveInt
    aim_result_timeout_seconds: PositiveInt = 600


class ExperimentIdentity(ContractModel):
    name: str
    description: str | None = None
    metadata: dict[str, Any] = {}


class EnvironmentSpec(ContractModel):
    name: str
    options: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("options", mode="before")
    @classmethod
    def require_finite_json(cls, value: Any) -> Any:
        _require_json_value(value, "options")
        json.dumps(value, allow_nan=False)
        return value

    @model_validator(mode="after")
    def require_name(self) -> EnvironmentSpec:
        if not self.name:
            raise ValueError("environment name must not be empty")
        return self

    def model_post_init(self, __context: Any) -> None:
        object.__setattr__(self, "options", freeze_json(dict(self.options)))


class TrainingBudgetSpec(ContractModel):
    env_steps: PositiveInt


class LoggingSpec(ContractModel):
    aim_every_env_steps: PositiveInt
    rerun_every_episodes: PositiveInt


class ResourcesSpec(ContractModel):
    profile: ResourceProfileName


class ExperimentDefaults(ContractModel):
    image: str
    environment: EnvironmentSpec | None = None
    training_budget: TrainingBudgetSpec | None = None
    logging: LoggingSpec | None = None
    resources: ResourcesSpec
    hpo: HpoSpec
    execution: ExecutionSpec
    parameters: dict[str, ParameterDomain] = {}


class GroupSpec(ContractModel):
    script: str
    image: str | None = None
    environment: EnvironmentSpec | None = None
    training_budget: TrainingBudgetSpec | None = None
    logging: LoggingSpec | None = None
    resources: ResourcesSpec | None = None
    hpo: HpoSpec | None = None
    execution: ExecutionSpec | None = None
    metadata: dict[str, Any] = {}
    parameters: dict[str, ParameterDomain] = {}
    overrides: dict[str, Any] = {}


class ExperimentSpec(ContractModel):
    experiment: ExperimentIdentity
    defaults: ExperimentDefaults
    groups: dict[str, GroupSpec]

    @model_validator(mode="after")
    def require_groups(self) -> ExperimentSpec:
        if not self.groups:
            raise ValueError("groups must not be empty")
        return self


class FieldConstraints(ContractModel):
    gt: float | None = None
    ge: float | None = None
    lt: float | None = None
    le: float | None = None


class FieldDescriptor(ContractModel):
    path: str
    type: Literal["str", "int", "float", "bool"]
    default: JsonScalar
    searchable: bool = False
    constraints: FieldConstraints = FieldConstraints()
    default_search: ParameterDomain | None = None
    choices: tuple[JsonScalar, ...] | None = None

    @model_validator(mode="after")
    def validate_choices(self) -> FieldDescriptor:
        if self.choices is None:
            return self
        if not self.choices:
            raise ValueError("choices must not be empty")
        for index, value in enumerate(self.choices):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("choices must contain only finite values")
            if any(value == previous for previous in self.choices[:index]):
                raise ValueError("duplicate choice values are not allowed")
        if not any(self.default == choice for choice in self.choices):
            raise ValueError("default must be one of choices")
        if isinstance(self.default_search, ContinuousDomain):
            raise ValueError("fields with choices require a discrete default search domain")
        if isinstance(self.default_search, DiscreteDomain):
            for value in self.default_search.values:
                if not any(value == choice for choice in self.choices):
                    raise ValueError("default_search values must be one of choices")
        return self


class DescriptorDefaults(ContractModel):
    environment: EnvironmentSpec
    training_budget: TrainingBudgetSpec
    logging: LoggingSpec


class ObjectiveSpec(ContractModel):
    metric: str
    direction: Literal["minimize", "maximize"]
    reduction: str


class ScriptDescriptor(ContractModel):
    name: str
    argv: tuple[str, ...]
    sdk_protocol_version: str
    defaults: DescriptorDefaults
    objective: ObjectiveSpec
    environments: tuple[str, ...]
    fields: dict[str, FieldDescriptor]

    @model_validator(mode="after")
    def require_unique_environments(self) -> ScriptDescriptor:
        if not self.environments:
            raise ValueError("environments must not be empty")
        for index, environment in enumerate(self.environments):
            if not environment:
                raise ValueError("environment names must not be empty")
            if environment in self.environments[:index]:
                raise ValueError(f"duplicate environment '{environment}' is not allowed")
        return self

    @model_validator(mode="after")
    def require_safe_unique_field_paths(self) -> ScriptDescriptor:
        paths: list[tuple[str, ...]] = []
        for name, descriptor in self.fields.items():
            segments = tuple(descriptor.path.split("."))
            if (
                not segments
                or any(
                    not segment.isidentifier() or segment.startswith("__")
                    for segment in segments
                )
            ):
                raise ValueError(f"field '{name}' has unsafe path {descriptor.path!r}")
            for previous in paths:
                if (
                    segments == previous
                    or segments[: len(previous)] == previous
                    or previous[: len(segments)] == segments
                ):
                    raise ValueError(
                        f"field '{name}' path {descriptor.path!r} conflicts "
                        "with another field path"
                    )
            paths.append(segments)
        return self


class ScriptCatalog(ContractModel):
    protocol_version: str
    scripts: dict[str, ScriptDescriptor]


class ResolvedConfiguration(ContractModel):
    image: str
    environment: EnvironmentSpec
    training_budget: TrainingBudgetSpec
    logging: LoggingSpec
    resources: ResourcesSpec
    hpo: HpoSpec
    execution: ExecutionSpec
    parameters: dict[str, ParameterDomain] = {}


@dataclass(frozen=True)
class DiscreteSearch:
    values: tuple[JsonScalar, ...]

    def __post_init__(self) -> None:
        for index, value in enumerate(self.values):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("categorical float values must be finite")
            if any(value == previous for previous in self.values[:index]):
                raise ValueError("duplicate categorical values are not allowed")


@dataclass(frozen=True)
class ContinuousSearch:
    low: float
    high: float
    log: bool
    integer: bool
    step: int | float | None

    def __post_init__(self) -> None:
        if self.integer and self.log and self.step not in (None, 1):
            raise ValueError("log integer domains require step 1")


SearchDomain: TypeAlias = DiscreteSearch | ContinuousSearch


@dataclass(frozen=True)
class ResolvedParameter:
    fixed_value: JsonScalar | None
    search_domain: SearchDomain | None


@dataclass(frozen=True)
class ResolvedGroup:
    name: str
    study_key: str
    image: str
    script: str
    argv: tuple[str, ...]
    sdk_protocol_version: str
    objective: ObjectiveSpec
    environment: EnvironmentSpec
    training_budget: TrainingBudgetSpec
    logging: LoggingSpec
    resources: ResourcesSpec
    hpo: HpoSpec
    execution: ExecutionSpec
    metadata: Mapping[str, Any]
    parameters: Mapping[str, ResolvedParameter]
    parameter_paths: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @property
    def fixed_parameters(self) -> Mapping[str, JsonScalar]:
        return MappingProxyType(
            {
                name: parameter.fixed_value
                for name, parameter in self.parameters.items()
                if parameter.search_domain is None
            }
        )

    def searchable_parameters(self) -> Mapping[str, SearchDomain]:
        return MappingProxyType(
            {
                name: parameter.search_domain
                for name, parameter in self.parameters.items()
                if parameter.search_domain is not None
            }
        )


@dataclass(frozen=True)
class ResolvedExperiment:
    name: str
    description: str | None
    metadata: Mapping[str, Any]
    groups: tuple[ResolvedGroup, ...]


@dataclass(frozen=True)
class ConcreteRun:
    study_key: str
    run_id: str
    run_name: str
    run_number: int
    trial_number: int
    image: str
    script: str
    argv: tuple[str, ...]
    sdk_protocol_version: str
    objective: ObjectiveSpec
    environment: EnvironmentSpec
    training_budget: TrainingBudgetSpec
    logging: LoggingSpec
    resources: ResourcesSpec
    hpo: HpoSpec
    execution: ExecutionSpec
    metadata: Mapping[str, Any]
    fixed_parameters: Mapping[str, Any]
    sampled_parameters: Mapping[str, Any]
    final_parameters: Mapping[str, Any]
    context: Mapping[str, Any]
    config_yaml: str
    config_sha256: str
    run_json: str
    run_sha256: str
