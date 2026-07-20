from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, TypeAlias, cast

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
FrozenJsonValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | tuple["FrozenJsonValue", ...]
    | Mapping[str, "FrozenJsonValue"]
)


def _freeze(value: JsonValue) -> FrozenJsonValue:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: FrozenJsonValue) -> JsonValue:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class RunContext:
    experiment_name: str
    experiment_id: str
    group: str
    script: str
    run_id: str
    run_number: int
    trial_number: int
    seed: int
    metadata: Mapping[str, JsonValue]
    environment: Mapping[str, JsonValue]
    training_budget: Mapping[str, JsonValue]
    fixed_parameters: Mapping[str, JsonValue]
    sampled_parameters: Mapping[str, JsonValue]
    final_parameters: Mapping[str, JsonValue]
    image_digest: str
    resource_profile: str
    artifact_directory: Path

    def __post_init__(self) -> None:
        for field_name in (
            "metadata",
            "environment",
            "training_budget",
            "fixed_parameters",
            "sampled_parameters",
            "final_parameters",
        ):
            value = cast(dict[str, JsonValue], dict(getattr(self, field_name)))
            object.__setattr__(self, field_name, _freeze(value))
        object.__setattr__(self, "artifact_directory", Path(self.artifact_directory))

    @classmethod
    def from_path(cls, path: Path) -> RunContext:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("run context must be a JSON object")
        return cls(**payload)

    @property
    def run_name(self) -> str:
        return f"{self.group}-{self.run_number:04d}"

    @property
    def hparams(self) -> dict[str, JsonValue]:
        return {
            "identity": {
                "group": self.group,
                "script": self.script,
                "run_number": self.run_number,
                "trial_number": self.trial_number,
                "seed": self.seed,
                "run_id": self.run_id,
            },
            "metadata": _thaw(cast(FrozenJsonValue, self.metadata)),
            "environment": _thaw(cast(FrozenJsonValue, self.environment)),
            "training_budget": _thaw(cast(FrozenJsonValue, self.training_budget)),
            "parameters": {
                "fixed": _thaw(cast(FrozenJsonValue, self.fixed_parameters)),
                "sampled": _thaw(cast(FrozenJsonValue, self.sampled_parameters)),
                "final": _thaw(cast(FrozenJsonValue, self.final_parameters)),
            },
            "infrastructure": {
                "image_digest": self.image_digest,
                "resource_profile": self.resource_profile,
            },
        }
