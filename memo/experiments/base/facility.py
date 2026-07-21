"""Strict host-side adapter for concrete trainer facility configurations."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from rtrrl_hopper.run import RTRRLHopperConfig
from stream_ac_kmemorychain.run import StreamACKMemoryChainConfig
from stream_ac_memorychain.run import StreamACMemoryChainConfig
from stream_ac_mujoco_masked.run import StreamACMujocoMaskedConfig


JsonValue = str | int | float | bool | None | tuple["JsonValue", ...] | Mapping[
    str, "JsonValue"
]


def _freeze_json(value: Any, path: str = "value") -> JsonValue:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain finite JSON numbers")
        return value
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} JSON object keys must be strings")
            frozen[key] = _freeze_json(item, f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{path}[]") for item in value)
    raise TypeError(f"{path} must contain only JSON-compatible values")


def _mapping(value: Any, name: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    return _freeze_json(value, name)


def _require_exact(mapping: Mapping[str, Any], expected: set[str], name: str) -> None:
    unknown = set(mapping) - expected
    missing = expected - set(mapping)
    if unknown:
        raise ValueError(f"unknown {name} fields: {sorted(unknown)!r}")
    if missing:
        raise ValueError(f"missing {name} fields: {sorted(missing)!r}")


@dataclass(frozen=True)
class FacilityInput:
    environment: Mapping[str, JsonValue]
    logging: Mapping[str, JsonValue]
    parameters: Mapping[str, JsonValue]
    training_budget: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        for name in ("environment", "logging", "parameters", "training_budget"):
            object.__setattr__(self, name, _mapping(getattr(self, name), name))
        _require_exact(self.environment, {"name", "options"}, "environment")
        if type(self.environment["name"]) is not str or not self.environment["name"]:
            raise ValueError("environment.name must be a non-empty string")
        if not isinstance(self.environment["options"], Mapping):
            raise TypeError("environment.options must be a JSON object")
        _require_exact(
            self.logging,
            {"aim_every_env_steps", "rerun_every_episodes"},
            "logging",
        )
        for field in ("aim_every_env_steps", "rerun_every_episodes"):
            if type(self.logging[field]) is not int or self.logging[field] <= 0:
                raise ValueError(f"logging.{field} must be a positive integer")
        _require_exact(self.training_budget, {"env_steps"}, "training_budget")
        if (
            type(self.training_budget["env_steps"]) is not int
            or self.training_budget["env_steps"] <= 0
        ):
            raise ValueError("training_budget.env_steps must be a positive integer")

    @classmethod
    def load(cls, path: str | Path) -> "FacilityInput":
        payload = yaml.safe_load(Path(path).read_text())
        if not isinstance(payload, Mapping):
            raise TypeError("facility config must be a YAML object")
        _require_exact(
            payload,
            {"environment", "logging", "parameters", "training_budget"},
            "facility config",
        )
        # JSON round-trip rejects YAML-native tags and non-finite numbers early.
        try:
            json.dumps(payload, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"facility config must be recursive finite JSON: {exc}") from exc
        return cls(**payload)


_STREAM_CONFIGS = {
    "memory_chain": StreamACMemoryChainConfig,
    "kmemory_chain": StreamACKMemoryChainConfig,
    "mujoco_masked": StreamACMujocoMaskedConfig,
}
_STREAM_OPTIONS = {
    "memory_chain": {"length": "chain_length", "max_episode_steps": "max_episode_steps"},
    "kmemory_chain": {
        "length": "chain_length",
        "num_bits": "num_bits",
        "max_episode_steps": "max_episode_steps",
    },
    "mujoco_masked": {
        "env_name": "env_name",
        "mode": "mode",
        "backend": "backend",
        "max_episode_steps": "max_episode_steps",
    },
}
_RTRRL_OPTIONS = {
    "mode": "mode",
    "backend": "backend",
    "max_episode_steps": "max_episode_steps",
    "normalize_obs": "normalize_obs",
    "normalize_reward": "normalize_reward",
}


def _build(config_type, value: FacilityInput, option_fields: Mapping[str, str]):
    options = value.environment["options"]
    assert isinstance(options, Mapping)
    unknown_options = set(options) - set(option_fields)
    if unknown_options:
        raise ValueError(f"unknown environment options: {sorted(unknown_options)!r}")
    allowed_parameters = {field.name for field in fields(config_type)}
    unknown_parameters = set(value.parameters) - allowed_parameters
    if unknown_parameters:
        raise ValueError(f"unknown parameters: {sorted(unknown_parameters)!r}")
    conflicting = set(value.parameters) & set(option_fields.values())
    if conflicting:
        raise ValueError(
            f"environment fields must be set through environment.options: {sorted(conflicting)!r}"
        )
    updates = dict(value.parameters)
    updates.update({option_fields[name]: item for name, item in options.items()})
    updates["total_timesteps"] = value.training_budget["env_steps"]
    config = config_type(**updates)
    if config.total_timesteps % config.num_envs:
        raise ValueError(
            "training_budget.env_steps must be divisible by parameters.num_envs "
            "to represent exact vector-environment interactions"
        )
    return config


def build_stream_ac_config(value: FacilityInput):
    name = value.environment["name"]
    if name not in _STREAM_CONFIGS:
        raise ValueError(
            f"memo_stream_ac does not support environment {name!r}; "
            f"use {sorted(_STREAM_CONFIGS)!r}"
        )
    if value.parameters.get("agent_type") != "rtu_rtrl":
        raise ValueError("memo_stream_ac requires agent_type='rtu_rtrl'")
    config = _build(_STREAM_CONFIGS[name], value, _STREAM_OPTIONS[name])
    if name in ("memory_chain", "kmemory_chain"):
        if config.max_episode_steps != config.chain_length:
            raise ValueError("max_episode_steps must equal length for memory-chain tasks")
    return config


def build_rtrrl_config(value: FacilityInput) -> RTRRLHopperConfig:
    if value.environment["name"] != "hopper":
        raise ValueError("memo_rtrrl supports only the 'hopper' environment")
    if value.parameters.get("rtrrl_topology") != "shared":
        raise ValueError("memo_rtrrl requires rtrrl_topology='shared'")
    return _build(RTRRLHopperConfig, value, _RTRRL_OPTIONS)
