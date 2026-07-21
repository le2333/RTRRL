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
    protocol_version: str
    environment: Mapping[str, JsonValue]
    logging: Mapping[str, JsonValue]
    parameters: Mapping[str, JsonValue]
    training_budget: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        if self.protocol_version != "1":
            raise ValueError("protocol_version must be '1'")
        for name in ("environment", "logging", "parameters", "training_budget"):
            object.__setattr__(self, name, _mapping(getattr(self, name), name))
        _require_exact(self.environment, {"name", "options"}, "environment")
        if type(self.environment["name"]) is not str or not self.environment["name"]:
            raise ValueError("environment.name must be a non-empty string")
        if not isinstance(self.environment["options"], Mapping):
            raise TypeError("environment.options must be a JSON object")
        if self.environment["name"] == "mujoco_masked":
            options = self.environment["options"]
            assert isinstance(options, Mapping)
            supported = {
                "env_name": {"ant", "halfcheetah", "hopper", "walker2d"},
                "mode": {"F", "P", "V"},
                "backend": {"generalized", "spring", "positional", "mjx"},
            }
            for field, choices in supported.items():
                if options.get(field) not in choices:
                    raise ValueError(
                        f"environment.options.{field} must be one of "
                        f"{sorted(choices)!r}"
                    )
        if self.environment["name"] == "hopper":
            options = self.environment["options"]
            assert isinstance(options, Mapping)
            supported = {
                "env_name": {"hopper"},
                "mode": {"F", "P", "V"},
                "backend": {"generalized", "spring", "positional", "mjx"},
            }
            for field, choices in supported.items():
                value = options.get(field, "hopper" if field == "env_name" else None)
                if value not in choices:
                    raise ValueError(
                        f"environment.options.{field} must be one of "
                        f"{sorted(choices)!r}"
                    )
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
            {
                "protocol_version",
                "environment",
                "logging",
                "parameters",
                "training_budget",
            },
            "facility config",
        )
        # JSON round-trip rejects YAML-native tags and non-finite numbers early.
        try:
            json.dumps(payload, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"facility config must be recursive finite JSON: {exc}") from exc
        return cls(**payload)


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


def _stream_configs():
    from stream_ac_kmemorychain.run import StreamACKMemoryChainConfig
    from stream_ac_memorychain.run import StreamACMemoryChainConfig
    from stream_ac_mujoco_masked.run import StreamACMujocoMaskedConfig

    return {
        "memory_chain": StreamACMemoryChainConfig,
        "kmemory_chain": StreamACKMemoryChainConfig,
        "mujoco_masked": StreamACMujocoMaskedConfig,
    }
_RTRRL_OPTIONS = {
    "env_name": "env_name",
    "mode": "mode",
    "backend": "backend",
    "max_episode_steps": "max_episode_steps",
    "normalize_obs": "normalize_obs",
    "normalize_reward": "normalize_reward",
}


_NETWORK_FIELDS = {
    "hidden_dim",
    "encoder_dim",
    "meta_rl",
    "use_encoder",
    "lru_output_dim",
    "backbone",
}
_RUNTIME_FIELDS = {
    "seed",
    "num_envs",
    "num_epochs",
    "eval_every",
    "eval_steps",
    "log_every",
}


def _parameter_updates(config_type, parameters: Mapping[str, JsonValue]) -> dict:
    """Resolve FieldDescriptor paths relative to the concrete parameters root."""

    allowed = {field.name for field in fields(config_type)}
    updates = {}
    for key, value in parameters.items():
        if isinstance(value, Mapping):
            if key not in {"algorithm", "network", "runtime"}:
                raise ValueError(f"unknown parameter namespace: {key!r}")
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, Mapping):
                    raise ValueError(
                        f"unknown nested parameter path: {key}.{nested_key}"
                    )
                if nested_key not in allowed:
                    raise ValueError(
                        f"unknown nested parameter path: {key}.{nested_key}"
                    )
                if key == "network" and nested_key not in _NETWORK_FIELDS:
                    raise ValueError(
                        f"parameter path network.{nested_key} is not a network field"
                    )
                if key == "runtime" and nested_key not in _RUNTIME_FIELDS:
                    raise ValueError(
                        f"parameter path runtime.{nested_key} is not a runtime field"
                    )
                if key == "algorithm" and nested_key in (
                    _NETWORK_FIELDS | _RUNTIME_FIELDS
                ):
                    raise ValueError(
                        f"parameter path algorithm.{nested_key} is not an algorithm field"
                    )
                if nested_key in updates:
                    raise ValueError(f"parameter path conflict for {nested_key!r}")
                updates[nested_key] = nested_value
            continue
        if key not in allowed:
            raise ValueError(f"unknown parameters: {[key]!r}")
        if key in updates:
            raise ValueError(f"parameter path conflict for {key!r}")
        updates[key] = value
    return updates


def _build(config_type, value: FacilityInput, option_fields: Mapping[str, str]):
    options = value.environment["options"]
    assert isinstance(options, Mapping)
    unknown_options = set(options) - set(option_fields)
    if unknown_options:
        raise ValueError(f"unknown environment options: {sorted(unknown_options)!r}")
    parameter_updates = _parameter_updates(config_type, value.parameters)
    conflicting = set(parameter_updates) & set(option_fields.values())
    if conflicting:
        raise ValueError(
            f"environment fields must be set through environment.options: {sorted(conflicting)!r}"
        )
    updates = parameter_updates
    updates.update({option_fields[name]: item for name, item in options.items()})
    updates["total_timesteps"] = value.training_budget["env_steps"]
    if updates.get("patience", 0) not in (0, None):
        raise ValueError("facility training does not allow early stopping")
    updates["patience"] = 0
    updates["require_full_budget"] = True
    config = config_type(**updates)
    epoch_quantum = config.num_envs * config.num_epochs
    if config.total_timesteps % epoch_quantum:
        raise ValueError(
            "training_budget.env_steps must be divisible by "
            "parameters.num_envs * parameters.num_epochs to represent exact "
            "fixed-length vector-environment epochs"
        )
    return config


def build_stream_ac_config(value: FacilityInput):
    name = value.environment["name"]
    configs = _stream_configs()
    if name not in configs:
        raise ValueError(
            f"memo_stream_ac does not support environment {name!r}; "
            f"use {sorted(configs)!r}"
        )
    config = _build(configs[name], value, _STREAM_OPTIONS[name])
    if config.agent_type != "rtu_rtrl":
        raise ValueError("memo_stream_ac requires agent_type='rtu_rtrl'")
    if name in ("memory_chain", "kmemory_chain"):
        if config.max_episode_steps != config.chain_length:
            raise ValueError("max_episode_steps must equal length for memory-chain tasks")
    return config


def build_rtrrl_config(value: FacilityInput):
    from rtrrl_hopper.run import RTRRLHopperConfig

    if value.environment["name"] != "hopper":
        raise ValueError("memo_rtrrl supports only the 'hopper' environment")
    config = _build(RTRRLHopperConfig, value, _RTRRL_OPTIONS)
    if config.rtrrl_topology != "shared":
        raise ValueError("memo_rtrrl requires rtrrl_topology='shared'")
    return config
