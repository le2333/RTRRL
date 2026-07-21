"""Lightweight migration helpers for the historical RTRRL command line."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict
import json
import math
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
from typing import Any, Sequence

import yaml

from .compatibility import (
    InvalidRTRRLConfig,
    LegacyRTRRLConfig,
    UnknownRTRRLField,
    UnsupportedRTRRLBranch,
    normalize_legacy_config,
    to_component_config,
)


EXPECTED_PLAN_CONFIG_COUNT = 686


def historical_rtrrl_metrics(
    summary,
    *,
    log_td_lr: bool,
    log_rnn_lr: bool,
    log_norms: bool,
) -> dict[str, Any]:
    """Translate an epoch summary to the frozen historical logger schema."""

    metrics = {
        "steps": summary.steps,
        "mean_reward": summary.mean_reward,
        "num_episodes": summary.num_episodes,
        "mean_delta": summary.mean_delta,
        "mean_r_bar": summary.mean_r_bar,
        "mean_v": summary.mean_v,
        "total_td_loss": summary.total_td_loss,
        "actor_loss": summary.actor_loss,
        "critic_loss": summary.critic_loss,
        "entropy": summary.entropy,
        "v_targ": summary.v_targ,
    }
    if summary.magnitude_loss is not None:
        metrics["magnitude_loss"] = summary.magnitude_loss
    if log_td_lr:
        metrics["lr/td"] = summary.learning_rate_td
    if log_rnn_lr:
        metrics["lr/rnn"] = summary.learning_rate_rnn
    if log_norms:
        metrics.update(
            {f"norms/{key}": value for key, value in summary.norms.items()}
        )
    return metrics


def run_mock_epoch() -> dict[str, Any]:
    """Build the approved Task-9 synthetic epoch and translate its summary."""

    def float32(value: float) -> float:
        return struct.unpack("!f", struct.pack("!f", value))[0]

    def mean_float32(values: Sequence[float]) -> float:
        return float32(sum(float32(value) for value in values) / len(values))

    rewards = (1.0, 2.0, 3.0)
    dones = (0.0, 0.5, 1.0)
    td_errors = (0.5, -0.25, 1.0)
    values = (10.0, 20.0, 30.0)
    value_targets = (11.0, 19.0, 32.0)
    entropies = (0.2, 0.4, 0.6)
    actor_losses = (-0.3, -0.6, -0.9)
    num_envs = 2
    num_episodes = round(sum(dones) * num_envs)
    divisor = max(num_episodes, 1)
    summary = SimpleNamespace(
        steps=15 * num_envs,
        mean_reward=float32(sum(rewards) * num_envs / divisor),
        num_episodes=num_episodes,
        mean_delta=float32(sum(td_errors) * num_envs / divisor),
        mean_r_bar=float32(float32(0.3) / divisor),
        mean_v=mean_float32(values),
        actor_loss=mean_float32(actor_losses),
        critic_loss=mean_float32(values),
        entropy=mean_float32(entropies),
        v_targ=mean_float32(value_targets),
        magnitude_loss=None,
        learning_rate_td=float32(1e-3),
        learning_rate_rnn=float32(2e-4),
        norms={
            "['z']['trace']": float32(math.sqrt(3.0**2 + 4.0**2)),
            "['params']['weight']": float32(
                math.sqrt(6.0**2 + 8.0**2)
            ),
            "['slow_params']['weight']": float32(
                math.sqrt(5.0**2 + 12.0**2)
            ),
        },
    )
    summary.total_td_loss = float32(
        summary.actor_loss + summary.critic_loss
    )
    return historical_rtrrl_metrics(
        summary,
        log_td_lr=True,
        log_rnn_lr=True,
        log_norms=True,
    )


def _set_nested(mapping: dict[str, Any], dotted_path: str, value: Any) -> None:
    target = mapping
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        child = target.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"cannot override nested field beneath {part!r}")
        target = child
    target[parts[-1]] = value


def _parse_overrides(arguments: Sequence[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if not token.startswith("--"):
            raise ValueError(f"unexpected positional argument: {token}")
        option = token[2:]
        if "=" in option:
            option, raw_value = option.split("=", 1)
            value = yaml.safe_load(raw_value)
            index += 1
        elif option.startswith(("no-", "no_")):
            option = option[3:]
            value = False
            index += 1
        elif index + 1 < len(arguments) and not arguments[
            index + 1
        ].startswith("--"):
            value = yaml.safe_load(arguments[index + 1])
            index += 2
        else:
            value = True
            index += 1
        name = option.replace("-", "_")
        _set_nested(overrides, name, value)
    return overrides


def _merge_nested(target: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_nested(target[key], value)
        else:
            target[key] = value


def load_legacy_mapping(
    config_path: str | Path | None,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """Load old YAML and command-line names without importing JAX or Brax."""

    raw: dict[str, Any] = {}
    if config_path is not None:
        loaded = yaml.safe_load(Path(config_path).read_text())
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise TypeError("RTRRL YAML root must be a mapping")
            raw.update(loaded)
    _merge_nested(raw, _parse_overrides(overrides))
    return raw


def normalize_legacy_invocation(raw: Any):
    """Normalize legacy budget aliases before the frozen Memorax schema."""

    if isinstance(raw, LegacyRTRRLConfig):
        return normalize_legacy_config(raw)
    values = dict(raw) if isinstance(raw, Mapping) else dict(vars(raw))
    if not isinstance(raw, Mapping) and "rnn_grad_clip" in values:
        optimizer = values.get("optimizer_params_rnn")
        if getattr(optimizer, "gradient_clip", None) == 1.0:
            optimizer_values = dict(vars(optimizer))
            optimizer_values.pop("gradient_clip", None)
            values["optimizer_params_rnn"] = optimizer_values
    environment = values.get("env_params")
    environment_num_envs = (
        environment.get("batch_size")
        if isinstance(environment, Mapping)
        else getattr(environment, "batch_size", None)
    )
    legacy_num_envs = values.get(
        "num_envs",
        environment_num_envs if environment_num_envs is not None else 1,
    )
    if "episodes" in values:
        values.setdefault("num_epochs", values["episodes"])
        values.setdefault(
            "total_timesteps",
            int(values["episodes"])
            * int(values.get("steps", 1000))
            * int(legacy_num_envs or 1),
        )
    return normalize_legacy_config(values)


def describe_legacy_build(config) -> dict[str, Any]:
    """Resolve the effective static AgentProgram recipe without an environment."""

    effective = to_component_config(config)
    environment_kwargs = dict(config.env_params.init_kwargs)
    backend = environment_kwargs.get("backend", config.backend)
    return {
        "environment_started": False,
        "jax_imported": "jax" in sys.modules,
        "effective": {
            "total_timesteps": config.total_timesteps,
            "num_epochs": config.num_epochs,
            "num_envs": config.num_envs,
            "profile": effective.profile,
            "logging": config.logging,
            "run_name": config.run_name,
            "td_learning_rate": (
                effective.optimizer_params_td.learning_rate
            ),
            "rnn_learning_rate": (
                effective.optimizer_params_rnn.learning_rate
            ),
            "rnn_gradient_clip": (
                effective.optimizer_params_rnn.gradient_clip
            ),
            "environment": {
                "env_name": config.env_params.env_name,
                "mode": config.mode,
                "backend": backend,
            },
            "builder": {
                "function": (
                    "build_independent_rtrrl_agent"
                    if effective.topology == "independent"
                    else "build_rtrrl_agent"
                ),
                "topology": effective.topology,
                "recurrent_component": effective.recurrent_component,
                "feature_component": effective.feature_component,
                "actor_component": effective.actor_component,
                "meta_rl": effective.meta_rl,
                "normalize_observation": effective.normalize_observation,
                "normalize_reward": effective.normalize_reward,
                "pass_observation": config.pass_obs,
            },
        },
    }


def repository_rtrrl_configs(repository_root: str | Path) -> tuple[Path, ...]:
    root = Path(repository_root)
    paths = (
        path
        for config_directory in (
            root / "rtrrl" / "config",
            root / "memo" / "config",
        )
        for path in config_directory.glob("*")
        if path.is_file()
        and path.suffix.lower() in {".yml", ".yaml"}
        and "rtrrl" in path.stem.lower()
    )
    return tuple(sorted(paths))


def audit_repository_configs(repository_root: str | Path) -> dict[str, Any]:
    """Parse and classify every repository RTRRL YAML without runtime startup."""

    classified: dict[str, list[dict[str, Any]]] = {
        "accepted": [],
        "unsupported": [],
        "unknown_fields": [],
        "deprecated_no_op": [],
        "invalid_config": [],
    }
    root = Path(repository_root)
    for path in repository_rtrrl_configs(root):
        relative = str(path.relative_to(root))
        try:
            config = normalize_legacy_invocation(load_legacy_mapping(path))
            to_component_config(config)
        except UnsupportedRTRRLBranch as error:
            classified["unsupported"].append(
                {"path": relative, "reason": str(error)}
            )
        except UnknownRTRRLField as error:
            classified["unknown_fields"].append(
                {"path": relative, "reason": str(error)}
            )
        except (
            InvalidRTRRLConfig,
            TypeError,
            ValueError,
            yaml.YAMLError,
        ) as error:
            classified["invalid_config"].append(
                {"path": relative, "reason": str(error)}
            )
        else:
            category = (
                "deprecated_no_op"
                if config.warning_records
                else "accepted"
            )
            classified[category].append(
                {
                    "path": relative,
                    "warnings": [
                        asdict(record) for record in config.warning_records
                    ],
                }
            )
    counts = {name: len(records) for name, records in classified.items()}
    discovered = sum(counts.values())
    return {
        "expected_plan_count": EXPECTED_PLAN_CONFIG_COUNT,
        "discovered": discovered,
        "count_delta": discovered - EXPECTED_PLAN_CONFIG_COUNT,
        "counts": counts,
        "files": classified,
    }


def parse_compatibility_cli(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Historical RTRRL CLI backed by Memorax AgentProgram",
        allow_abbrev=False,
    )
    parser.add_argument("--config_path", "--config-path")
    parser.add_argument(
        "--compat-action",
        choices=("train", "build", "audit", "mock-epoch"),
        default="train",
    )
    parsed, overrides = parser.parse_known_args(argv)
    return parsed, overrides


def emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True))


__all__ = [
    "audit_repository_configs",
    "describe_legacy_build",
    "emit_json",
    "historical_rtrrl_metrics",
    "load_legacy_mapping",
    "normalize_legacy_invocation",
    "parse_compatibility_cli",
    "repository_rtrrl_configs",
    "run_mock_epoch",
]
