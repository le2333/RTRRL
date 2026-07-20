"""Lightweight migration helpers for the historical RTRRL command line."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Sequence

import yaml

from .compatibility import (
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
    """Exercise the production metric translation with a pinned mock summary."""

    summary = SimpleNamespace(
        steps=30,
        mean_reward=4.0,
        num_episodes=1,
        mean_delta=0.8333333134651184,
        mean_r_bar=0.10000000149011612,
        mean_v=20.0,
        total_td_loss=19.399999618530273,
        actor_loss=-0.6000000238418579,
        critic_loss=20.0,
        entropy=0.4000000059604645,
        v_targ=20.66666603088379,
        magnitude_loss=None,
        learning_rate_td=0.0010000000474974513,
        learning_rate_rnn=0.00019999999494757503,
        norms={
            "['z']['trace']": 5.0,
            "['params']['weight']": 10.0,
            "['slow_params']['weight']": 13.0,
        },
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
        name = token[2:].replace("-", "_")
        if index + 1 < len(arguments) and not arguments[index + 1].startswith("--"):
            value = yaml.safe_load(arguments[index + 1])
            index += 2
        else:
            value = True
            index += 1
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


def normalize_legacy_invocation(raw: dict[str, Any]):
    """Normalize legacy budget aliases before the frozen Memorax schema."""

    values = dict(raw)
    environment = values.get("env_params")
    legacy_num_envs = (
        environment.get("batch_size", 1)
        if isinstance(environment, dict)
        else values.get("num_envs", 1)
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
    return {
        "environment_started": False,
        "jax_imported": "jax" in sys.modules,
        "construction": "memorax.RTRRL/AgentProgram",
        "effective": {
            "total_timesteps": config.total_timesteps,
            "num_epochs": config.num_epochs,
            "num_envs": config.num_envs,
            "profile": effective.profile,
            "logging": config.logging,
            "run_name": config.run_name,
            "td_learning_rate": config.td_lr,
            "rnn_learning_rate": config.rnn_lr,
            "rnn_gradient_clip": config.rnn_grad_clip,
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
    }
    root = Path(repository_root)
    for path in repository_rtrrl_configs(root):
        relative = str(path.relative_to(root))
        try:
            config = normalize_legacy_invocation(load_legacy_mapping(path))
        except UnsupportedRTRRLBranch as error:
            classified["unsupported"].append(
                {"path": relative, "reason": str(error)}
            )
        except ValueError as error:
            classified["unknown_fields"].append(
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
        description="Historical RTRRL CLI backed by Memorax AgentProgram"
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
