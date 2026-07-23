from __future__ import annotations

import copy
import dataclasses
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from brax_ppo_acceptance.config import AcceptanceConfig

VALID: dict[str, Any] = {
    "protocol_version": "1",
    "environment": {
        "name": "inverted_pendulum",
        "options": {"backend": "generalized"},
    },
    "logging": {"aim_every_env_steps": 1, "rerun_every_episodes": 1},
    "parameters": {
        "runtime": {"seed": 7},
        "algorithm": {
            "learning_rate": 0.0003,
            "num_envs": 4,
            "episode_length": 32,
            "failure_mode": "none",
        },
    },
    "training_budget": {"env_steps": 128},
}

KEY_PATHS = (
    (),
    ("environment",),
    ("environment", "options"),
    ("logging",),
    ("parameters",),
    ("parameters", "runtime"),
    ("parameters", "algorithm"),
    ("training_budget",),
)


def _write(tmp_path: Path, payload: Any) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _mapping_at(payload: MutableMapping[str, Any], path: tuple[str, ...]) -> MutableMapping[str, Any]:
    current = payload
    for key in path:
        current = current[key]
    return current


def test_loads_valid_config_and_resolves_parameters(tmp_path: Path) -> None:
    config = AcceptanceConfig.load(_write(tmp_path, VALID), environ={})

    assert dataclasses.is_dataclass(config)
    assert config.protocol_version == "1"
    assert config.environment_name == "inverted_pendulum"
    assert config.backend == "generalized"
    assert config.aim_every_env_steps == 1
    assert config.rerun_every_episodes == 1
    assert config.seed == 7
    assert config.learning_rate == pytest.approx(0.0003)
    assert config.num_envs == 4
    assert config.episode_length == 32
    assert config.num_timesteps == 128
    assert config.failure_mode == "none"
    assert config.fast_mode is False


@pytest.mark.parametrize("key_path", KEY_PATHS)
def test_rejects_every_missing_field(tmp_path: Path, key_path: tuple[str, ...]) -> None:
    payload = copy.deepcopy(VALID)
    target = _mapping_at(payload, key_path)
    target.pop(next(iter(target)))

    with pytest.raises(ValueError, match="keys"):
        AcceptanceConfig.load(_write(tmp_path, payload), environ={})


@pytest.mark.parametrize("key_path", KEY_PATHS)
def test_rejects_extra_fields_at_every_level(tmp_path: Path, key_path: tuple[str, ...]) -> None:
    payload = copy.deepcopy(VALID)
    _mapping_at(payload, key_path)["unexpected"] = "value"

    with pytest.raises(ValueError, match="keys"):
        AcceptanceConfig.load(_write(tmp_path, payload), environ={})


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("logging", "aim_every_env_steps"), True),
        (("logging", "rerun_every_episodes"), False),
        (("parameters", "runtime", "seed"), True),
        (("parameters", "algorithm", "num_envs"), True),
        (("parameters", "algorithm", "episode_length"), False),
        (("training_budget", "env_steps"), True),
    ],
)
def test_rejects_bool_where_integer_is_required(
    tmp_path: Path, path: tuple[str, ...], value: bool
) -> None:
    payload = copy.deepcopy(VALID)
    parent = _mapping_at(payload, path[:-1])
    parent[path[-1]] = value

    with pytest.raises(ValueError, match="integer"):
        AcceptanceConfig.load(_write(tmp_path, payload), environ={})


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("logging", "aim_every_env_steps"), 0),
        (("logging", "rerun_every_episodes"), -1),
        (("parameters", "runtime", "seed"), -1),
        (("parameters", "algorithm", "num_envs"), 0),
        (("parameters", "algorithm", "episode_length"), -1),
        (("training_budget", "env_steps"), 0),
    ],
)
def test_rejects_out_of_range_integers(
    tmp_path: Path, path: tuple[str, ...], value: int
) -> None:
    payload = copy.deepcopy(VALID)
    parent = _mapping_at(payload, path[:-1])
    parent[path[-1]] = value

    with pytest.raises(ValueError, match="positive|non-negative"):
        AcceptanceConfig.load(_write(tmp_path, payload), environ={})


@pytest.mark.parametrize("value", [True, "0.0003", float("nan"), float("inf"), -float("inf")])
def test_rejects_non_numeric_or_non_finite_learning_rate(tmp_path: Path, value: Any) -> None:
    payload = copy.deepcopy(VALID)
    payload["parameters"]["algorithm"]["learning_rate"] = value

    with pytest.raises(ValueError, match="finite number"):
        AcceptanceConfig.load(_write(tmp_path, payload), environ={})


def test_rejects_non_positive_learning_rate(tmp_path: Path) -> None:
    payload = copy.deepcopy(VALID)
    payload["parameters"]["algorithm"]["learning_rate"] = 0.0

    with pytest.raises(ValueError, match="positive"):
        AcceptanceConfig.load(_write(tmp_path, payload), environ={})


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("protocol_version",), "2", "protocol_version"),
        (("environment", "name"), "ant", "inverted_pendulum"),
        (("environment", "options", "backend"), "spring", "generalized"),
        (("parameters", "algorithm", "failure_mode"), "sometimes", "failure_mode"),
    ],
)
def test_rejects_unsupported_literal_values(
    tmp_path: Path, path: tuple[str, ...], value: str, message: str
) -> None:
    payload = copy.deepcopy(VALID)
    parent = _mapping_at(payload, path[:-1])
    parent[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        AcceptanceConfig.load(_write(tmp_path, payload), environ={})


def test_rejects_non_mapping_yaml_document(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mapping"):
        AcceptanceConfig.load(_write(tmp_path, ["not", "a", "mapping"]), environ={})


def test_rejects_environment_budget_mismatch(tmp_path: Path) -> None:
    payload = copy.deepcopy(VALID)
    payload["training_budget"]["env_steps"] = 64

    with pytest.raises(ValueError, match="num_envs \\* episode_length"):
        AcceptanceConfig.load(_write(tmp_path, payload), environ={})


@pytest.mark.parametrize("failure_mode", ["before_training", "after_training", "after_checkpoint"])
def test_non_none_failure_requires_test_environment(
    tmp_path: Path, failure_mode: str
) -> None:
    payload = copy.deepcopy(VALID)
    payload["parameters"]["algorithm"]["failure_mode"] = failure_mode
    path = _write(tmp_path, payload)

    with pytest.raises(ValueError, match="BRAX_ACCEPTANCE_TEST_MODE=1"):
        AcceptanceConfig.load(path, environ={})
    assert (
        AcceptanceConfig.load(path, environ={"BRAX_ACCEPTANCE_TEST_MODE": "1"}).failure_mode
        == failure_mode
    )


def test_fast_mode_requires_both_environment_gates(tmp_path: Path) -> None:
    path = _write(tmp_path, VALID)

    with pytest.raises(ValueError, match="BRAX_ACCEPTANCE_TEST_MODE=1"):
        AcceptanceConfig.load(path, environ={"BRAX_ACCEPTANCE_E2E_FAST": "1"})
    assert AcceptanceConfig.load(
        path,
        environ={
            "BRAX_ACCEPTANCE_TEST_MODE": "1",
            "BRAX_ACCEPTANCE_E2E_FAST": "1",
        },
    ).fast_mode


@pytest.mark.parametrize(
    "environ",
    [
        {"BRAX_ACCEPTANCE_TEST_MODE": "yes"},
        {"BRAX_ACCEPTANCE_E2E_FAST": "yes"},
        {"BRAX_ACCEPTANCE_TEST_MODE": "1", "BRAX_ACCEPTANCE_E2E_FAST": "0"},
    ],
)
def test_rejects_invalid_gate_values(tmp_path: Path, environ: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="must be exactly '1'"):
        AcceptanceConfig.load(_write(tmp_path, VALID), environ=environ)


def test_default_environment_is_captured_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = copy.deepcopy(VALID)
    payload["parameters"]["algorithm"]["failure_mode"] = "before_training"
    monkeypatch.setenv("BRAX_ACCEPTANCE_TEST_MODE", "1")

    with pytest.raises(ValueError, match="BRAX_ACCEPTANCE_TEST_MODE=1"):
        AcceptanceConfig.load(_write(tmp_path, payload))


def test_nested_configuration_is_deeply_immutable(tmp_path: Path) -> None:
    config = AcceptanceConfig.load(_write(tmp_path, VALID), environ={})

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.protocol_version = "2"  # type: ignore[misc]
    with pytest.raises(TypeError):
        config.environment["name"] = "ant"  # type: ignore[index]
    with pytest.raises(TypeError):
        config.environment["options"]["backend"] = "spring"  # type: ignore[index]
    with pytest.raises(TypeError):
        config.parameters["runtime"]["seed"] = 9  # type: ignore[index]
