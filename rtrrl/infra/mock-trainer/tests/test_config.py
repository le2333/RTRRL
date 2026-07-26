from __future__ import annotations

import copy
import dataclasses
import inspect
import os
import subprocess
import sys
from collections import UserDict
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

REQUIRED_FIELD_PATHS = (
    ("protocol_version",),
    ("environment",),
    ("logging",),
    ("parameters",),
    ("training_budget",),
    ("environment", "name"),
    ("environment", "options"),
    ("environment", "options", "backend"),
    ("logging", "aim_every_env_steps"),
    ("logging", "rerun_every_episodes"),
    ("parameters", "runtime"),
    ("parameters", "algorithm"),
    ("parameters", "runtime", "seed"),
    ("parameters", "algorithm", "learning_rate"),
    ("parameters", "algorithm", "num_envs"),
    ("parameters", "algorithm", "episode_length"),
    ("parameters", "algorithm", "failure_mode"),
    ("training_budget", "env_steps"),
)

KEY_PATHS = tuple(path[:-1] for path in REQUIRED_FIELD_PATHS)


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


@pytest.mark.parametrize("field_path", REQUIRED_FIELD_PATHS)
def test_rejects_every_missing_field(tmp_path: Path, field_path: tuple[str, ...]) -> None:
    payload = copy.deepcopy(VALID)
    target = _mapping_at(payload, field_path[:-1])
    target.pop(field_path[-1])

    with pytest.raises(ValueError, match="keys"):
        AcceptanceConfig.load(_write(tmp_path, payload), environ={})


@pytest.mark.parametrize("key_path", sorted(set(KEY_PATHS)))
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


def test_rejects_learning_rate_that_overflows_float(tmp_path: Path) -> None:
    payload = copy.deepcopy(VALID)
    payload["parameters"]["algorithm"]["learning_rate"] = 10**1000

    with pytest.raises(ValueError, match="finite number"):
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


def test_budget_is_independent_of_environment_shape(tmp_path: Path) -> None:
    payload = copy.deepcopy(VALID)
    payload["training_budget"]["env_steps"] = 65

    config = AcceptanceConfig.load(_write(tmp_path, payload), environ={})

    assert config.num_timesteps == 65


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


def test_default_environment_is_read_live(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = copy.deepcopy(VALID)
    payload["parameters"]["algorithm"]["failure_mode"] = "before_training"
    path = _write(tmp_path, payload)
    monkeypatch.setenv("BRAX_ACCEPTANCE_TEST_MODE", "1")

    assert AcceptanceConfig.load(path).failure_mode == "before_training"

    monkeypatch.delenv("BRAX_ACCEPTANCE_TEST_MODE")
    with pytest.raises(ValueError, match="BRAX_ACCEPTANCE_TEST_MODE=1"):
        AcceptanceConfig.load(path)


def test_load_signature_defaults_to_live_os_environ_object() -> None:
    parameter = inspect.signature(AcceptanceConfig.load).parameters["environ"]

    assert parameter.annotation == "Mapping[str, str]"
    assert parameter.default is os.environ


def test_explicit_none_environment_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="environ"):
        AcceptanceConfig.load(_write(tmp_path, VALID), environ=None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "arguments",
    [
        ("1", {}, {}, {}, {}, False),
        ("1", [], [], [], [], False),
        ("1", UserDict(), UserDict(), UserDict(), UserDict(), False),
    ],
)
def test_direct_construction_is_rejected(arguments: tuple[Any, ...]) -> None:
    with pytest.raises(TypeError, match="AcceptanceConfig.load"):
        AcceptanceConfig(*arguments)


def test_source_mapping_drift_does_not_change_instance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = UserDict(copy.deepcopy(VALID))
    monkeypatch.setattr(yaml, "safe_load", lambda _: payload)
    path = tmp_path / "config.yaml"
    path.write_text("ignored by patched loader", encoding="utf-8")

    config = AcceptanceConfig.load(path, environ={})
    payload["environment"]["name"] = "ant"
    payload["parameters"]["runtime"]["seed"] = 99

    assert config.environment_name == "inverted_pendulum"
    assert config.seed == 7


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


def test_package_is_importable_from_installed_environment(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import brax_ppo_acceptance; print(brax_ppo_acceptance.__file__)",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    expected = Path(__file__).parents[1] / "src/brax_ppo_acceptance/__init__.py"
    assert Path(completed.stdout.strip()).resolve() == expected.resolve()
