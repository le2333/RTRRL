from __future__ import annotations

import subprocess
import sys
from importlib import import_module
from pathlib import Path

import pytest
import yaml

MEMO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(MEMO_ROOT))
sys.path.insert(0, str(MEMO_ROOT / "experiments"))

from base.facility import (  # noqa: E402
    FacilityInput,
    build_rtrrl_config,
    build_stream_ac_config,
)


def _write(tmp_path: Path, **updates) -> Path:
    payload = {
        "protocol_version": "1",
        "environment": {
            "name": "memory_chain",
            "options": {"length": 7, "max_episode_steps": 7},
        },
        "logging": {"aim_every_env_steps": 10, "rerun_every_episodes": 2},
        "parameters": {"agent_type": "rtu_rtrl", "num_envs": 1, "num_epochs": 2},
        "training_budget": {"env_steps": 8},
    }
    payload.update(updates)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def test_facility_input_requires_exact_concrete_top_level(tmp_path):
    path = _write(tmp_path)
    payload = yaml.safe_load(path.read_text())
    payload["extra"] = True
    path.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match="unknown.*extra"):
        FacilityInput.load(path)


@pytest.mark.parametrize("protocol_version", [None, "2", 1])
def test_facility_input_requires_protocol_version_one(tmp_path, protocol_version):
    path = _write(tmp_path)
    payload = yaml.safe_load(path.read_text())
    if protocol_version is None:
        del payload["protocol_version"]
    else:
        payload["protocol_version"] = protocol_version
    path.write_text(yaml.safe_dump(payload))

    with pytest.raises((TypeError, ValueError), match="protocol_version|missing"):
        FacilityInput.load(path)


@pytest.mark.parametrize("bad", [{1: "value"}, {"x": float("nan")}, {"x": {1, 2}}])
def test_facility_input_rejects_non_json_recursively(tmp_path, bad):
    with pytest.raises((TypeError, ValueError), match="JSON|finite|string"):
        FacilityInput(
            protocol_version="1",
            environment={"name": "memory_chain", "options": {}},
            logging={},
            parameters=bad,
            training_budget={"env_steps": 1},
        )


@pytest.mark.parametrize("environment", ["memory_chain", "kmemory_chain", "mujoco_masked"])
def test_stream_launcher_accepts_only_registered_environments(tmp_path, environment):
    options = {
        "memory_chain": {"length": 8, "max_episode_steps": 8},
        "kmemory_chain": {"length": 8, "num_bits": 2, "max_episode_steps": 8},
        "mujoco_masked": {
            "env_name": "hopper",
            "mode": "F",
            "backend": "spring",
            "max_episode_steps": 17,
        },
    }[environment]
    value = FacilityInput.load(
        _write(tmp_path, environment={"name": environment, "options": options})
    )

    config = build_stream_ac_config(value)

    assert config.agent_type == "rtu_rtrl"
    assert config.total_timesteps == 8
    assert config.max_episode_steps == options["max_episode_steps"]


@pytest.mark.parametrize("agent_type", ["rtu_tbptt", "lru_rtrl"])
def test_stream_launcher_rejects_other_agent_types(tmp_path, agent_type):
    value = FacilityInput.load(
        _write(
            tmp_path,
            parameters={
                "agent_type": agent_type,
                "num_envs": 1,
                "num_epochs": 2,
            },
        )
    )
    with pytest.raises(ValueError, match="rtu_rtrl"):
        build_stream_ac_config(value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("env_name", "humanoid"),
        ("mode", "unknown"),
        ("backend", "invalid"),
    ],
)
def test_mujoco_options_reject_unsupported_values_at_load(tmp_path, field, value):
    options = {
        "env_name": "hopper",
        "mode": "F",
        "backend": "spring",
        "max_episode_steps": 8,
    }
    options[field] = value
    path = _write(
        tmp_path,
        environment={"name": "mujoco_masked", "options": options},
    )

    with pytest.raises(ValueError, match=field):
        FacilityInput.load(path)


def test_rtrrl_launcher_requires_hopper_and_shared(tmp_path):
    path = _write(
        tmp_path,
        environment={
            "name": "hopper",
            "options": {
                "mode": "F",
                "backend": "spring",
                "max_episode_steps": 19,
                "normalize_obs": True,
                "normalize_reward": True,
            },
        },
        parameters={
            "rtrrl_topology": "shared",
            "num_envs": 1,
            "num_epochs": 2,
        },
    )
    config = build_rtrrl_config(FacilityInput.load(path))
    assert config.rtrrl_topology == "shared"
    assert config.env_name == "hopper"
    assert config.max_episode_steps == 19

    value = FacilityInput.load(path)
    object.__setattr__(value, "parameters", {"rtrrl_topology": "independent"})
    with pytest.raises(ValueError, match="shared"):
        build_rtrrl_config(value)


@pytest.mark.parametrize("launcher", ["memo_stream_ac", "memo_rtrrl"])
def test_launcher_cli_exposes_exact_config_flag(launcher):
    result = subprocess.run(
        [sys.executable, str(MEMO_ROOT / "experiments" / launcher / "run.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--config CONFIG" in result.stdout
    assert "--config_path" not in result.stdout


class LifecycleRun:
    def __init__(self):
        self.finished = 0
        self.failed = []

    def finish(self, metrics):
        self.finished += 1

    def fail(self, error):
        self.failed.append(error)


@pytest.mark.parametrize(
    ("module_name", "config_kind"),
    [
        ("memo_stream_ac.run", "stream"),
        ("memo_rtrrl.run", "rtrrl"),
    ],
)
def test_launcher_main_uses_builder_bootstrap_and_success_lifecycle(
    tmp_path, monkeypatch, module_name, config_kind
):
    module = import_module(module_name)
    run = LifecycleRun()
    calls = []
    if config_kind == "stream":
        path = _write(tmp_path)
    else:
        path = _write(
            tmp_path,
            environment={
                "name": "hopper",
                "options": {
                    "mode": "F",
                    "backend": "spring",
                    "max_episode_steps": 8,
                    "normalize_obs": True,
                    "normalize_reward": True,
                },
            },
            parameters={
                "rtrrl_topology": "shared",
                "num_envs": 1,
                "num_epochs": 2,
            },
        )
    original_builder = (
        module.build_stream_ac_config
        if config_kind == "stream"
        else module.build_rtrrl_config
    )

    def builder(value):
        calls.append(("builder", value.environment["name"]))
        return original_builder(value)

    def fake_train(config, logger):
        calls.append(("train", config.experiment))
        logger.finish({"eval/rewards": 1.0})

    if config_kind == "stream":
        monkeypatch.setitem(
            module._TRAINERS,
            "memory_chain",
            fake_train,
        )
    else:
        monkeypatch.setattr(module, "train", fake_train)

    def execute(trainer, config, project_name):
        calls.append(("run_experiment", project_name))
        trainer(config, run)

    monkeypatch.setattr(
        module,
        "build_stream_ac_config" if config_kind == "stream" else "build_rtrrl_config",
        builder,
    )
    monkeypatch.setattr(module, "bootstrap_from_environment", lambda: run)
    monkeypatch.setattr(module, "run_experiment", execute)

    module.main(["--config", str(path)])

    assert [call[0] for call in calls] == [
        "builder",
        "run_experiment",
        "train",
    ]
    assert run.finished == 1
    assert run.failed == []


def test_launcher_main_failure_fails_once_without_finish(tmp_path, monkeypatch):
    module = import_module("memo_stream_ac.run")
    run = LifecycleRun()
    error = RuntimeError("training failed")
    monkeypatch.setattr(module, "bootstrap_from_environment", lambda: run)
    monkeypatch.setattr(
        module,
        "run_experiment",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(RuntimeError) as raised:
        module.main(["--config", str(_write(tmp_path))])

    assert raised.value is error
    assert run.failed == [error]
    assert run.finished == 0
