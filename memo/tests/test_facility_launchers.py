from __future__ import annotations

import subprocess
import sys
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


@pytest.mark.parametrize("bad", [{1: "value"}, {"x": float("nan")}, {"x": {1, 2}}])
def test_facility_input_rejects_non_json_recursively(tmp_path, bad):
    with pytest.raises((TypeError, ValueError), match="JSON|finite|string"):
        FacilityInput(
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
