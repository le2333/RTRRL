from __future__ import annotations

import subprocess
import sys
import types
from importlib import import_module
from pathlib import Path

import pytest
import yaml

MEMO_ROOT = Path(__file__).parents[1]
REPOSITORY_ROOT = MEMO_ROOT.parent
sys.path.insert(0, str(MEMO_ROOT))
sys.path.insert(0, str(MEMO_ROOT / "experiments"))
sys.path.insert(
    0,
    str(REPOSITORY_ROOT / "rtrrl" / "infra" / "control-plane" / "src"),
)

if "optuna" not in sys.modules:
    optuna_module = types.ModuleType("optuna")
    optuna_trial_module = types.ModuleType("optuna.trial")
    optuna_trial_module.Trial = object
    optuna_module.trial = optuna_trial_module
    sys.modules["optuna"] = optuna_module
    sys.modules["optuna.trial"] = optuna_trial_module

from base.facility import (  # noqa: E402
    FacilityInput,
    build_rtrrl_config,
    build_stream_ac_config,
)
from trainer_infra.materialize import materialize_run  # noqa: E402
from trainer_infra.models import (  # noqa: E402
    DescriptorDefaults,
    EnvironmentSpec,
    ExecutionSpec,
    ExperimentDefaults,
    ExperimentIdentity,
    ExperimentSpec,
    FieldDescriptor,
    GroupSpec,
    HpoSpec,
    LoggingSpec,
    ObjectiveSpec,
    ResourcesSpec,
    ScriptCatalog,
    ScriptDescriptor,
    TrainingBudgetSpec,
)
from trainer_infra.resolve import resolve_experiment  # noqa: E402


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
    assert config.patience == 0
    assert config.require_full_budget is True


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
    assert config.patience == 0
    assert config.require_full_budget is True

    value = FacilityInput.load(path)
    object.__setattr__(
        value,
        "parameters",
        {"rtrrl_topology": "independent", "num_envs": 1, "num_epochs": 2},
    )
    with pytest.raises(ValueError, match="shared"):
        build_rtrrl_config(value)


@pytest.mark.parametrize(
    ("field", "value"),
    [("env_name", "ant"), ("mode", "bad"), ("backend", "bad")],
)
def test_hopper_options_reject_unsupported_values_at_load(tmp_path, field, value):
    options = {
        "env_name": "hopper",
        "mode": "F",
        "backend": "spring",
        "max_episode_steps": 8,
        "normalize_obs": True,
        "normalize_reward": True,
    }
    options[field] = value
    path = _write(
        tmp_path,
        environment={"name": "hopper", "options": options},
        parameters={
            "rtrrl_topology": "shared",
            "num_envs": 1,
            "num_epochs": 2,
        },
    )

    with pytest.raises(ValueError, match=field):
        FacilityInput.load(path)


def test_nested_parameter_paths_reject_unknown_leaf_early(tmp_path):
    value = FacilityInput.load(
        _write(
            tmp_path,
            parameters={
                "algorithm": {"agent_type": "rtu_rtrl"},
                "network": {"not_a_real_field": 32},
                "num_envs": 1,
                "num_epochs": 2,
            },
        )
    )

    with pytest.raises(ValueError, match="network.not_a_real_field"):
        build_stream_ac_config(value)


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


@pytest.mark.parametrize("kind", ["stream", "rtrrl"])
def test_real_descriptor_paths_reach_real_memo_config_dataclass(tmp_path, kind):
    image = "repo/memo@sha256:" + "a" * 64
    is_stream = kind == "stream"
    environment = (
        EnvironmentSpec(
            name="memory_chain",
            options={"length": 8, "max_episode_steps": 8},
        )
        if is_stream
        else EnvironmentSpec(
            name="hopper",
            options={
                "env_name": "hopper",
                "mode": "F",
                "backend": "spring",
                "max_episode_steps": 8,
                "normalize_obs": True,
                "normalize_reward": True,
            },
        )
    )
    topology_name = "agent_type" if is_stream else "rtrrl_topology"
    topology_value = "rtu_rtrl" if is_stream else "shared"
    script_name = "memo_stream_ac" if is_stream else "memo_rtrrl"
    descriptor = ScriptDescriptor(
        name=script_name,
        argv=("python", "run.py", "--config", "{config_path}"),
        sdk_protocol_version="1",
        defaults=DescriptorDefaults(
            environment=environment,
            training_budget=TrainingBudgetSpec(env_steps=8),
            logging=LoggingSpec(
                aim_every_env_steps=4,
                rerun_every_episodes=1,
            ),
        ),
        objective=ObjectiveSpec(
            metric="eval/rewards",
            direction="maximize",
            reduction="last",
        ),
        environments=(environment.name,),
        fields={
            topology_name: FieldDescriptor(
                path=f"algorithm.{topology_name}",
                type="str",
                default=topology_value,
                choices=(topology_value,),
            ),
            "hidden_dim": FieldDescriptor(
                path="network.hidden_dim",
                type="int",
                default=32,
            ),
            "seed": FieldDescriptor(
                path="runtime.seed",
                type="int",
                default=11,
            ),
            "num_envs": FieldDescriptor(
                path="runtime.num_envs",
                type="int",
                default=1,
            ),
            "num_epochs": FieldDescriptor(
                path="runtime.num_epochs",
                type="int",
                default=2,
            ),
        },
    )
    spec = ExperimentSpec(
        experiment=ExperimentIdentity(name=f"{kind}-contract"),
        defaults=ExperimentDefaults(
            image=image,
            resources=ResourcesSpec(profile="c7am"),
            hpo=HpoSpec(total_trials=1, configs_per_batch=1),
            execution=ExecutionSpec(runs_per_job=1),
        ),
        groups={"group": GroupSpec(script=script_name)},
    )
    group = resolve_experiment(
        spec,
        {
            image: ScriptCatalog(
                protocol_version="1",
                scripts={script_name: descriptor},
            )
        },
    ).groups[0]
    trial = types.SimpleNamespace(number=0)
    concrete = materialize_run(group, trial, {}, run_number=1)
    path = tmp_path / f"{kind}.yaml"
    path.write_text(concrete.config_yaml)
    facility = FacilityInput.load(path)

    config = (
        build_stream_ac_config(facility)
        if is_stream
        else build_rtrrl_config(facility)
    )

    assert config.hidden_dim == 32
    assert config.seed == 11
    assert config.num_envs == 1
    assert config.num_epochs == 2
    assert getattr(config, topology_name) == topology_value
    assert concrete.final_parameters == {
        topology_name: topology_value,
        "hidden_dim": 32,
        "seed": 11,
        "num_envs": 1,
        "num_epochs": 2,
    }
