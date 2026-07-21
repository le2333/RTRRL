from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[2]
_CHILD_PROGRAM = r"""
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType

repository_root = Path(sys.argv[1])
script = sys.argv[2]
memo_root = repository_root / "memo"
sys.path[:0] = [
    str(repository_root / "rtrrl" / "infra" / "control-plane" / "src"),
    str(memo_root / "experiments"),
    str(memo_root),
]

optuna_module = ModuleType("optuna")
trial_module = ModuleType("optuna.trial")
trial_module.Trial = object
optuna_module.trial = trial_module
sys.modules["optuna"] = optuna_module
sys.modules["optuna.trial"] = trial_module

from experiments.base.facility import (
    FacilityInput,
    build_rtrrl_config,
    build_stream_ac_config,
)
from trainer_infra.image_catalog import load_catalog_index
from trainer_infra.materialize import materialize_run
from trainer_infra.models import (
    ExecutionSpec,
    ExperimentDefaults,
    ExperimentIdentity,
    ExperimentSpec,
    GroupSpec,
    HpoSpec,
    ParameterPolicy,
    ResourcesSpec,
)
from trainer_infra.resolve import resolve_experiment

class FakeTrial:
    number = 0

image = "repo/memo@sha256:" + "a" * 64
catalog = load_catalog_index(memo_root / "infra" / "scripts" / "index.yaml")
spec = ExperimentSpec(
    experiment=ExperimentIdentity(name=f"{script}-contract"),
    defaults=ExperimentDefaults(
        image=image,
        resources=ResourcesSpec(profile="c7am"),
        hpo=HpoSpec(
            total_trials=1,
            configs_per_batch=1,
            parameter_policy=ParameterPolicy.EXPLICIT_SCAN,
        ),
        execution=ExecutionSpec(runs_per_job=1),
    ),
    groups={"group": GroupSpec(script=script)},
)
group = resolve_experiment(spec, {image: catalog}).groups[0]
concrete = materialize_run(group, FakeTrial(), {}, run_number=1)
with tempfile.TemporaryDirectory() as temporary:
    config_path = Path(temporary) / f"{script}.yaml"
    config_path.write_text(concrete.config_yaml)
    facility = FacilityInput.load(config_path)
    builder = (
        build_stream_ac_config if script == "memo_stream_ac" else build_rtrrl_config
    )
    config = builder(facility)
print(json.dumps({
    "agent_type": getattr(config, "agent_type", None),
    "rtrrl_topology": getattr(config, "rtrrl_topology", None),
    "backbone": getattr(config, "backbone", None),
    "hidden_dim": config.hidden_dim,
    "encoder_dim": config.encoder_dim,
    "num_envs": config.num_envs,
    "total_timesteps": config.total_timesteps,
}))
"""


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        (
            "memo_stream_ac",
            {
                "agent_type": "rtu_rtrl",
                "rtrrl_topology": None,
                "backbone": None,
                "hidden_dim": 192,
                "encoder_dim": 64,
                "num_envs": 16,
                "total_timesteps": 500_000,
            },
        ),
        (
            "memo_rtrrl",
            {
                "agent_type": "rtu_rtrl",
                "rtrrl_topology": "shared",
                "backbone": "lru",
                "hidden_dim": 32,
                "encoder_dim": 32,
                "num_envs": 1,
                "total_timesteps": 1_000_000,
            },
        ),
    ],
)
def test_real_catalog_materializes_through_facility_input_into_real_config(
    script: str,
    expected: dict[str, object],
) -> None:
    original_path = tuple(sys.path)
    original_modules = set(sys.modules)

    completed = subprocess.run(
        [sys.executable, "-c", _CHILD_PROGRAM, str(REPOSITORY_ROOT), script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == expected
    assert tuple(sys.path) == original_path
    assert set(sys.modules) == original_modules
