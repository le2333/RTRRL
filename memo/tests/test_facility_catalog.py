from __future__ import annotations

import sys
from types import ModuleType
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[2]
MEMO_ROOT = REPOSITORY_ROOT / "memo"
CONTROL_PLANE_SRC = REPOSITORY_ROOT / "rtrrl" / "infra" / "control-plane" / "src"
sys.path.insert(0, str(MEMO_ROOT))
sys.path.insert(0, str(MEMO_ROOT / "experiments"))
sys.path.insert(0, str(CONTROL_PLANE_SRC))

if "optuna.trial" not in sys.modules:
    optuna_module = ModuleType("optuna")
    trial_module = ModuleType("optuna.trial")
    trial_module.Trial = object
    optuna_module.trial = trial_module
    sys.modules["optuna"] = optuna_module
    sys.modules["optuna.trial"] = trial_module

from experiments.base.facility import (  # noqa: E402
    FacilityInput,
    build_rtrrl_config,
    build_stream_ac_config,
)
from trainer_infra.image_catalog import load_catalog_index  # noqa: E402
from trainer_infra.materialize import materialize_run  # noqa: E402
from trainer_infra.models import (  # noqa: E402
    ExecutionSpec,
    ExperimentDefaults,
    ExperimentIdentity,
    ExperimentSpec,
    GroupSpec,
    HpoSpec,
    ParameterPolicy,
    ResourcesSpec,
)
from trainer_infra.resolve import resolve_experiment  # noqa: E402

CATALOG_INDEX = REPOSITORY_ROOT / "memo" / "infra" / "scripts" / "index.yaml"


class FakeTrial:
    number = 0


def _resolved_group(script: str):
    image = "repo/memo@sha256:" + "a" * 64
    catalog = load_catalog_index(CATALOG_INDEX)
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
    return resolve_experiment(spec, {image: catalog}).groups[0]


@pytest.mark.parametrize(
    ("script", "builder", "expected"),
    [
        (
            "memo_stream_ac",
            build_stream_ac_config,
            {
                "agent_type": "rtu_rtrl",
                "hidden_dim": 192,
                "encoder_dim": 64,
                "num_envs": 16,
                "total_timesteps": 500_000,
            },
        ),
        (
            "memo_rtrrl",
            build_rtrrl_config,
            {
                "rtrrl_topology": "shared",
                "backbone": "lru",
                "hidden_dim": 32,
                "total_timesteps": 1_000_000,
            },
        ),
    ],
)
def test_real_catalog_materializes_through_facility_input_into_real_config(
    tmp_path: Path,
    script: str,
    builder,
    expected: dict[str, object],
) -> None:
    concrete = materialize_run(
        _resolved_group(script),
        FakeTrial(),
        {},
        run_number=1,
    )
    config_path = tmp_path / f"{script}.yaml"
    config_path.write_text(concrete.config_yaml)

    config = builder(FacilityInput.load(config_path))

    for name, value in expected.items():
        assert getattr(config, name) == value
