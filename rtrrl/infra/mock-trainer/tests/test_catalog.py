import re
import sys
from pathlib import Path

from trainer_infra.image_catalog import (
    decode_catalog,
    encode_catalog,
    load_catalog_index,
)
from trainer_infra.materialize import materialize_run
from trainer_infra.models import (
    ExecutionSpec,
    ExperimentDefaults,
    ExperimentIdentity,
    ExperimentSpec,
    GroupSpec,
    HpoSpec,
    ResourcesSpec,
)
from trainer_infra.resolve import resolve_experiment

ROOT = Path(__file__).parents[4]
SCRIPTS = ROOT / "rtrrl" / "infra" / "mock-trainer" / "scripts"
sys.path.insert(0, str(ROOT / "rtrrl" / "infra" / "mock-trainer" / "src"))

from brax_ppo_acceptance.config import AcceptanceConfig


class FakeTrial:
    number = 7


def test_acceptance_catalog_round_trips_through_bounded_image_label_codec() -> None:
    catalog = load_catalog_index(SCRIPTS / "index.yaml")

    label = encode_catalog(catalog)
    decoded = decode_catalog(label)

    assert decoded == catalog
    assert decoded.protocol_version == "1"
    assert set(decoded.scripts) == {"brax_ppo_acceptance"}
    assert re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", label)
    assert len(label.encode("ascii")) <= 4096


def test_brax_ppo_descriptor_matches_the_complete_runtime_contract() -> None:
    descriptor = load_catalog_index(SCRIPTS / "index.yaml").scripts["brax_ppo_acceptance"]

    assert descriptor.model_dump(mode="json") == {
        "name": "brax_ppo_acceptance",
        "argv": ["python", "-m", "brax_ppo_acceptance", "--config", "{config_path}"],
        "sdk_protocol_version": "1",
        "environments": ["inverted_pendulum"],
        "defaults": {
            "environment": {
                "name": "inverted_pendulum",
                "options": {"backend": "generalized"},
            },
            "training_budget": {"env_steps": 128},
            "logging": {
                "aim_every_env_steps": 1,
                "rerun_every_episodes": 1,
            },
        },
        "objective": {
            "metric": "eval/episode_return",
            "direction": "maximize",
            "reduction": "last",
        },
        "fields": {
            "seed": {
                "path": "runtime.seed",
                "type": "int",
                "default": 0,
                "searchable": False,
                "constraints": {"gt": None, "ge": 0.0, "lt": None, "le": None},
                "default_search": None,
                "choices": None,
            },
            "learning_rate": {
                "path": "algorithm.learning_rate",
                "type": "float",
                "default": 0.0003,
                "searchable": True,
                "constraints": {"gt": 0.0, "ge": None, "lt": None, "le": None},
                "default_search": {
                    "values": [0.0002, 0.0003, 0.0004, 0.0005, 0.0006],
                },
                "choices": None,
            },
            "num_envs": {
                "path": "algorithm.num_envs",
                "type": "int",
                "default": 4,
                "searchable": False,
                "constraints": {"gt": None, "ge": None, "lt": None, "le": None},
                "default_search": None,
                "choices": [4],
            },
            "episode_length": {
                "path": "algorithm.episode_length",
                "type": "int",
                "default": 32,
                "searchable": False,
                "constraints": {"gt": None, "ge": None, "lt": None, "le": None},
                "default_search": None,
                "choices": [32],
            },
            "failure_mode": {
                "path": "algorithm.failure_mode",
                "type": "str",
                "default": "none",
                "searchable": False,
                "constraints": {"gt": None, "ge": None, "lt": None, "le": None},
                "default_search": None,
                "choices": ["none"],
            },
        },
    }


def test_descriptor_materializes_into_the_real_acceptance_config(tmp_path: Path) -> None:
    catalog = load_catalog_index(SCRIPTS / "index.yaml")
    image = "repo/acceptance@sha256:" + "a" * 64
    spec = ExperimentSpec(
        experiment=ExperimentIdentity(name="acceptance"),
        defaults=ExperimentDefaults(
            image=image,
            resources=ResourcesSpec(profile="c7am"),
            hpo=HpoSpec(total_trials=1, configs_per_batch=1),
            execution=ExecutionSpec(runs_per_job=1),
        ),
        groups={"cpu": GroupSpec(script="brax_ppo_acceptance")},
    )
    group = resolve_experiment(spec, {image: catalog}).groups[0]

    concrete = materialize_run(
        group,
        FakeTrial(),
        {"learning_rate": 0.0003},
        run_number=1,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(concrete.config_yaml, encoding="utf-8")
    config = AcceptanceConfig.load(config_path, environ={})

    assert config.environment_name == "inverted_pendulum"
    assert config.backend == "generalized"
    assert config.num_timesteps == 128
    assert config.aim_every_env_steps == 1
    assert config.rerun_every_episodes == 1
    assert config.seed == 0
    assert config.learning_rate == 0.0003
    assert config.num_envs == 4
    assert config.episode_length == 32
    assert config.failure_mode == "none"
