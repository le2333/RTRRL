import pytest
from pydantic import ValidationError

from training_sdk.contract import (
    CONTRACT_VERSION,
    Catalog,
    ChoiceSpec,
    FloatSpec,
    RunConfig,
)


def test_contract_version_is_two() -> None:
    assert CONTRACT_VERSION == 2


def test_catalog_parses_float_and_choice_entries() -> None:
    catalog = Catalog.model_validate(
        {
            "contract": 2,
            "entries": {
                "brax_ppo": {
                    "command": ["python", "-m", "brax_ppo.train"],
                    "source_hash": "sha256:41b0",
                    "metrics": ["episode_return"],
                    "space": {
                        "total_steps": [128],
                        "learning_rate": {
                            "type": "float",
                            "low": 1e-6,
                            "high": 1e-2,
                            "log": True,
                        },
                    },
                }
            },
        }
    )
    entry = catalog.entries["brax_ppo"]
    assert isinstance(entry.space["total_steps"], ChoiceSpec)
    assert entry.space["total_steps"].choices == (128,)
    assert isinstance(entry.space["learning_rate"], FloatSpec)
    assert entry.space["learning_rate"].log is True


def test_choice_spec_rejects_non_scalar_choices() -> None:
    with pytest.raises(ValidationError):
        Catalog.model_validate(
            {
                "contract": 2,
                "entries": {
                    "e": {
                        "command": ["run"],
                        "source_hash": "sha256:0",
                        "metrics": ["m"],
                        "space": {"hidden_sizes": [[256, 256]]},
                    }
                },
            }
        )


def test_run_config_round_trips() -> None:
    payload = {
        "contract": 2,
        "run_id": "sweep-20260725-051400-t7",
        "experiment": "locomotion",
        "name": "sweep",
        "launch_id": "20260725-051400",
        "trial": 7,
        "entry": "brax_ppo",
        "params": {"total_steps": 128, "learning_rate": 0.0003},
        "logging": {"aim": "aim://127.0.0.1:53801", "every_steps": 1},
        "score": {
            "metric": "episode_return",
            "window_steps": [0, 128],
            "reduce": "mean",
            "direction": "maximize",
            "non_finite": "worst",
            "s3": "s3://bucket/score.json",
        },
    }
    config = RunConfig.model_validate(payload)
    assert RunConfig.model_validate(config.model_dump(mode="json")) == config
    assert config.model_dump(mode="json", exclude_none=True) == payload
