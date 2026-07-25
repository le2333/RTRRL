import pytest
from pydantic import ValidationError

from training_sdk.contract import (
    CONTRACT_VERSION,
    Catalog,
    ChoiceSpec,
    EntryDescriptor,
    FloatSpec,
    IntSpec,
    RunConfig,
    ScoreConfig,
)


def test_contract_version_is_two() -> None:
    assert CONTRACT_VERSION == 2


def test_catalog_parses_float_int_and_choice_entries() -> None:
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
                        "batch_size": {
                            "type": "int",
                            "low": 256,
                            "high": 8192,
                            "step": 256,
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
    assert isinstance(entry.space["batch_size"], IntSpec)
    assert entry.space["batch_size"].step == 256


def test_float_spec_rejects_low_greater_than_high() -> None:
    with pytest.raises(ValidationError, match="float low must not exceed high"):
        FloatSpec.model_validate({"type": "float", "low": 2.0, "high": 1.0})


def test_int_spec_rejects_low_greater_than_high() -> None:
    with pytest.raises(ValidationError, match="int low must not exceed high"):
        IntSpec.model_validate({"type": "int", "low": 10, "high": 5})


def test_int_spec_rejects_non_positive_step() -> None:
    with pytest.raises(ValidationError, match="int step must be positive"):
        IntSpec.model_validate({"type": "int", "low": 1, "high": 10, "step": 0})


def test_choice_spec_rejects_empty_list() -> None:
    with pytest.raises(ValidationError, match="choice list must not be empty"):
        ChoiceSpec.model_validate([])


def test_entry_descriptor_rejects_empty_command() -> None:
    with pytest.raises(ValidationError, match="command must not be empty"):
        EntryDescriptor.model_validate(
            {
                "command": [],
                "source_hash": "sha256:0",
                "metrics": ["m"],
                "space": {},
            }
        )


def test_entry_descriptor_rejects_empty_metrics() -> None:
    with pytest.raises(ValidationError, match="metrics must not be empty"):
        EntryDescriptor.model_validate(
            {
                "command": ["run"],
                "source_hash": "sha256:0",
                "metrics": [],
                "space": {},
            }
        )


def test_score_config_rejects_descending_window_steps() -> None:
    with pytest.raises(ValidationError, match="window_steps must be ordered"):
        ScoreConfig.model_validate(
            {
                "metric": "episode_return",
                "window_steps": [128, 0],
                "reduce": "mean",
                "direction": "maximize",
                "non_finite": "worst",
                "s3": "s3://bucket/score.json",
            }
        )


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
