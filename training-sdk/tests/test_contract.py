import pytest
from pydantic import ValidationError

from training_sdk.contract import (
    CONTRACT_VERSION,
    Catalog,
    ChoiceSpec,
    EntryDescriptor,
    EnvironmentConfig,
    EvaluationConfig,
    FloatSpec,
    IntSpec,
    RunConfig,
    ScoreConfig,
    TrainingConfig,
)


def run_config_kwargs() -> dict:
    return {
        "contract": 6,
        "run_id": "sweep-20260725-051400-t7",
        "experiment": "locomotion",
        "name": "sweep",
        "launch_id": "20260725-051400",
        "trial": 7,
        "entry": "brax_ppo",
        "digest": "registry.example/trainer@sha256:" + "a" * 64,
        "environment": {
            "id": "brax::hopper",
            "backend": "spring",
            "seed": 0,
        },
        "training": {"num_envs": 1, "total_steps": 100, "epoch_steps": 100},
        "evaluation": {"steps": 0, "num_envs": 1},
        "params": {"learning_rate": 0.0003},
        "logging": {
            "aim": "aim://127.0.0.1:53801",
            "enable_rerun": False,
        },
        "score": {
            "metric": "episode_return",
            "window_steps": [0, 128],
            "reduce": "mean",
            "direction": "maximize",
            "non_finite": "worst",
            "s3": "s3://bucket/score.json",
        },
    }


def test_contract_version_is_six() -> None:
    assert CONTRACT_VERSION == 6


def _learning_rate() -> dict:
    return {
        "kind": "param",
        "value_type": "float",
        "valid": {"type": "float", "low": 1e-9, "high": 10.0},
        "search": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "placeholder": 0.001,
    }


def test_catalog_parses_parameter_and_structure_entries() -> None:
    catalog = Catalog.model_validate(
        {
            "contract": 6,
            "entries": {
                "agent": {
                    "command": ["python", "-m", "agent"],
                    "metrics": ["eval/episode/return"],
                    "parameters": {
                        "learning_rate": _learning_rate(),
                        "optimizer_base": {
                            "kind": "structure",
                            "placeholder": "sgd",
                            "branches": {
                                "sgd": {},
                                "adam": {
                                    "b1": {
                                        "kind": "param",
                                        "value_type": "float",
                                        "valid": {
                                            "type": "float",
                                            "low": 0.0,
                                            "high": 1.0,
                                        },
                                        "search": [0.9],
                                        "placeholder": 0.9,
                                    }
                                },
                            },
                        },
                    },
                }
            },
        }
    )

    entry = catalog.entries["agent"]
    assert entry.parameters["learning_rate"].placeholder == 0.001
    assert entry.parameters["learning_rate"].search.log is True
    assert entry.parameters["optimizer_base"].branches["adam"]["b1"].placeholder == 0.9
    assert entry.parameters["optimizer_base"].branches["sgd"] == {}


def test_a_parameter_must_carry_a_search_domain() -> None:
    with pytest.raises(ValidationError):
        EntryDescriptor.model_validate(
            {
                "command": ["run"],
                "metrics": ["m"],
                "parameters": {
                    "reward_trace_reset_on_done": {
                        "kind": "param",
                        "value_type": "bool",
                        "valid": [False, True],
                        "placeholder": True,
                    }
                },
            }
        )


def test_a_valid_domain_may_be_open_on_one_side() -> None:
    entry = EntryDescriptor.model_validate(
        {
            "command": ["run"],
            "metrics": ["m"],
            "parameters": {
                "eta_pi": {
                    "kind": "param",
                    "value_type": "float",
                    "valid": {"type": "float", "low": 0.0},
                    "search": [0.0],
                    "placeholder": 0.0,
                }
            },
        }
    )

    assert entry.parameters["eta_pi"].valid.high is None


def test_entry_descriptor_rejects_the_retired_space_field() -> None:
    with pytest.raises(ValidationError):
        EntryDescriptor.model_validate(
            {"command": ["run"], "metrics": ["m"], "space": {"x": [1]}}
        )


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
                "metrics": ["m"],
                "parameters": {},
            }
        )


def test_entry_descriptor_rejects_empty_metrics() -> None:
    with pytest.raises(ValidationError, match="metrics must not be empty"):
        EntryDescriptor.model_validate(
            {
                "command": ["run"],
                "metrics": [],
                "parameters": {},
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
                "contract": 6,
                "entries": {
                    "e": {
                        "command": ["run"],
                        "metrics": ["m"],
                        "parameters": {"hidden_sizes": [[256, 256]]},
                    }
                },
            }
        )


def test_an_entry_descriptor_has_no_source_hash() -> None:
    """The image digest answers which code ran; a source digest also failed on
    edits that changed no behaviour."""

    with pytest.raises(ValidationError):
        EntryDescriptor(
            command=("train",),
            metrics=("eval/episode/return",),
            parameters={},
            source_hash="sha256:0",
        )


def test_a_run_config_has_no_source_hash() -> None:
    with pytest.raises(ValidationError):
        RunConfig(**(run_config_kwargs() | {"source_hash": "sha256:0"}))


def test_run_config_round_trips() -> None:
    payload = run_config_kwargs()
    config = RunConfig.model_validate(payload)
    assert RunConfig.model_validate(config.model_dump(mode="json")) == config
    assert config.model_dump(mode="json", exclude_none=True) == payload
    assert config.digest == "registry.example/trainer@sha256:" + "a" * 64


def test_environment_carries_seed_but_not_training_streams() -> None:
    environment = EnvironmentConfig(
        id="brax::hopper", backend="spring", seed=7, observed=(0, 1, 2, 3, 4)
    )

    assert environment.seed == 7
    assert environment.observed == (0, 1, 2, 3, 4)
    assert "num_envs" not in environment.model_dump()


def test_an_environment_without_observed_is_fully_observed():
    environment = EnvironmentConfig(id="brax::hopper", backend="spring", seed=0)

    assert environment.observed is None


@pytest.mark.parametrize(
    "observed", [(), (0, 0, 1), (-1, 0)], ids=["empty", "repeated", "negative"]
)
def test_an_index_list_that_selects_nothing_usable_is_refused(observed):
    with pytest.raises(ValidationError):
        EnvironmentConfig(
            id="brax::hopper", backend="spring", seed=0, observed=observed
        )


def test_training_must_divide_into_whole_epochs_and_stream_rounds() -> None:
    with pytest.raises(ValidationError, match="total_steps 1000"):
        TrainingConfig(total_steps=1000, epoch_steps=300, num_envs=1)

    with pytest.raises(ValidationError, match="epoch_steps 1000"):
        TrainingConfig(total_steps=2000, epoch_steps=1000, num_envs=3)


def test_chunk_steps_must_divide_total_and_epoch_when_present() -> None:
    with pytest.raises(ValidationError, match="chunk_steps"):
        TrainingConfig(total_steps=2000, epoch_steps=1000, num_envs=1, chunk_steps=300)

    training = TrainingConfig(
        total_steps=2000, epoch_steps=1000, num_envs=1, chunk_steps=1000
    )
    assert training.chunk_steps == 1000


def test_evaluation_names_rollout_length_and_parallel_streams() -> None:
    evaluation = EvaluationConfig(steps=1000, num_envs=10)

    assert evaluation.steps == 1000
    assert evaluation.num_envs == 10
