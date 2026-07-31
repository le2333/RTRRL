import optuna
import pytest
from training_sdk.contract import ChoiceSpec, EntryDescriptor

from trainer_infra.space import (
    SpaceError,
    distributions,
    resolve_space,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_entry(space: dict) -> EntryDescriptor:
    return EntryDescriptor.model_validate(
        {
            "command": ["run"],
            "metrics": ["episode_return"],
            "space": space,
        }
    )


def test_override_replaces_entry_by_key() -> None:
    entry = make_entry(
        {
            "total_steps": {"type": "int", "low": 1, "high": 1000},
            "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-2},
        }
    )
    resolved = resolve_space(entry, {"total_steps": ChoiceSpec.model_validate([128])})
    assert resolved["total_steps"].choices == (128,)
    assert resolved["learning_rate"].high == 1e-2


def test_unknown_override_key_is_rejected() -> None:
    entry = make_entry({"total_steps": [128]})
    with pytest.raises(SpaceError, match="learnign_rate"):
        resolve_space(entry, {"learnign_rate": ChoiceSpec.model_validate([0.1])})


def test_distributions_cover_every_key() -> None:
    entry = make_entry(
        {
            "total_steps": [128],
            "env": ["walker2d", "ant"],
            "num_envs": {"type": "int", "low": 256, "high": 1024, "step": 256},
            "warmup_steps": {"type": "int", "low": 100, "high": 10000, "log": True},
            "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
        }
    )
    built = distributions(resolve_space(entry, {}))
    assert set(built) == {
        "total_steps",
        "env",
        "num_envs",
        "warmup_steps",
        "learning_rate",
    }
    assert isinstance(built["total_steps"], optuna.distributions.CategoricalDistribution)
    assert isinstance(built["num_envs"], optuna.distributions.IntDistribution)
    assert built["num_envs"].step == 256
    assert built["num_envs"].log is False
    assert isinstance(built["warmup_steps"], optuna.distributions.IntDistribution)
    assert built["warmup_steps"].log is True
    assert isinstance(built["learning_rate"], optuna.distributions.FloatDistribution)
    assert built["learning_rate"].log is True


def test_optuna_can_sample_every_built_distribution() -> None:
    entry = make_entry(
        {
            "total_steps": [128],
            "env": ["walker2d", "ant"],
            "num_envs": {"type": "int", "low": 256, "high": 1024, "step": 256},
            "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
        }
    )
    built = distributions(resolve_space(entry, {}))
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    trial = study.ask(built)
    assert trial.params["total_steps"] == 128
    assert trial.params["env"] in {"walker2d", "ant"}
    assert 256 <= trial.params["num_envs"] <= 1024
    assert trial.params["num_envs"] % 256 == 0
    assert 1e-6 <= trial.params["learning_rate"] <= 1e-2


def test_unknown_override_key_lists_declared_keys() -> None:
    entry = make_entry({"total_steps": [128], "learning_rate": [0.1]})
    with pytest.raises(
        SpaceError,
        match="experiment declares parameters the entry does not accept: learnign_rate; "
        "entry declares: learning_rate, total_steps",
    ):
        resolve_space(entry, {"learnign_rate": ChoiceSpec.model_validate([0.1])})


def test_distributions_rejects_unsupported_space_entry() -> None:
    class UnsupportedSpec:
        pass

    with pytest.raises(SpaceError, match="unsupported space entry for rogue"):
        distributions({"rogue": UnsupportedSpec()})  # type: ignore[dict-item]
