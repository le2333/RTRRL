import json
from pathlib import Path

from training_sdk.contract import CONTRACT_VERSION, Catalog

CATALOG = Path("catalog.json")
RESERVED = frozenset(
    {
        "env",
        "backend",
        "environment",
        "env_mode",
        "env_backend",
        "observed",
        "num_envs",
        "total_steps",
        "epoch_steps",
        "eval_steps",
    }
)


def test_catalog_declares_current_contract_and_only_algorithm_parameters() -> None:
    catalog = Catalog.model_validate(json.loads(CATALOG.read_text()))
    assert catalog.contract == CONTRACT_VERSION
    entry = catalog.entries["brax_ppo_acceptance"]
    taken = RESERVED & set(entry.space)
    assert not taken, f"brax_ppo_acceptance still declares {sorted(taken)}"
    assert set(entry.metrics) >= {"episode_return", "episode_length"}
