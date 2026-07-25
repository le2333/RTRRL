import json
from pathlib import Path

from training_sdk.contract import CONTRACT_VERSION, Catalog

CATALOG = Path("catalog.json")


def test_catalog_declares_contract_two_and_the_reserved_parameter() -> None:
    catalog = Catalog.model_validate(json.loads(CATALOG.read_text()))
    assert catalog.contract == CONTRACT_VERSION
    entry = catalog.entries["brax_ppo_acceptance"]
    assert "total_steps" in entry.space
    assert set(entry.metrics) >= {"episode_return", "episode_length"}


def test_source_hash_matches_the_current_sources() -> None:
    from scripts.build_catalog import source_hash

    catalog = Catalog.model_validate(json.loads(CATALOG.read_text()))
    assert catalog.entries["brax_ppo_acceptance"].source_hash == source_hash()
