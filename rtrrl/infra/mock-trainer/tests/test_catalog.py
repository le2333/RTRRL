import json
from pathlib import Path

from training_sdk.contract import CONTRACT_VERSION, Catalog

CATALOG = Path("catalog.json")


def test_catalog_declares_current_contract_and_the_reserved_parameter() -> None:
    catalog = Catalog.model_validate(json.loads(CATALOG.read_text()))
    assert catalog.contract == CONTRACT_VERSION
    entry = catalog.entries["brax_ppo_acceptance"]
    assert "total_steps" in entry.space
    assert set(entry.metrics) >= {"episode_return", "episode_length"}


def test_source_hash_matches_the_current_sources() -> None:
    from scripts.build_catalog import source_hash

    catalog = Catalog.model_validate(json.loads(CATALOG.read_text()))
    assert catalog.entries["brax_ppo_acceptance"].source_hash == source_hash()


def test_source_hash_ignores_bytecode_caches(tmp_path: Path) -> None:
    """The same sources must hash the same whether or not they have been run.

    Otherwise the image and the checkout it was built from disagree, and the
    control plane reads that as an algorithm change.
    """
    from scripts.build_catalog import source_hash

    (tmp_path / "algorithm.py").write_text("value = 1\n", encoding="utf-8")
    clean = source_hash(tmp_path)

    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "algorithm.cpython-312.pyc").write_bytes(b"\x00compiled")
    (tmp_path / "notes.txt").write_text("not source\n", encoding="utf-8")

    assert source_hash(tmp_path) == clean


def test_source_hash_follows_a_source_edit(tmp_path: Path) -> None:
    from scripts.build_catalog import source_hash

    source = tmp_path / "algorithm.py"
    source.write_text("value = 1\n", encoding="utf-8")
    before = source_hash(tmp_path)
    source.write_text("value = 2\n", encoding="utf-8")

    assert source_hash(tmp_path) != before
