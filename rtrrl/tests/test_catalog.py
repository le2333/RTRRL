"""What the image advertises, checked without the image and without a GPU.

The catalog is the only thing the control plane reads before it spends money: it
decides which parameters may be sampled, which metric may be scored, and what
command a Batch job runs. It is also the thing most easily left stale, since
editing an entry and not rebuilding produces no error anywhere. These tests are
what makes that a failure here instead.
"""

from __future__ import annotations

import json
from pathlib import Path

from training_sdk.contract import CONTRACT_VERSION, Catalog

from entries import rtrrl_aaai
from scripts.build_catalog import CATALOG_PATH, source_hash

ENTRY_NAME = "rtrrl_aaai"


def catalog() -> Catalog:
    return Catalog.model_validate(json.loads(CATALOG_PATH.read_text(encoding="utf-8")))


def test_the_catalog_declares_this_contract_and_this_entry() -> None:
    declared = catalog()

    assert declared.contract == CONTRACT_VERSION
    entry = declared.entries[ENTRY_NAME]
    assert entry.command == ("python", "-m", "entries.rtrrl_aaai")
    # The control plane resolves every run's budget through this one name.
    assert "total_steps" in entry.space
    assert "eval/episode_return" in entry.metrics


def test_the_catalog_on_disk_is_the_one_the_entry_would_produce() -> None:
    """A space edited and not rebuilt is a search over a space nobody read."""

    entry = catalog().entries[ENTRY_NAME]

    assert set(entry.space) == set(rtrrl_aaai.SPACE)
    assert entry.metrics == tuple(rtrrl_aaai.METRICS)
    assert entry.source_hash == source_hash()


def test_the_source_hash_ignores_bytecode_caches(tmp_path: Path) -> None:
    """The same sources must hash the same whether or not they have been run.

    Otherwise the image and the checkout it was built from disagree, and the
    control plane reads that as an algorithm change.
    """

    (tmp_path / "algorithm.py").write_text("value = 1\n", encoding="utf-8")
    clean = source_hash(tmp_path)

    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "algorithm.cpython-312.pyc").write_bytes(b"\x00compiled")
    (tmp_path / "notes.txt").write_text("not source\n", encoding="utf-8")

    assert source_hash(tmp_path) == clean


def test_the_source_hash_follows_a_source_edit(tmp_path: Path) -> None:
    source = tmp_path / "algorithm.py"
    source.write_text("value = 1\n", encoding="utf-8")
    before = source_hash(tmp_path)
    source.write_text("value = 2\n", encoding="utf-8")

    assert source_hash(tmp_path) != before


def test_the_source_hash_leaves_out_the_projects_this_image_does_not_carry(
    tmp_path: Path,
) -> None:
    """`infra` holds the control plane, which never ships inside a trainer.

    It also holds a virtual environment with several thousand files in it, so
    walking into it would make the hash both wrong and slow.
    """

    (tmp_path / "algorithm.py").write_text("value = 1\n", encoding="utf-8")
    alone = source_hash(tmp_path)

    (tmp_path / "infra" / "control-plane").mkdir(parents=True)
    (tmp_path / "infra" / "control-plane" / "cli.py").write_text(
        "value = 2\n", encoding="utf-8"
    )

    assert source_hash(tmp_path) == alone
