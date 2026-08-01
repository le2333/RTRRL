"""What the image advertises, checked without the image and without a GPU.

The catalog is the only thing the control plane reads before it spends money: it
decides which parameters may be sampled, which metric may be scored, and what
command a Batch job runs. It is also the thing most easily left stale, since
editing an entry and not rebuilding produces no error anywhere. These tests are
what makes that a failure here instead.
"""

from __future__ import annotations

import json

from training_sdk.contract import CONTRACT_VERSION, Catalog

from entries import rtrrl_aaai
from scripts.build_catalog import CATALOG_PATH

ENTRY_NAME = "rtrrl_aaai"


def catalog() -> Catalog:
    return Catalog.model_validate(json.loads(CATALOG_PATH.read_text(encoding="utf-8")))


def test_the_catalog_declares_this_contract_and_this_entry() -> None:
    declared = catalog()

    assert declared.contract == CONTRACT_VERSION
    entry = declared.entries[ENTRY_NAME]
    assert entry.command == ("python", "-m", "entries.rtrrl_aaai")
    assert "eval/episode/return" in entry.metrics


def test_the_catalog_on_disk_is_the_one_the_entry_would_produce() -> None:
    """A space edited and not rebuilt is a search over a space nobody read."""

    entry = catalog().entries[ENTRY_NAME]

    assert set(entry.space) == set(rtrrl_aaai.SPACE)
    assert entry.metrics == tuple(rtrrl_aaai.METRICS)
