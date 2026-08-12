"""The serialized deployment contract that the refactor starts from."""

from __future__ import annotations

import json
from pathlib import Path

from runner.catalog import build_catalog, write_catalog
from worker.contract import CONTRACT_VERSION, Catalog, RunConfig

FIXTURES = Path(__file__).resolve().parents[4] / "tests" / "contracts" / "v7"


def read_json(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_version_7_catalog_and_run_are_receiver_validated() -> None:
    catalog = Catalog.model_validate(read_json("catalog.json"))
    run = RunConfig.model_validate(read_json("run.json"))

    assert catalog.contract == CONTRACT_VERSION == run.contract == 7
    assert run.entry in catalog.entries


def test_version_7_manifest_names_serialized_run_configurations() -> None:
    manifest = read_json("manifest.json")

    assert manifest == {"runs": ["s3://artifacts/trainer/configs/stream-ac-t0.json"]}


def test_image_build_catalog_matches_the_current_registry(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    write_catalog(path)

    catalog = read_json_from(path)
    discovered = build_catalog().model_dump(mode="json")

    assert catalog["contract"] == CONTRACT_VERSION
    assert set(catalog["entries"]) == {"stream_ac"}
    assert catalog["entries"]["stream_ac"] == discovered["entries"]["stream_ac"]


def read_json_from(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
