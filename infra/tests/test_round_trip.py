"""Infra emits the same serialized shape the image-side fixture specifies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from trainer_infra.experiment import ExperimentError, ExperimentRunner, _absent

CONTRACT = Path(__file__).resolve().parents[2] / "tests" / "contracts" / "v9"
TEMPLATE = Path(__file__).resolve().parents[2] / "experiments" / "streamac template.yaml"


def read_json(name: str) -> dict[str, Any]:
    return json.loads((CONTRACT / name).read_text(encoding="utf-8"))


def test_catalog_fixture_is_the_contract_infra_emits(catalog: Any) -> None:
    assert catalog["contract"] == read_json("catalog.json")["contract"] == 9


def test_a_round_emits_the_serialized_run_spec_shape(
    experiment: Any, catalog: Any, tmp_path: Path
) -> None:
    runs = ExperimentRunner(
        experiment=experiment,
        catalog=catalog,
        database=tmp_path / "study.db",
        launch_id="20260813-120000",
    ).next_round()

    expected = read_json("run.json")
    for run in runs:
        assert set(run) == set(expected)
        for block in ("identity", "artifacts", "algorithm", "training", "evaluation", "logging"):
            assert set(run[block]) == set(expected[block])
        assert set(run["algorithm"]["environment"]) == set(expected["algorithm"]["environment"])
        assert json.loads(json.dumps(run)) == run


def test_score_policy_stays_with_infra(experiment: Any, catalog: Any, tmp_path: Path) -> None:
    run = ExperimentRunner(
        experiment=experiment,
        catalog=catalog,
        database=tmp_path / "study.db",
    ).next_round()[0]

    assert "score" not in run
    assert experiment["score"]["metric"] in catalog["entries"][run["entry"]]["metrics"]


def test_shipped_template_is_complete_for_infra(catalog: Any) -> None:
    template = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))

    assert sorted(_absent(template)) == []
    assert "@sha256:" in template["image"]
    assert template["entry"] in catalog["entries"]


def test_an_empty_file_is_refused_before_catalog_use() -> None:
    with pytest.raises(ExperimentError, match="does not say"):
        ExperimentRunner(experiment={}, catalog={}, database=Path("unused.db"))
