import json
from pathlib import Path

import pytest
import yaml
from training_sdk.contract import CONTRACT_VERSION, Catalog, ChoiceSpec

from trainer_infra.experiment import Experiment, load_experiment
from trainer_infra.preflight import PreflightError, check_offline, format_space
from tests.helpers import CATALOG, EXAMPLE, _document, replace_once


def modified(tmp_path: Path, old: str, new: str) -> Experiment:
    text = replace_once(EXAMPLE.read_text(), old, new)
    path = tmp_path / "experiment.yaml"
    path.write_text(text, encoding="utf-8")
    return load_experiment(path)


def write_catalog(tmp_path: Path, catalog: Catalog = CATALOG) -> Path:
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog.model_dump()), encoding="utf-8")
    return path


def _catalog() -> Catalog:
    return Catalog.model_validate(
        {
            "contract": CONTRACT_VERSION,
            "entries": {
                "demo_entry": {
                    "command": ["run"],
                    "source_hash": "sha256:0",
                    "metrics": ("eval/episode_return",),
                    "space": {"learning_rate": ChoiceSpec(choices=(0.001,))},
                }
            },
        }
    )


def _written(tmp_path, *, window, total_steps):
    document = _document()
    document["score"]["window_steps"] = list(window)
    document["budget"]["total_steps"] = total_steps
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return load_experiment(path)


def test_a_score_window_past_the_budget_is_refused(tmp_path):
    experiment = _written(tmp_path, window=(0, 4000), total_steps=2000)

    with pytest.raises(PreflightError) as raised:
        check_offline(experiment, _catalog())

    assert "4000" in str(raised.value)


def test_a_score_window_inside_the_budget_is_accepted(tmp_path):
    experiment = _written(tmp_path, window=(0, 2000), total_steps=2000)

    assert "learning_rate" in check_offline(experiment, _catalog())


def test_example_passes_offline_checks() -> None:
    space = check_offline(load_experiment(EXAMPLE), CATALOG)
    assert space["total_steps"].choices == (128,)


def test_unknown_entry_is_rejected(tmp_path: Path) -> None:
    experiment = modified(tmp_path, "entry: brax_ppo_acceptance", "entry: missing")
    with pytest.raises(PreflightError, match="missing"):
        check_offline(experiment, CATALOG)


def test_unsupported_contract_is_rejected() -> None:
    catalog = Catalog.model_validate(CATALOG.model_dump() | {"contract": 99})
    with pytest.raises(PreflightError, match="contract"):
        check_offline(load_experiment(EXAMPLE), catalog)


def test_metric_not_reported_by_entry_is_rejected(tmp_path: Path) -> None:
    experiment = modified(tmp_path, "metric: episode_return", "metric: reward")
    with pytest.raises(PreflightError, match="reward"):
        check_offline(experiment, CATALOG)


def test_window_beyond_smallest_total_steps_is_rejected(tmp_path: Path) -> None:
    experiment = modified(tmp_path, "window_steps: [0, 128]", "window_steps: [0, 129]")
    with pytest.raises(PreflightError, match="window"):
        check_offline(experiment, CATALOG)


def test_format_space_lists_every_key() -> None:
    space = check_offline(load_experiment(EXAMPLE), CATALOG)
    text = format_space(space)
    for key in space:
        assert key in text
