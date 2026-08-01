import json
from pathlib import Path

import pytest
import yaml
from training_sdk.contract import CONTRACT_VERSION, Catalog

from trainer_infra.experiment import Experiment, load_experiment
from trainer_infra.space import flatten
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
                    "metrics": ("eval/episode/return",),
                    "parameters": {
                        "learning_rate": {
                            "kind": "param",
                            "value_type": "float",
                            "valid": {"type": "float", "low": 1e-9, "high": 1.0},
                            "search": [0.001],
                            "placeholder": 0.001,
                        }
                    },
                }
            },
        }
    )


def _written(tmp_path, *, window, total_steps):
    document = _document()
    document["score"]["window_steps"] = list(window)
    document["training"]["total_steps"] = total_steps
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return load_experiment(path)


def test_a_score_window_past_training_is_refused(tmp_path):
    experiment = _written(tmp_path, window=(0, 4000), total_steps=2000)

    with pytest.raises(PreflightError) as raised:
        check_offline(experiment, _catalog())

    assert "4000" in str(raised.value)


def test_a_score_window_inside_training_is_accepted(tmp_path):
    experiment = _written(tmp_path, window=(0, 2000), total_steps=2000)

    resolved = check_offline(experiment, _catalog())
    assert "learning_rate" in flatten(resolved.tree)


def test_example_passes_offline_checks() -> None:
    resolved = check_offline(load_experiment(EXAMPLE), CATALOG)
    assert flatten(resolved.tree)["learning_rate"].placeholder == 0.001
    assert resolved.overrides["learning_rate"].model_dump() == {
        "type": "float",
        "low": 1e-4,
        "high": 1e-3,
        "log": True,
    }


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
    resolved = check_offline(load_experiment(EXAMPLE), CATALOG)
    text = format_space(resolved)
    for key in flatten(resolved.tree):
        assert key in text
