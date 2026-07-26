import json
from pathlib import Path

import pytest
from training_sdk.contract import Catalog

from trainer_infra.experiment import Experiment, load_experiment
from trainer_infra.preflight import PreflightError, check_offline, format_space
from tests.helpers import CATALOG, EXAMPLE, replace_once


def modified(tmp_path: Path, old: str, new: str) -> Experiment:
    text = replace_once(EXAMPLE.read_text(), old, new)
    path = tmp_path / "experiment.yaml"
    path.write_text(text, encoding="utf-8")
    return load_experiment(path)


def write_catalog(tmp_path: Path, catalog: Catalog = CATALOG) -> Path:
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog.model_dump()), encoding="utf-8")
    return path


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
