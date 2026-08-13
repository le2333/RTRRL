"""Opt-in acceptance for Infra -> real Worker -> fake Entry -> Infra scoring."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from trainer_infra.experiment import ExperimentRunner
from trainer_infra.local import LocalRoundExecutor

pytestmark = [pytest.mark.integration, pytest.mark.service]

ENTRY = """
import json, os
from pathlib import Path

config = json.loads(Path(os.environ["TRAINER_RUN_CONFIG"]).read_text())
artifacts = Path(os.environ["TRAINER_SCRATCH"]) / "artifacts"
artifacts.mkdir(parents=True)
trial = config["identity"]["trial"]
(artifacts / "metrics.jsonl").write_text(
    json.dumps({"step": 10, "metrics": {"objective": float(trial + 1)}}) + "\\n"
)
"""


def test_real_worker_completes_a_local_hpo_round(tmp_path: Path) -> None:
    entry = tmp_path / "entry.py"
    entry.write_text(ENTRY, encoding="utf-8")
    catalog = {
        "contract": 8,
        "entries": {
            "e": {
                "command": [sys.executable, str(entry)],
                "metrics": ["objective"],
                "parameters": {},
            }
        },
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    experiment = {
        "experiment": "local-acceptance",
        "name": "local-acceptance",
        "image": "local@sha256:" + "a" * 64,
        "entry": "e",
        "storage": (tmp_path / "artifacts").resolve().as_uri(),
        "environment": {
            "id": "fake",
            "backend": "fake",
            "seed": 0,
            "episode_length": 10,
        },
        "training": {"num_envs": 1, "total_steps": 10, "epoch_steps": 10},
        "evaluation": {"steps": 0},
        "logging": {"aim": "unused"},
        "score": {
            "metric": "objective",
            "window_steps": [0, 10],
            "reduce": "last",
            "non_finite": "worst",
            "direction": "maximize",
        },
        "hpo": {"rounds": 1, "trials_per_round": 2, "startup_trials": 2, "seed": 0},
        "space": {},
    }
    runner = ExperimentRunner(
        experiment=experiment,
        catalog=catalog,
        database=tmp_path / "study.db",
        launch_id="launch",
    )
    executor = LocalRoundExecutor(
        catalog=catalog_path,
        exchange=tmp_path / "exchange",
        workspace=tmp_path / "worker",
        worker_command=(sys.executable, "-m", "worker"),
    )

    study = runner.run(executor)

    assert [trial.value for trial in study.trials] == [1.0, 2.0]
    assert study.best_trial.number == 1
    assert (
        json.loads(
            (
                tmp_path
                / "artifacts"
                / "local-acceptance"
                / "launch"
                / "local-acceptance-launch-t1"
                / "result.json"
            ).read_text()
        )["success"]
        is True
    )
