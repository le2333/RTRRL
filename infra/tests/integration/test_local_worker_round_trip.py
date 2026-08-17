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
        "contract": 10,
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
            # Two seeds, so what this covers includes the fan-out: one
            # configuration, two runs, two artifact trees, one score.
            "seeds": [0, 1],
            "episode_length": 10,
        },
        "training": {"num_envs": 1, "total_steps": 10, "chunk_steps": 10},
        "evaluation": {
            "every_steps": 10,
            "episodes": 0,
            "chunk_steps": 10,
            "seed": 1000,
        },
        # The fake entry builds no reporter, so this is passed through and
        # never read; it still has to be the shape the worker validates.
        "logging": {"aim": {"url": "unused"}},
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

    # The fake entry reports the trial number, so both seeds of one trial
    # report the same value and the mean is that value.
    assert [trial.value for trial in study.trials] == [1.0, 2.0]
    assert study.best_trial.number == 1
    assert runner.seed_scores == {0: {0: 1.0, 1: 1.0}, 1: {0: 2.0, 1: 2.0}}

    launch = tmp_path / "artifacts" / "local-acceptance" / "launch"
    for seed in (0, 1):
        result = json.loads(
            (launch / f"local-acceptance-launch-t1-s{seed}" / "result.json").read_text()
        )
        assert result["success"] is True
        assert result["identity"]["seed"] == seed
        assert result["identity"]["role"] == "tuning"
