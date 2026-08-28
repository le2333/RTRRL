from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

import optuna
import yaml

from trainer_infra import ExperimentRunner, cli
from trainer_infra.cli import main

LAUNCH = "20260807-120000"

WORKER = """
import json, os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

def path(uri):
    return Path(url2pathname(urlparse(uri).path))

manifest = json.loads(path(os.environ["TRAINER_MANIFEST"]).read_text())
for config_uri in manifest["runs"]:
    config = json.loads(path(config_uri).read_text())
    root = path(config["artifacts"]["root"])
    root.mkdir(parents=True)
    trial = config["identity"]["trial"]
    (root / "metrics.jsonl").write_text(
        json.dumps({"step": 10, "metrics": {"objective": float(trial + 1)}}) + "\\n"
    )
    (root / "result.json").write_text(json.dumps({
        "contract": config["contract"],
        "identity": config["identity"],
        "success": True,
        "artifacts": ["metrics.jsonl"],
    }))
"""


def test_run_command_executes_every_hpo_round(
    tmp_path: Path,
    capsys: Any,
    experiment: dict[str, Any],
    catalog: dict[str, Any],
) -> None:
    experiment["storage"] = (tmp_path / "artifacts").resolve().as_uri()
    experiment["hpo"]["rounds"] = 2
    experiment["score"] = {
        "metric": "objective",
        "window_steps": [0, 10],
        "reduce": "last",
        "non_finite": "worst",
        "direction": "maximize",
    }
    catalog["entries"]["stream_ac"]["metrics"] = ["objective"]
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment), encoding="utf-8")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    worker = tmp_path / "worker.py"
    worker.write_text(WORKER, encoding="utf-8")

    assert (
        main(
            [
                "run",
                str(experiment_path),
                "--backend",
                "local",
                "--catalog",
                str(catalog_path),
                "--database",
                str(tmp_path / "study.db"),
                "--launch-id",
                LAUNCH,
                "--exchange",
                str(tmp_path / "exchange"),
                "--workspace",
                str(tmp_path / "worker-scratch"),
                "--worker-command",
                sys.executable,
                str(worker),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["study"] == experiment["name"]
    assert [trial["number"] for trial in payload["trials"]] == [0, 1, 2, 3]
    assert [trial["value"] for trial in payload["trials"]] == [1.0, 2.0, 3.0, 4.0]
    assert payload["best"] == {"number": 3, "value": 4.0}


def test_settle_command_scores_open_trials_and_leaves_unfinished_ones_alone(
    tmp_path: Path,
    capsys: Any,
    experiment: dict[str, Any],
    catalog: dict[str, Any],
) -> None:
    """Resuming a launch a killed controller left behind, from the terminal.

    Trial 0's worker finished and uploaded; trial 1's has not. The command
    starts nothing either way -- it reads what is there, tells the study about
    trial 0, and says why trial 1 is still open.
    """

    experiment["storage"] = (tmp_path / "artifacts").resolve().as_uri()
    experiment["score"] = {
        "metric": "objective",
        "window_steps": [0, 10],
        "reduce": "last",
        "non_finite": "worst",
        "direction": "maximize",
    }
    catalog["entries"]["stream_ac"]["metrics"] = ["objective"]
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment), encoding="utf-8")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    database = tmp_path / "study.db"

    asked = ExperimentRunner(
        experiment=experiment,
        catalog=catalog,
        database=database,
        launch_id=LAUNCH,
    ).next_round()
    finished = Path(url2pathname(urlparse(asked[0]["artifacts"]["root"]).path))
    finished.mkdir(parents=True)
    (finished / "metrics.jsonl").write_text(
        json.dumps({"step": 10, "metrics": {"objective": 21.0}}) + "\n", encoding="utf-8"
    )
    (finished / "result.json").write_text(
        json.dumps(
            {
                "contract": asked[0]["contract"],
                "identity": asked[0]["identity"],
                "success": True,
                "artifacts": ["metrics.jsonl"],
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "settle",
                str(experiment_path),
                "--backend",
                "local",
                "--catalog",
                str(catalog_path),
                "--database",
                str(database),
                "--launch-id",
                LAUNCH,
                "--exchange",
                str(tmp_path / "exchange"),
                "--workspace",
                str(tmp_path / "worker-scratch"),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["launch_id"] == LAUNCH
    # The seed's own score beside the mean the study was told, because with
    # one seed they are the same number and with ten they are not.
    assert payload["settled"] == [
        {"trial": 0, "value": 21.0, "seed_values": {"0": 21.0}}
    ]
    assert payload["seeds"] == [0]
    assert [entry["trial"] for entry in payload["still_running"]] == [1]
    persisted = optuna.load_study(study_name=experiment["name"], storage=f"sqlite:///{database}")
    assert [trial.state.name for trial in persisted.trials] == ["COMPLETE", "RUNNING"]
    assert not (tmp_path / "exchange").exists(), "settling submitted work"


def test_batch_run_routes_deployment_fields_to_batch_executor(
    tmp_path: Path,
    capsys: Any,
    monkeypatch: Any,
    experiment: dict[str, Any],
    catalog: dict[str, Any],
) -> None:
    experiment["compute"] = {"instance_type": "c7a.large", "timeout_minutes": 90}
    experiment["hpo"]["parallel_jobs"] = 2
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment), encoding="utf-8")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    captured: dict[str, Any] = {}

    class Session:
        def client(self, name: str) -> str:
            return name

    class Executor:
        def __init__(self, **options: Any) -> None:
            captured.update(options)

        def claim(self, launch: dict[str, Any], *, exclusive: bool) -> str:
            captured["claim"] = launch
            captured["exclusive"] = exclusive
            return "s3://claimed"

        def __call__(self, configurations: tuple[dict, ...], score: Any) -> tuple[dict, ...]:
            return tuple(
                {
                    "trial": configuration["identity"]["trial"],
                    "seed": configuration["identity"]["seed"],
                    "value": float(configuration["identity"]["trial"]),
                }
                for configuration in configurations
            )

    monkeypatch.setattr(cli, "_batch_session", Session)
    monkeypatch.setattr(cli, "BatchRoundExecutor", Executor)

    assert (
        main(
            [
                "run",
                str(experiment_path),
                "--backend",
                "batch",
                "--catalog",
                str(catalog_path),
                "--database",
                str(tmp_path / "study.db"),
                "--launch-id",
                LAUNCH,
                "--queues",
                "dev",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["best"] == {"number": 1, "value": 1.0}
    assert captured["s3"] == "s3"
    assert captured["batch"] == "batch"
    assert captured["logs"] == "logs"
    assert captured["job_queue"] == "dev-cpu-c7al-queue"
    assert captured["job_definition"] == "trainer-c7al-" + "b" * 64
    assert captured["timeout_seconds"] == 5400
    # The control prefix is taken before the first round is submitted into it,
    # and an operator-named launch id says the prefix may already be theirs.
    assert captured["claim"]["launch_id"] == LAUNCH
    assert captured["claim"]["experiment"] == experiment["experiment"]
    assert captured["exclusive"] is False
    assert captured["parallel_jobs"] == 2
