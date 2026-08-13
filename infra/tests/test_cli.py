from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

from trainer_infra import cli
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

        def __call__(self, configurations: tuple[dict, ...], score: Any) -> tuple[dict, ...]:
            return tuple(
                {
                    "trial": configuration["identity"]["trial"],
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
    assert captured["parallel_jobs"] == 2
