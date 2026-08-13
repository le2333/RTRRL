from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

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
