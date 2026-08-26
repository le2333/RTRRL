from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from trainer_infra.scheduler_cli import main


def valid_task_files(tmp_path: Path) -> tuple[str, ...]:
    config = tmp_path / "experiment.yml"
    config.write_text(yaml.safe_dump({"name": "R1-1-Minesweeper-DRQN-LSTM"}))
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}")
    return (
        str(config),
        "--catalog",
        str(catalog),
        "--database",
        str(tmp_path / "study.sqlite"),
    )


def test_add_then_list_reports_human_readable_task(
    tmp_path: Path, capsys: Any
) -> None:
    state = tmp_path / "queue.sqlite"

    assert main(["--state", str(state), "add", *valid_task_files(tmp_path)]) == 0
    capsys.readouterr()
    assert main(["--state", str(state), "list"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output[0]["name"] == "R1-1-Minesweeper-DRQN-LSTM"
    assert output[0]["state"] == "queued"


def test_add_rejects_missing_config(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}")

    assert (
        main(
            [
                "--state",
                str(tmp_path / "queue.sqlite"),
                "add",
                str(tmp_path / "missing.yml"),
                "--catalog",
                str(catalog),
                "--database",
                str(tmp_path / "study.sqlite"),
            ]
        )
        == 2
    )
