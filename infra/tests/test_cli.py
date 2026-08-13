import json
from pathlib import Path
from typing import Any

from conftest import DIGEST, EXPERIMENT_YAML

from trainer_infra.cli import main

LAUNCH = "20260807-120000"


def test_run_command_emits_first_round_without_metric_feedback(
    tmp_path: Path,
    capsys: Any,
    catalog: Any,
) -> None:
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(EXPERIMENT_YAML, encoding="utf-8")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    main(
        [
            "run",
            str(experiment_path),
            "--catalog",
            str(catalog_path),
            "--database",
            str(tmp_path / "study.db"),
            "--launch-id",
            LAUNCH,
        ]
    )

    configurations = json.loads(capsys.readouterr().out)["configurations"]

    assert [configuration["identity"]["trial"] for configuration in configurations] == [0, 1]
    for configuration in configurations:
        assert configuration["entry"] == "stream_ac"
        assert configuration["identity"]["digest"] == DIGEST
        assert configuration["algorithm"]["environment"]["id"] == "brax::hopper"
        assert configuration["algorithm"]["num_envs"] == 4
        assert "score" not in configuration
        parameters = configuration["algorithm"]["parameters"]
        assert parameters["gamma"] in (0.9, 0.95)
        assert parameters["backbone.rtu.hidden_dim"] == 32
        assert parameters["backbone.kind"] == "rtu"
