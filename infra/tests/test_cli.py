import json
from pathlib import Path

from trainer_infra.cli import main


def test_run_command_emits_first_round_without_metric_feedback(
    tmp_path: Path,
    capsys: object,
) -> None:
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        """\
name: stream-ac-test
algorithm: stream_ac
hpo:
  rounds: 2
  trials_per_round: 2
  startup_trials: 2
  seed: 7
objective:
  metric: episodic_return
  reduction: last
  direction: maximize
loggers:
  aim:
    endpoint: aim://aim:53800
  local:
    directory: logs
parameters:
  gamma: [0.9, 0.95]
run:
  environment:
    id: Pendulum-v1
    seed: 0
  training:
    total_steps: 10
  debug: false
  overshooting_info: false
  render: false
""",
        encoding="utf-8",
    )
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "version": 1,
                "algorithms": {
                    "stream_ac": {
                        "entry": "algorithms.stream_ac:main",
                        "configure": "algorithms.stream_ac:configure",
                        "parameters": {
                            "lr": {
                                "valid": {"type": "float", "low": 0.0, "high": None},
                                "search": {
                                    "type": "float",
                                    "low": 0.1,
                                    "high": 1.0,
                                },
                            },
                            "gamma": {
                                "valid": {"type": "float", "low": 0.0, "high": 1.0},
                                "search": {
                                    "type": "float",
                                    "low": 0.9,
                                    "high": 0.99,
                                },
                            },
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    main(
        [
            "run",
            str(experiment_path),
            "--catalog",
            str(catalog_path),
            "--database",
            str(tmp_path / "study.db"),
        ]
    )

    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    configurations = payload["configurations"]
    assert len(configurations) == 2
    assert [configuration["trial"] for configuration in configurations] == [0, 1]
    for configuration in configurations:
        assert configuration["algorithm"] == "stream_ac"
        assert configuration["run"]["environment"]["id"] == "Pendulum-v1"
        assert configuration["objective"] == {
            "metric": "episodic_return",
            "reduction": "last",
            "direction": "maximize",
        }
        assert configuration["loggers"] == {
            "aim": {"endpoint": "aim://aim:53800"},
            "local": {"directory": "logs"},
        }
        assert 0.1 <= configuration["parameters"]["lr"] <= 1.0
        assert configuration["parameters"]["gamma"] in [0.9, 0.95]
