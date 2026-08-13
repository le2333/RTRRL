from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

from trainer_infra.experiment import ExperimentRunner
from trainer_infra.local import LocalRoundExecutor


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trainerctl")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("experiment", type=Path)
    run_parser.add_argument("--backend", choices=("local",), required=True)
    run_parser.add_argument("--catalog", type=Path, required=True)
    run_parser.add_argument("--database", type=Path, required=True)
    # One launch names one set of runs. Passing it lets a round be asked for
    # again without the artifacts landing somewhere new.
    run_parser.add_argument("--launch-id", default=None)
    run_parser.add_argument("--exchange", type=Path, required=True)
    run_parser.add_argument("--workspace", type=Path, required=True)
    run_parser.add_argument(
        "--worker-command",
        nargs=argparse.REMAINDER,
        default=[sys.executable, "-m", "worker"],
        help="Worker command and arguments; this option must be last",
    )
    arguments = parser.parse_args(argv)

    experiment = yaml.safe_load(arguments.experiment.read_text(encoding="utf-8"))
    catalog = json.loads(arguments.catalog.read_text(encoding="utf-8"))
    runner = ExperimentRunner(
        experiment=experiment,
        catalog=catalog,
        database=arguments.database,
        launch_id=arguments.launch_id,
    )
    executor = LocalRoundExecutor(
        catalog=arguments.catalog,
        exchange=arguments.exchange,
        workspace=arguments.workspace,
        worker_command=arguments.worker_command,
    )
    study = runner.run(executor)
    trials = [
        {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "parameters": trial.params,
        }
        for trial in study.trials
    ]
    best = study.best_trial
    print(
        json.dumps(
            {
                "study": study.study_name,
                "trials": trials,
                "best": {"number": best.number, "value": best.value},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
