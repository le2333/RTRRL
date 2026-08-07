from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import yaml

from trainer_infra.experiment import ExperimentRunner


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="trainerctl")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("experiment", type=Path)
    run_parser.add_argument("--catalog", type=Path, required=True)
    run_parser.add_argument("--database", type=Path, required=True)
    # One launch names one set of runs. Passing it lets a round be asked for
    # again without the artifacts landing somewhere new.
    run_parser.add_argument("--launch-id", default=None)
    arguments = parser.parse_args(argv)

    experiment = yaml.safe_load(arguments.experiment.read_text(encoding="utf-8"))
    catalog = json.loads(arguments.catalog.read_text(encoding="utf-8"))
    configurations = ExperimentRunner(
        experiment=experiment,
        catalog=catalog,
        database=arguments.database,
        launch_id=arguments.launch_id,
    ).next_round()
    print(json.dumps({"configurations": configurations}, indent=2))
