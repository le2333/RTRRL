from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from trainer_infra.batch import REGION, BatchRoundExecutor, batch_target
from trainer_infra.experiment import ExperimentRunner
from trainer_infra.local import LocalRoundExecutor


def _batch_session() -> Any:
    import boto3

    return boto3.Session(region_name=REGION)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trainerctl")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("experiment", type=Path)
    run_parser.add_argument("--backend", choices=("local", "batch"), required=True)
    run_parser.add_argument("--catalog", type=Path, required=True)
    run_parser.add_argument("--database", type=Path, required=True)
    # One launch names one set of runs. Passing it lets a round be asked for
    # again without the artifacts landing somewhere new.
    run_parser.add_argument("--launch-id", default=None)
    run_parser.add_argument("--exchange", type=Path)
    run_parser.add_argument("--workspace", type=Path)
    run_parser.add_argument("--queues", choices=("run", "dev"), default="run")
    run_parser.add_argument("--poll-seconds", type=float, default=20.0)
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
    if arguments.backend == "local":
        if arguments.exchange is None or arguments.workspace is None:
            parser.error("local backend requires --exchange and --workspace")
        executor = LocalRoundExecutor(
            catalog=arguments.catalog,
            exchange=arguments.exchange,
            workspace=arguments.workspace,
            worker_command=arguments.worker_command,
        )
    else:
        target = batch_target(
            experiment["compute"]["instance_type"],
            arguments.queues,
            runner.digest,
        )
        session = _batch_session()
        executor = BatchRoundExecutor(
            s3=session.client("s3"),
            batch=session.client("batch"),
            logs=session.client("logs"),
            exchange=(
                f"{str(experiment['storage']).rstrip('/')}"
                f"/{experiment['experiment']}/{runner.launch_id}/control"
            ),
            job_name=f"{experiment['name']}-{runner.launch_id}",
            job_queue=target.queue,
            job_definition=target.job_definition,
            timeout_seconds=int(experiment["compute"]["timeout_minutes"]) * 60,
            parallel_jobs=int(experiment["hpo"]["parallel_jobs"]),
            poll_seconds=arguments.poll_seconds,
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
