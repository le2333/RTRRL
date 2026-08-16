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


def _add_launch_arguments(subparser: argparse.ArgumentParser) -> None:
    """The arguments that name one launch, whichever way it is being driven."""

    subparser.add_argument("experiment", type=Path)
    subparser.add_argument("--backend", choices=("local", "batch"), required=True)
    subparser.add_argument("--catalog", type=Path, required=True)
    subparser.add_argument("--database", type=Path, required=True)
    subparser.add_argument("--exchange", type=Path)
    subparser.add_argument("--workspace", type=Path)
    subparser.add_argument("--queues", choices=("run", "dev"), default="run")
    subparser.add_argument("--poll-seconds", type=float, default=20.0)
    subparser.add_argument(
        "--worker-command",
        nargs=argparse.REMAINDER,
        default=[sys.executable, "-m", "worker"],
        help="Worker command and arguments; this option must be last",
    )


def _executor(
    arguments: argparse.Namespace,
    parser: argparse.ArgumentParser,
    experiment: dict[str, Any],
    runner: ExperimentRunner,
) -> LocalRoundExecutor | BatchRoundExecutor:
    if arguments.backend == "local":
        if arguments.exchange is None or arguments.workspace is None:
            parser.error("local backend requires --exchange and --workspace")
        return LocalRoundExecutor(
            catalog=arguments.catalog,
            exchange=arguments.exchange,
            workspace=arguments.workspace,
            worker_command=arguments.worker_command,
        )
    target = batch_target(
        experiment["compute"]["instance_type"],
        arguments.queues,
        runner.digest,
    )
    session = _batch_session()
    return BatchRoundExecutor(
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trainerctl")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    # One launch names one set of runs. Passing it lets a round be asked for
    # again without the artifacts landing somewhere new.
    run_parser.add_argument("--launch-id", default=None)
    _add_launch_arguments(run_parser)
    settle_parser = subparsers.add_parser(
        "settle",
        help=(
            "score the finished work of trials a stopped controller left running, "
            "without submitting anything"
        ),
    )
    # Required here, unlike run: the launch id is what names the artifacts
    # this reads, and a fresh one would name a launch that never happened.
    settle_parser.add_argument("--launch-id", required=True)
    _add_launch_arguments(settle_parser)
    arguments = parser.parse_args(argv)

    experiment = yaml.safe_load(arguments.experiment.read_text(encoding="utf-8"))
    catalog = json.loads(arguments.catalog.read_text(encoding="utf-8"))
    runner = ExperimentRunner(
        experiment=experiment,
        catalog=catalog,
        database=arguments.database,
        launch_id=arguments.launch_id,
    )
    executor = _executor(arguments, parser, experiment, runner)
    if arguments.command == "settle":
        settlements = runner.settle(executor.score)
        print(
            json.dumps(
                {
                    "launch_id": runner.launch_id,
                    "settled": [
                        {"trial": settlement.trial, "value": settlement.value}
                        for settlement in settlements
                        if settlement.reason is None
                    ],
                    "still_running": [
                        {"trial": settlement.trial, "reason": settlement.reason}
                        for settlement in settlements
                        if settlement.reason is not None
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
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
