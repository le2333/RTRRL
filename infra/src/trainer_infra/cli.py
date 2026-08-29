from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from trainer_infra.batch import REGION, BatchRoundExecutor, batch_target
from trainer_infra.ensemble import BatchEnsembleExecutor, LocalEnsembleExecutor
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
    # Which channel runs the round is the entry's own declaration, read off the
    # image's catalog. An experiment asks for the parallel one by naming an
    # entry that takes a group, and nothing else in the file has a say -- two
    # places to ask would be two places to disagree.
    parallel = {"static": runner.static_parameters} if runner.grouped else {}
    if arguments.backend == "local":
        if arguments.exchange is None or arguments.workspace is None:
            parser.error("local backend requires --exchange and --workspace")
        local = LocalEnsembleExecutor if runner.grouped else LocalRoundExecutor
        return local(
            catalog=arguments.catalog,
            exchange=arguments.exchange,
            workspace=arguments.workspace,
            worker_command=arguments.worker_command,
            **parallel,
        )
    target = batch_target(
        experiment["compute"]["instance_type"],
        arguments.queues,
        runner.digest,
    )
    session = _batch_session()
    batch = BatchEnsembleExecutor if runner.grouped else BatchRoundExecutor
    return batch(
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
        **parallel,
    )


def _claim(
    executor: LocalRoundExecutor | BatchRoundExecutor,
    arguments: argparse.Namespace,
    experiment: dict[str, Any],
    runner: ExperimentRunner,
) -> None:
    """Take the control prefix, before the first round is submitted into it.

    Only the Batch exchange is named after the launch -- a local one is
    wherever ``--exchange`` pointed, shared by every launch that names it --
    and only S3 offers a create two processes cannot both win. So this is a
    Batch-side guarantee, and a local exchange stays the operator's to keep
    distinct.
    """

    if not isinstance(executor, BatchRoundExecutor):
        return
    executor.claim(
        {
            "launch_id": runner.launch_id,
            "experiment": experiment["experiment"],
            "name": experiment["name"],
            "role": runner.role,
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "pid": os.getpid(),
        },
        # A launch id this process invented must name a prefix nobody holds; one
        # the operator passed is their assertion that they meant this one.
        exclusive=arguments.launch_id is None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trainerctl")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    # One launch names one set of runs. Passing it lets a round be asked for
    # again without the artifacts landing somewhere new, and is the only way to
    # write into a control prefix some other launch has already claimed.
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
                    "seeds": list(runner.seeds),
                    "settled": [
                        {
                            "trial": settlement.trial,
                            "value": settlement.value,
                            "seed_values": runner.seed_scores.get(settlement.trial, {}),
                        }
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
    _claim(executor, arguments, experiment, runner)
    # Said the moment the prefix is taken, not once the study finishes. A
    # generated id is no longer the start time, and `settle` requires one, so a
    # controller that dies in its third round has to have already told the
    # operator which launch to settle -- which the final JSON, by then never
    # printed, would not have. stderr because stdout is the machine-readable
    # channel and stays one document.
    print(f"launch {runner.launch_id}", file=sys.stderr, flush=True)
    study = runner.run(executor)
    trials = [
        {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            # What the study stores is the mean over the seeds. A result table
            # reports the seeds, so they are printed beside it rather than
            # recovered afterwards from the artifact tree.
            "seed_values": runner.seed_scores.get(trial.number, {}),
            "parameters": trial.params,
        }
        for trial in study.trials
    ]
    best = study.best_trial
    print(
        json.dumps(
            {
                "study": study.study_name,
                # Repeated here so a captured report is self-contained; the
                # copy that matters for recovery went to stderr before the
                # first round was submitted.
                "launch_id": runner.launch_id,
                "role": runner.role,
                "seeds": list(runner.seeds),
                "evaluation_seed": experiment["evaluation"]["seed"],
                "selection": runner.selection,
                # A trial's parameters name the variable that was drawn, not the
                # paths it was written to, so a report that did not say which
                # those were could not be used to freeze the configuration it
                # found.
                "bindings": [binding.record() for binding in runner.bindings],
                "trials": trials,
                "best": {"number": best.number, "value": best.value},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
