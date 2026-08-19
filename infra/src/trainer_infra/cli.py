from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from trainer_infra.batch import REGION, BatchRoundExecutor, batch_target
from trainer_infra.collapse import CollapseSpec, analyze, decisions
from trainer_infra.experiment import ExperimentRunner
from trainer_infra.fork import (
    branch_documents,
    checkpoint_uri,
    manifest,
    preceding_checkpoint,
)
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


def _run_metrics(values: Sequence[str]) -> list[tuple[str, Path]]:
    """``--run <id>=<metrics path>``, one per seed, kept in the order given."""

    found = []
    for value in values:
        run_id, separator, path = value.partition("=")
        if not separator or not run_id or not path:
            raise SystemExit(f"--run wants <run-id>=<metrics path>, got {value!r}")
        found.append((run_id, Path(path)))
    return found


def _collapse(arguments: argparse.Namespace) -> int:
    """Decide each seed on its own and print every decision, collapse or not.

    Nothing is aggregated here. A collapse is an event in one run with a step
    attached, and the mean of five curves has neither -- so the seeds that
    collapsed, the seeds that diverged and the seeds that never learned are all
    named, and what to conclude from the mixture is the reader's.
    """

    # YAML rather than JSON so the frozen file can say, beside each number,
    # where it came from. Every JSON specification is also a YAML one.
    spec = CollapseSpec.from_mapping(
        yaml.safe_load(arguments.spec.read_text(encoding="utf-8"))
    )
    # The run id is what names a decision, because it is what names the
    # artifact the decision was read out of. Which training seed produced that
    # run is written in the run's own document, and repeating a guess at it
    # here would be a second answer to a question that already has one.
    found = [
        analyze(path, spec, run_id=run_id, window_steps=arguments.window_steps)
        for run_id, path in _run_metrics(arguments.run)
    ]
    document = decisions(found)
    print(json.dumps(document, indent=2, sort_keys=True))
    if arguments.decisions is not None:
        arguments.decisions.write_text(
            json.dumps(document, indent=2, sort_keys=True), encoding="utf-8"
        )
    return 0


def _fork(arguments: argparse.Namespace) -> int:
    """Write the three branch documents and the manifest that names them."""

    parent = json.loads(arguments.parent.read_text(encoding="utf-8"))
    decision = json.loads(arguments.decision.read_text(encoding="utf-8"))
    collapse = decision.get("collapse")
    if collapse is None:
        raise SystemExit(
            f"{decision.get('run_id', 'this run')} is {decision.get('verdict')}: "
            f"{decision.get('reason')}. There is no collapse to branch from."
        )
    boundary = preceding_checkpoint(parent, int(collapse["step"]))
    documents = branch_documents(
        parent,
        checkpoint=checkpoint_uri(parent, boundary),
        from_steps=boundary,
        steps=arguments.steps,
    )
    arguments.into.mkdir(parents=True, exist_ok=True)
    uris = []
    for document in documents:
        path = arguments.into / f"{document['identity']['run_id']}.json"
        path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
        uris.append(path.resolve().as_uri())
    manifest_path = arguments.into / "manifest.json"
    manifest_path.write_text(manifest(uris), encoding="utf-8")
    print(
        json.dumps(
            {
                "parent": parent["identity"]["run_id"],
                "checkpoint": checkpoint_uri(parent, boundary),
                "from_steps": boundary,
                "collapse_step": int(collapse["step"]),
                "manifest": manifest_path.resolve().as_uri(),
                "branches": [document["identity"]["run_id"] for document in documents],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trainerctl")
    subparsers = parser.add_subparsers(dest="command", required=True)
    collapse_parser = subparsers.add_parser(
        "collapse",
        help="decide, per seed, whether a fixed-evaluation curve collapsed",
    )
    collapse_parser.add_argument(
        "--run",
        action="append",
        default=[],
        required=True,
        metavar="ID=PATH",
        help="one seed's run id and its metrics.jsonl; repeat per seed",
    )
    collapse_parser.add_argument("--spec", type=Path, required=True)
    collapse_parser.add_argument("--decisions", type=Path)
    collapse_parser.add_argument(
        "--window-steps",
        type=int,
        default=0,
        help="how far either side of the event the update telemetry is read",
    )
    fork_parser = subparsers.add_parser(
        "fork",
        help="write the three branch documents for one seed's first collapse",
    )
    fork_parser.add_argument("--parent", type=Path, required=True)
    fork_parser.add_argument("--decision", type=Path, required=True)
    fork_parser.add_argument("--into", type=Path, required=True)
    fork_parser.add_argument("--steps", type=int, default=50000)
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

    # Reading the artifacts of finished runs, which starts nothing and needs
    # neither an image nor a study.
    if arguments.command == "collapse":
        return _collapse(arguments)
    if arguments.command == "fork":
        return _fork(arguments)

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
