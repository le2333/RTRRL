from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any
from urllib.request import urlopen

from training_sdk.contract import Catalog

from trainer_infra.backends.batch import BatchBackend
from trainer_infra.backends.local import LocalBackend
from trainer_infra.experiment import load_experiment
from trainer_infra.launch import create_launch
from trainer_infra.loop import LaunchFailed, run_launch
from trainer_infra.preflight import LaunchPlan, PreflightError, check_aws, check_offline, format_space
from trainer_infra.queues import REGION
from trainer_infra.space import SpaceError


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def _read_ecr_url(url: str) -> bytes:
    with urlopen(url) as response:  # noqa: S310 - URL is issued by ECR.
        return response.read()


def _batch_session_factory() -> Any:
    """Talk to the region the queues are in, whatever the environment says.

    The queues, job definitions and ECR repository all live in one region, named by
    `trainer_infra.queues`. Taking the region from the environment instead would
    fail when nothing sets it, and — worse — quietly address a different region when
    something sets it to one.
    """
    import boto3

    return boto3.Session(region_name=REGION)


def _warn_dev_queues(tier: str) -> None:
    if tier == "dev":
        print(
            "warning: dev queues are for infrastructure development only; "
            "delivered runs use run queues.",
            file=sys.stderr,
        )


def _batch_preflight(
    experiment_path: Path,
    tier: str,
    *,
    session_factory: Callable[[], Any] | None = None,
    read_url: Callable[[str], bytes] | None = None,
) -> tuple[Any, dict[str, Any]]:
    experiment = load_experiment(experiment_path)
    factory = _batch_session_factory if session_factory is None else session_factory
    url_reader = _read_ecr_url if read_url is None else read_url
    session = factory()
    ecr = session.client("ecr")
    batch = session.client("batch")
    s3 = session.client("s3")
    from trainer_infra.images import resolve_image

    resolved = resolve_image(experiment.image, ecr, url_reader)
    space = check_offline(experiment, resolved.catalog)
    plan = check_aws(
        experiment,
        resolved.catalog,
        space,
        ecr_client=ecr,
        batch_client=batch,
        s3_client=s3,
        read_url=url_reader,
        tier=tier,
    )
    return plan, space


def validate_command(experiment_path: Path, catalog_path: Path) -> int:
    experiment = load_experiment(experiment_path)
    catalog = Catalog.model_validate(json.loads(catalog_path.read_text(encoding="utf-8")))
    try:
        space = check_offline(experiment, catalog)
    except (PreflightError, SpaceError, ValueError) as error:
        print(f"preflight failed: {error}", file=sys.stderr)
        return 1
    print(format_space(space))
    return 0


def validate_batch_command(
    experiment_path: Path,
    tier: str,
    *,
    session_factory: Callable[[], Any] | None = None,
    read_url: Callable[[str], bytes] | None = None,
) -> int:
    _warn_dev_queues(tier)
    try:
        _plan, space = _batch_preflight(
            experiment_path,
            tier,
            session_factory=session_factory,
            read_url=read_url,
        )
    except (PreflightError, SpaceError, ValueError) as error:
        print(f"preflight failed: {error}", file=sys.stderr)
        return 1
    print(format_space(space))
    return 0


def run_batch_command(
    experiment_path: Path,
    archive_dir: Path,
    tier: str,
    *,
    session_factory: Callable[[], Any] | None = None,
    read_url: Callable[[str], bytes] | None = None,
    poll_seconds: float = 20.0,
) -> int:
    _warn_dev_queues(tier)
    try:
        plan, _space = _batch_preflight(
            experiment_path,
            tier,
            session_factory=session_factory,
            read_url=read_url,
        )
    except (PreflightError, SpaceError, ValueError) as error:
        print(f"preflight failed: {error}", file=sys.stderr)
        return 1

    session = (_batch_session_factory if session_factory is None else session_factory)()
    batch = session.client("batch")
    logs = session.client("logs")
    backend = BatchBackend(batch, logs, poll_seconds=poll_seconds)
    launch = create_launch(plan, archive_dir, experiment_path, datetime.now(UTC))

    try:
        report = run_launch(
            launch,
            backend,
            printer=lambda line: print(line, file=sys.stderr),
        )
    except LaunchFailed as error:
        print(str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("trainerctl: interrupted", file=sys.stderr)
        return 130

    print(json.dumps(report.payload(), sort_keys=True, indent=2))
    return 0


def run_local_command(
    experiment_path: Path,
    catalog_path: Path,
    archive_dir: Path,
    jobs_dir: Path,
) -> int:
    experiment = load_experiment(experiment_path)
    catalog = Catalog.model_validate(json.loads(catalog_path.read_text(encoding="utf-8")))
    try:
        space = check_offline(experiment, catalog)
    except (PreflightError, SpaceError, ValueError) as error:
        print(f"preflight failed: {error}", file=sys.stderr)
        return 1
    entry = catalog.entries[experiment.entry]
    plan = LaunchPlan(
        experiment=experiment,
        entry_name=experiment.entry,
        entry=entry,
        parameters=space,
        digest="local",
        queue="local",
        job_definition="local",
    )
    launch = create_launch(plan, archive_dir, experiment_path, datetime.now(UTC))
    backend = LocalBackend(jobs_dir, catalog_path)
    try:
        # Progress goes to stderr so stdout carries nothing but the report,
        # which is what makes `trainerctl run ... > report.json` usable.
        report = run_launch(launch, backend, printer=lambda line: print(line, file=sys.stderr))
    except LaunchFailed as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(report.payload(), sort_keys=True, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="trainerctl")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="check an experiment file")
    validate.add_argument("experiment", type=Path)
    validate.add_argument("--catalog", type=Path)
    validate.add_argument("--backend", choices=("batch",))
    validate.add_argument(
        "--queues",
        choices=("run", "dev"),
        default="run",
        help=(
            "dev queues are for infrastructure development only; "
            "delivered runs use run queues"
        ),
    )
    run = subparsers.add_parser("run")
    run.add_argument("experiment", type=Path)
    run.add_argument("--catalog", type=Path)
    run.add_argument("--archive-dir", type=Path, default=Path("archive"))
    run.add_argument("--jobs-dir", type=Path, default=Path("jobs"))
    run.add_argument("--backend", choices=("local", "batch"))
    run.add_argument(
        "--queues",
        choices=("run", "dev"),
        default="run",
        help=(
            "dev queues are for infrastructure development only; "
            "delivered runs use run queues"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except ValueError as error:
        print(f"trainerctl: error: {error}", file=sys.stderr)
        return 2

    if args.command == "validate":
        has_catalog = args.catalog is not None
        has_batch = args.backend == "batch"
        if has_catalog == has_batch:
            print(
                "trainerctl: error: validate requires exactly one of "
                "--catalog or --backend batch",
                file=sys.stderr,
            )
            return 2
        if has_catalog:
            return validate_command(args.experiment, args.catalog)
        return validate_batch_command(args.experiment, args.queues)

    if args.backend is None:
        print(
            "trainerctl: error: run requires --backend local or --backend batch",
            file=sys.stderr,
        )
        return 2

    if args.backend == "local":
        if args.catalog is None:
            print("trainerctl: error: --catalog is required for --backend local", file=sys.stderr)
            return 2
        return run_local_command(args.experiment, args.catalog, args.archive_dir, args.jobs_dir)

    return run_batch_command(args.experiment, args.archive_dir, args.queues)
