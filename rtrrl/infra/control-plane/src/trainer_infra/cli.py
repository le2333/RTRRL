from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, PositiveFloat
import yaml
from training_sdk.contract import Catalog

from trainer_infra.adapters.aws_batch import (
    AwsBatchAdapter,
    AwsBatchPreflight,
    AwsBatchPreflightContract,
    JobDefinitionExpectation,
)
from trainer_infra.adapters.s3 import S3ObjectStore
from trainer_infra.aim_reader import AimReader
from trainer_infra.controller import ExperimentController, ExperimentRunError
from trainer_infra.ecr import BotoEcrCatalogReader
from trainer_infra.experiment import load_experiment
from trainer_infra.identities import canonical_json
from trainer_infra.preflight import PreflightError, check_offline, format_space
from trainer_infra.space import SpaceError


class ControlConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    region: str
    account_id: str
    bucket: str
    prefix: Literal["experiments"] = "experiments"
    aim_repo: str
    poll_interval_seconds: PositiveFloat = 5
    batch_timeout_seconds: PositiveFloat = 3600
    subnets: tuple[str, ...]
    security_group_ids: tuple[str, ...]
    instance_role: str
    job_definitions: dict[str, JobDefinitionExpectation]


def load_control(path: Path) -> ControlConfig:
    with Path(path).open(encoding="utf-8") as stream:
        return ControlConfig.model_validate(yaml.safe_load(stream))


class _ReadOnlyPreflight:
    def __init__(
        self,
        *,
        batch: Any,
        s3: Any,
        sts: Any,
        config: ControlConfig,
        aim_probe: Callable[[str], None],
    ) -> None:
        self._batch = batch
        self._s3 = s3
        self._sts = sts
        self._config = config
        self._aim_probe = aim_probe

    def validate(self, resolved: object) -> dict[str, Any]:
        del resolved
        self._s3.head_bucket(Bucket=self._config.bucket)
        identity = self._sts.get_caller_identity()
        if identity.get("Account") != self._config.account_id:
            raise ValueError("AWS account does not match control configuration")
        self._aim_probe(self._config.aim_repo)
        contract = AwsBatchPreflightContract(
            subnets=self._config.subnets,
            security_group_ids=self._config.security_group_ids,
            instance_role=self._config.instance_role,
            job_definitions=self._config.job_definitions,
        )
        definitions = AwsBatchPreflight(self._batch).validate(contract)
        return {item.resource_profile: item for item in definitions}


def _probe_aim(repo_path: str) -> None:
    from aim.sdk.repo import Repo, RepoStatus

    status = Repo.check_repo_status(repo_path)
    if status == RepoStatus.MISSING:
        raise ValueError(f"Aim repository does not exist: {repo_path!r}")
    if status == RepoStatus.UPDATE_REQUIRED:
        raise ValueError(f"Aim repository requires an update: {repo_path!r}")
    Repo.from_path(repo_path)


class _S3SpoolReplayer:
    def __init__(self, s3: Any, bucket: str, aim_repo: str) -> None:
        self._s3 = s3
        self._bucket = bucket
        self._aim_repo = aim_repo

    def __call__(self, run_id: str) -> None:
        from training_sdk import EventSpool, RunContext
        from training_sdk.aim_adapter import AimAdapter

        experiment_id, group, _ = run_id.rsplit(":", 2)
        root = f"experiments/{experiment_id}/groups/{group}/runs/{run_id}/"
        context_data = self._s3.get_object(
            Bucket=self._bucket,
            Key=f"{root}input/run-context.json",
        )["Body"].read()
        spool_data = self._s3.get_object(
            Bucket=self._bucket,
            Key=f"{root}aim-buffer/events.jsonl",
        )["Body"].read()
        context_payload = json.loads(context_data)
        with tempfile.TemporaryDirectory(prefix="trainer-aim-replay-") as temporary:
            spool_path = Path(temporary) / "events.jsonl"
            spool_path.write_bytes(spool_data)
            adapter = AimAdapter(repo=self._aim_repo)
            try:
                adapter.start(RunContext(**context_payload))
                EventSpool(spool_path).replay(adapter)
            finally:
                adapter.close()


def build_controller(control_path: Path) -> ExperimentController:
    import boto3

    config = load_control(control_path)
    session = boto3.Session(region_name=config.region)
    ecr = session.client("ecr")
    s3 = session.client("s3")
    batch = session.client("batch")
    sts = session.client("sts")
    catalog_reader = BotoEcrCatalogReader(
        ecr,
        account_id=config.account_id,
        region=config.region,
    )
    preflight = _ReadOnlyPreflight(
        batch=batch,
        s3=s3,
        sts=sts,
        config=config,
        aim_probe=_probe_aim,
    )

    return ExperimentController(
        catalog_reader=catalog_reader,
        preflight=preflight,
        store_factory=lambda experiment_prefix: S3ObjectStore(s3, experiment_prefix),
        batch_factory=lambda experiment_prefix, _store: AwsBatchAdapter(
            batch, experiment_prefix
        ),
        aim_reader=AimReader(
            config.aim_repo,
            replay_spool=_S3SpoolReplayer(s3, config.bucket, config.aim_repo),
            poll_interval=float(config.poll_interval_seconds),
        ),
        bucket=config.bucket,
        prefix=config.prefix,
        poll_interval=float(config.poll_interval_seconds),
        batch_timeout=float(config.batch_timeout_seconds),
    )


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def validate_command(experiment_path: Path, catalog_path: Path) -> int:
    experiment = load_experiment(experiment_path)
    catalog = Catalog.model_validate(json.loads(catalog_path.read_text(encoding="utf-8")))
    try:
        space = check_offline(experiment, catalog)
    except (PreflightError, SpaceError) as error:
        print(f"preflight failed: {error}", file=sys.stderr)
        return 1
    print(format_space(space))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="trainerctl")
    parser.add_argument(
        "--control",
        type=Path,
        default=Path(os.environ.get("TRAINER_CONTROL_CONFIG", "config/control.yaml")),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="check an experiment file")
    validate.add_argument("experiment", type=Path)
    validate.add_argument("--catalog", type=Path)
    run = subparsers.add_parser("run")
    run.add_argument("experiment", type=Path)
    return parser


def main(
    argv: Sequence[str] | None = None,
    controller_factory: Callable[[Path], ExperimentController] = build_controller,
) -> int:
    try:
        args = _parser().parse_args(argv)
    except ValueError as error:
        print(f"trainerctl: error: {error}", file=sys.stderr)
        return 2

    if args.command == "validate" and args.catalog is not None:
        return validate_command(args.experiment, args.catalog)

    try:
        controller = controller_factory(args.control)
        report = getattr(controller, args.command)(args.experiment)
        print(report.to_json())
        return 0
    except ExperimentRunError as error:
        cause = error.original_cause
        error_payload = (
            {
                "message": str(cause),
                "type": type(cause).__name__,
            }
            if cause is not None
            else {
                "message": "final experiment persistence failed",
                "type": "PersistenceError",
            }
        )
        payload = {
            "status": "failed",
            "error": error_payload,
            "report": error.report.model_dump(mode="json"),
            "submitted_job_ids": list(error.report.submitted_job_ids),
            "persistence_errors": [
                {
                    "message": str(persistence_error),
                    "type": type(persistence_error).__name__,
                }
                for persistence_error in error.persistence_errors
            ],
        }
        print(canonical_json(payload), file=sys.stderr)
        return 1
    except BaseException as error:
        payload = {
            "error": {
                "message": str(error),
                "type": type(error).__name__,
            }
        }
        print(canonical_json(payload), file=sys.stderr)
        return 1
