from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
import json
import sys

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from trainer_infra.batch_admin import (
    BatchAdminServices,
    DeploymentError,
    deploy_queues,
    inventory,
)
from trainer_infra.batch_smoke import SmokeServices, run_smoke
from trainer_infra.batch_topology import REGION


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _print_json(value: object, *, error: bool = False) -> None:
    print(
        json.dumps(value, allow_nan=False, default=_json_default, sort_keys=True),
        file=sys.stderr if error else sys.stdout,
    )


def _services() -> BatchAdminServices:
    return BatchAdminServices(
        batch=boto3.client("batch", region_name=REGION),
        sts=boto3.client("sts", region_name=REGION),
    )


def _smoke_services() -> SmokeServices:
    return SmokeServices(
        batch=boto3.client("batch", region_name=REGION),
        logs=boto3.client("logs", region_name=REGION),
        sts=boto3.client("sts", region_name=REGION),
        ecs=boto3.client("ecs", region_name=REGION),
        ec2=boto3.client("ec2", region_name=REGION),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inventory and safely deploy the shared AWS Batch queues."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inventory", help="print complete Batch inventory")
    deploy = subparsers.add_parser(
        "deploy", help="plan queue creation without mutation"
    )
    deploy.add_argument(
        "--execute",
        action="store_true",
        help="create missing queues after exact validation",
    )
    smoke = subparsers.add_parser(
        "smoke", help="plan the fixed eight-job shared-queue smoke matrix"
    )
    smoke.add_argument("--cpu-image", required=True)
    smoke.add_argument("--gpu-image", required=True)
    smoke.add_argument(
        "--execute",
        action="store_true",
        help="submit the fixed matrix and write its evidence report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "smoke":
            result = run_smoke(
                _smoke_services(),
                cpu_image=args.cpu_image,
                gpu_image=args.gpu_image,
                execute=args.execute,
            )
            if args.execute and not result.passed:
                _print_json(
                    {"error": "smoke_failed", "report": asdict(result)},
                    error=True,
                )
                return 1
        else:
            services = _services()
            if args.command == "inventory":
                result = inventory(services)
            else:
                result = deploy_queues(services, execute=args.execute)
    except DeploymentError as error:
        _print_json(
            {
                "error": "deployment_error",
                "message": str(error),
                "report": asdict(error.report),
            },
            error=True,
        )
        return 1
    except (BotoCoreError, ClientError, ValueError, RuntimeError) as error:
        _print_json(
            {"error": type(error).__name__, "message": str(error)}, error=True
        )
        return 1
    _print_json(asdict(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
