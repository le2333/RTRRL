from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import sys
from typing import Sequence

import boto3

from trainer_infra.heavy_tests import AggregateJobFailure, HeavyTestRunner, TEST_PROFILES


def _print_json(value: object) -> None:
    print(json.dumps(value, sort_keys=True))


def _runner() -> HeavyTestRunner:
    region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "eu-north-1"))
    return HeavyTestRunner(
        boto3.client("batch", region_name=region),
        boto3.client("logs", region_name=region),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit one digest-bound AWS Batch job per heavy pytest file."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser("submit", help="submit one Batch job per test file")
    submit.add_argument("--profile", required=True, choices=tuple(TEST_PROFILES))
    submit.add_argument("--image", required=True, help="immutable IMAGE@sha256:DIGEST")
    submit.add_argument("tests", nargs="+", metavar="TEST_FILE")

    wait = subparsers.add_parser("wait", help="wait for jobs and print evidence")
    wait.add_argument("job_ids", nargs="+", metavar="JOB_ID")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    runner = _runner()
    if args.command == "submit":
        for job in runner.submit(
            profile=args.profile,
            image=args.image,
            tests=args.tests,
        ):
            _print_json(asdict(job))
        return 0

    try:
        evidence = runner.wait(args.job_ids)
    except AggregateJobFailure as error:
        for item in error.evidence:
            _print_json(asdict(item))
        print(str(error), file=sys.stderr)
        return 1
    for item in evidence:
        _print_json(asdict(item))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
