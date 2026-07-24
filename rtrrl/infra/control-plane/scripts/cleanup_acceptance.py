from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any, NamedTuple
import uuid

import boto3

from trainer_infra.facility_control import FacilityControl, load_facility_control


class CleanupRequest(NamedTuple):
    experiment_id: str
    confirm_prefix: str | None
    execute: bool


class CleanupReport(NamedTuple):
    aim_run_hashes: tuple[str, ...]
    expected_prefix: str
    experiment_id: str
    mode: str
    s3_keys: tuple[str, ...]
    writes_performed: bool


class _Snapshot(NamedTuple):
    aim_run_hashes: tuple[str, ...]
    s3_keys: tuple[str, ...]


def canonical_json(report: CleanupReport) -> str:
    return json.dumps(report._asdict(), separators=(",", ":"), sort_keys=True)


def _canonical_experiment_id(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("experiment_id must be a canonical UUID") from error
    if str(parsed) != value:
        raise ValueError("experiment_id must be a canonical UUID")
    return value


def _repo_path(repo: Any) -> Path:
    path = getattr(repo, "path", None)
    if not isinstance(path, (str, Path)):
        raise ValueError("Aim repository does not expose an exact path")
    resolved = Path(path).resolve()
    return resolved.parent if resolved.name == ".aim" else resolved


def _validate_aim_repo(control: FacilityControl, aim_repo: Any) -> None:
    configured = control.aim.repo.resolve()
    main = control.aim.main_repo.resolve()
    if configured == main or _repo_path(aim_repo) == main:
        raise ValueError("main Aim repository is forbidden")
    if _repo_path(aim_repo) != configured:
        raise ValueError("Aim repository does not match facility control")


def _list_s3_keys(s3: Any, control: FacilityControl, experiment_id: str) -> tuple[str, ...]:
    key_prefix = f"{control.prefix}/{experiment_id}/"
    keys: list[str] = []
    continuation: str | None = None
    while True:
        arguments: dict[str, Any] = {
            "Bucket": control.bucket,
            "Prefix": key_prefix,
        }
        if continuation is not None:
            arguments["ContinuationToken"] = continuation
        response = s3.list_objects_v2(**arguments)
        for item in response.get("Contents", []):
            key = item.get("Key") if isinstance(item, Mapping) else None
            if not isinstance(key, str) or not key.startswith(key_prefix):
                raise ValueError("S3 listing returned a key outside the exact prefix")
            keys.append(key)
        if not response.get("IsTruncated"):
            break
        token = response.get("NextContinuationToken")
        if not isinstance(token, str) or not token:
            raise ValueError("truncated S3 listing has no continuation token")
        continuation = token
    if len(keys) != len(set(keys)):
        raise ValueError("S3 listing returned duplicate keys")
    return tuple(sorted(keys))


def _validate_report(
    s3: Any,
    control: FacilityControl,
    experiment_id: str,
    keys: tuple[str, ...],
) -> None:
    report_key = f"{control.prefix}/{experiment_id}/report.json"
    if report_key not in keys:
        raise ValueError("canonical report.json is missing")
    try:
        body = s3.get_object(Bucket=control.bucket, Key=report_key)["Body"].read()
        report = json.loads(body)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("canonical report.json is unreadable") from error
    if not isinstance(report, Mapping):
        raise ValueError("canonical report.json must be an object")
    metadata = report.get("experiment_metadata")
    if (
        report.get("experiment_name") != "infra-brax-ppo-acceptance"
        or not isinstance(metadata, Mapping)
        or metadata.get("purpose") != "infra-acceptance"
    ):
        raise ValueError("canonical report.json metadata does not prove acceptance ownership")


def _list_aim_hashes(aim_repo: Any, experiment_id: str) -> tuple[str, ...]:
    hashes: list[str] = []
    for run in aim_repo.iter_runs():
        context = run.get("context", None)
        if not isinstance(context, Mapping):
            continue
        if context.get("experiment_id") != experiment_id:
            continue
        run_hash = getattr(run, "hash", None)
        if not isinstance(run_hash, str) or not run_hash:
            raise ValueError("matching Aim run has no exact hash")
        hashes.append(run_hash)
    if len(hashes) != len(set(hashes)):
        raise ValueError("Aim enumeration returned duplicate run hashes")
    return tuple(sorted(hashes))


def _snapshot(
    *,
    control: FacilityControl,
    s3: Any,
    aim_repo: Any,
    experiment_id: str,
    validate_report: bool,
) -> _Snapshot:
    s3_keys = _list_s3_keys(s3, control, experiment_id)
    if validate_report:
        _validate_report(s3, control, experiment_id, s3_keys)
    return _Snapshot(
        aim_run_hashes=_list_aim_hashes(aim_repo, experiment_id),
        s3_keys=s3_keys,
    )


def cleanup(
    request: CleanupRequest,
    *,
    control: FacilityControl,
    s3: Any,
    aim_repo: Any,
) -> CleanupReport:
    experiment_id = _canonical_experiment_id(request.experiment_id)
    expected_prefix = (
        f"s3://{control.bucket}/{control.prefix}/{experiment_id}/"
    )
    _validate_aim_repo(control, aim_repo)
    if request.execute and request.confirm_prefix != expected_prefix:
        raise ValueError(f"execute requires confirm_prefix {expected_prefix}")

    planned = _snapshot(
        control=control,
        s3=s3,
        aim_repo=aim_repo,
        experiment_id=experiment_id,
        validate_report=True,
    )
    if not request.execute:
        return CleanupReport(
            aim_run_hashes=planned.aim_run_hashes,
            expected_prefix=expected_prefix,
            experiment_id=experiment_id,
            mode="dry-run",
            s3_keys=planned.s3_keys,
            writes_performed=False,
        )

    confirmed = _snapshot(
        control=control,
        s3=s3,
        aim_repo=aim_repo,
        experiment_id=experiment_id,
        validate_report=True,
    )
    if confirmed != planned:
        raise RuntimeError("cleanup target set changed after confirmation")

    for key in confirmed.s3_keys:
        s3.delete_object(Bucket=control.bucket, Key=key)
    for run_hash in confirmed.aim_run_hashes:
        if aim_repo.delete_run(run_hash) is not True:
            raise RuntimeError(f"Aim refused exact run deletion: {run_hash}")

    remaining = _snapshot(
        control=control,
        s3=s3,
        aim_repo=aim_repo,
        experiment_id=experiment_id,
        validate_report=False,
    )
    if remaining.s3_keys or remaining.aim_run_hashes:
        raise RuntimeError("cleanup postverification found remaining targets")
    return CleanupReport(
        aim_run_hashes=confirmed.aim_run_hashes,
        expected_prefix=expected_prefix,
        experiment_id=experiment_id,
        mode="execute",
        s3_keys=confirmed.s3_keys,
        writes_performed=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run or exactly clean one proven infra acceptance experiment."
    )
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--confirm-prefix")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    from aim import Repo

    arguments = _parser().parse_args(argv)
    control = load_facility_control(arguments.control)
    request = CleanupRequest(
        experiment_id=arguments.experiment_id,
        confirm_prefix=arguments.confirm_prefix,
        execute=arguments.execute,
    )
    session = boto3.Session(region_name=control.region)
    report = cleanup(
        request,
        control=control,
        s3=session.client("s3"),
        aim_repo=Repo(str(control.aim.repo)),
    )
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
