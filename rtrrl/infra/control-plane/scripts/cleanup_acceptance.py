from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
import sys
import tempfile
from typing import Any, NamedTuple
import uuid

import boto3

from trainer_infra.facility_control import FacilityControl, load_facility_control


# Audited constants for this test acceptance facility only. Control data cannot
# authorize a different local repository.
ACCEPTANCE_AIM_SCRATCH = Path("/home/ubuntu/trainer/task7-aim-scratch")
ACCEPTANCE_MAIN_REPO = Path("/home/ubuntu/trainer/streaming-rtrrl")
_DELETE_BATCH_SIZE = 1000


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


RepoFactory = Callable[..., Any]


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


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first in second.parents
        or second in first.parents
    )


def _has_dot_aim_ancestor(path: Path) -> bool:
    return any(parent.name == ".aim" for parent in (path, *path.parents))


def _validate_control_aim_paths(control: FacilityControl) -> Path:
    scratch = Path(control.aim.repo)
    main = Path(control.aim.main_repo)
    if scratch != ACCEPTANCE_AIM_SCRATCH or main != ACCEPTANCE_MAIN_REPO:
        raise ValueError("Aim paths must match the audited acceptance-only constants")

    resolved_scratch = scratch.resolve()
    resolved_main = main.resolve()
    lexical_overlap = _paths_overlap(scratch, main)
    resolved_overlap = _paths_overlap(resolved_scratch, resolved_main)
    dot_aim_overlap = (
        _has_dot_aim_ancestor(scratch)
        or _has_dot_aim_ancestor(main)
        or _has_dot_aim_ancestor(resolved_scratch)
        or _has_dot_aim_ancestor(resolved_main)
        or _paths_overlap(resolved_scratch / ".aim", resolved_main / ".aim")
    )
    if lexical_overlap or resolved_overlap or dot_aim_overlap:
        raise ValueError("Aim scratch and main repository paths overlap")
    return scratch


def _default_repo_factory(path: str, *, read_only: bool, init: bool) -> Any:
    from aim import Repo
    from aim.sdk.repo import RepoStatus

    if init:
        raise ValueError("acceptance cleanup never initializes an Aim repository")
    if Repo.check_repo_status(path) is not RepoStatus.UPDATED:
        raise ValueError("Aim repository must already be updated before cleanup")
    try:
        return Repo(path, read_only=read_only, init=False)
    except NotImplementedError:
        if not read_only:
            return Repo(path, init=False)

        temporary = tempfile.TemporaryDirectory(prefix="acceptance-aim-readonly-")
        mirror_root = Path(temporary.name) / "repo"
        mirror_root.mkdir()
        shutil.copytree(Path(path) / ".aim", mirror_root / ".aim", symlinks=True)
        mirror = Repo(str(mirror_root), init=False)
        mirror.read_only = True

        class ReadOnlyAcceptanceRepo:
            def iter_runs(self) -> Any:
                return mirror.iter_runs()

            def close(self) -> None:
                mirror.close()
                temporary.cleanup()

        return ReadOnlyAcceptanceRepo()


def _list_s3_keys(s3: Any, control: FacilityControl, experiment_id: str) -> tuple[str, ...]:
    key_prefix = f"{control.prefix}/{experiment_id}/"
    keys: list[str] = []
    continuation: str | None = None
    seen_tokens: set[str] = set()
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
        if token in seen_tokens:
            raise ValueError("S3 listing repeated a continuation token")
        seen_tokens.add(token)
        continuation = token
    if len(keys) != len(set(keys)):
        raise ValueError("S3 listing returned duplicate keys")
    return tuple(sorted(keys))


def _validate_report(
    s3: Any,
    control: FacilityControl,
    experiment_id: str,
    keys: tuple[str, ...],
) -> str:
    report_key = f"{control.prefix}/{experiment_id}/report.json"
    if report_key not in keys:
        raise ValueError("canonical report.json is missing")
    try:
        body = s3.get_object(Bucket=control.bucket, Key=report_key)["Body"].read()
        report = json.loads(body)
    except (KeyError, TypeError, ValueError) as error:
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
    return report_key


def _list_aim_hashes(repo: Any, experiment_id: str) -> tuple[str, ...]:
    hashes: list[str] = []
    for run in repo.iter_runs():
        context = run.get("context", None)
        if not isinstance(context, Mapping) or context.get("experiment_id") != experiment_id:
            continue
        run_hash = getattr(run, "hash", None)
        if not isinstance(run_hash, str) or not run_hash:
            raise ValueError("matching Aim run has no exact hash")
        hashes.append(run_hash)
    if len(hashes) != len(set(hashes)):
        raise ValueError("Aim enumeration returned duplicate run hashes")
    return tuple(sorted(hashes))


def _read_aim_hashes(
    repo_factory: RepoFactory,
    repo_path: Path,
    experiment_id: str,
) -> tuple[str, ...]:
    repo = repo_factory(str(repo_path), read_only=True, init=False)
    try:
        return _list_aim_hashes(repo, experiment_id)
    finally:
        repo.close()


def _snapshot(
    *,
    control: FacilityControl,
    s3: Any,
    repo_factory: RepoFactory,
    repo_path: Path,
    experiment_id: str,
    validate_report: bool,
) -> _Snapshot:
    s3_keys = _list_s3_keys(s3, control, experiment_id)
    if validate_report:
        _validate_report(s3, control, experiment_id, s3_keys)
    return _Snapshot(
        aim_run_hashes=_read_aim_hashes(repo_factory, repo_path, experiment_id),
        s3_keys=s3_keys,
    )


def _delete_s3_keys(s3: Any, control: FacilityControl, keys: tuple[str, ...]) -> None:
    for offset in range(0, len(keys), _DELETE_BATCH_SIZE):
        batch = keys[offset : offset + _DELETE_BATCH_SIZE]
        response = s3.delete_objects(
            Bucket=control.bucket,
            Delete={"Objects": [{"Key": key} for key in batch], "Quiet": False},
        )
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        errors = response.get("Errors", [])
        deleted = response.get("Deleted", [])
        deleted_keys = {
            item.get("Key") for item in deleted if isinstance(item, Mapping)
        }
        if status != 200 or errors or deleted_keys != set(batch):
            raise RuntimeError(
                f"S3 deletion failed for exact batch: status={status!r}, errors={errors!r}"
            )


def _delete_aim_hashes(
    repo_factory: RepoFactory,
    repo_path: Path,
    hashes: tuple[str, ...],
) -> None:
    if not hashes:
        return
    repo = repo_factory(str(repo_path), read_only=False, init=False)
    try:
        for run_hash in hashes:
            if repo.delete_run(run_hash) is not True:
                raise RuntimeError(f"Aim refused exact run deletion: {run_hash}")
    finally:
        repo.close()


def cleanup(
    request: CleanupRequest,
    *,
    control: FacilityControl,
    s3: Any,
    repo_factory: RepoFactory = _default_repo_factory,
) -> CleanupReport:
    experiment_id = _canonical_experiment_id(request.experiment_id)
    repo_path = _validate_control_aim_paths(control)
    expected_prefix = f"s3://{control.bucket}/{control.prefix}/{experiment_id}/"
    if request.execute and request.confirm_prefix != expected_prefix:
        raise ValueError(f"execute requires confirm_prefix {expected_prefix}")

    planned = _snapshot(
        control=control,
        s3=s3,
        repo_factory=repo_factory,
        repo_path=repo_path,
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
        repo_factory=repo_factory,
        repo_path=repo_path,
        experiment_id=experiment_id,
        validate_report=True,
    )
    if confirmed != planned:
        raise RuntimeError("cleanup target set changed after confirmation")

    report_key = _validate_report(s3, control, experiment_id, confirmed.s3_keys)
    nonreport_keys = tuple(key for key in confirmed.s3_keys if key != report_key)
    _delete_s3_keys(s3, control, nonreport_keys)
    _delete_aim_hashes(repo_factory, repo_path, confirmed.aim_run_hashes)

    remaining_keys = _list_s3_keys(s3, control, experiment_id)
    _validate_report(s3, control, experiment_id, remaining_keys)
    remaining_nonreport = tuple(key for key in remaining_keys if key != report_key)
    remaining_hashes = _read_aim_hashes(repo_factory, repo_path, experiment_id)
    if remaining_nonreport or remaining_hashes:
        raise RuntimeError("cleanup postverification found remaining non-sentinel targets")

    _delete_s3_keys(s3, control, (report_key,))
    if _list_s3_keys(s3, control, experiment_id):
        raise RuntimeError("cleanup postverification found remaining S3 targets")
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
    )
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
