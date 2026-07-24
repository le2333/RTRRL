from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, NamedTuple, Protocol
import uuid

import boto3

from trainer_infra.aim_scratch import assert_aim_scratch_inactive
from trainer_infra.facility_control import FacilityControl, load_facility_control


ACCEPTANCE_AIM_SCRATCH = Path("/home/ubuntu/trainer/task7-aim-scratch")
ACCEPTANCE_MAIN_REPO = Path("/home/ubuntu/trainer/streaming-rtrrl")
MANIFEST_SCHEMA = "infra-acceptance-cleanup"
MANIFEST_VERSION = 1
_DELETE_BATCH_SIZE = 1000
_SHA256 = re.compile(r"[0-9a-f]{64}")


class CleanupRequest(NamedTuple):
    experiment_id: str
    confirm_prefix: str | None
    execute: bool
    manifest: Path | None
    confirm_manifest_sha256: str | None


class CleanupReport(NamedTuple):
    aim_run_hashes: tuple[str, ...]
    expected_prefix: str
    experiment_id: str
    ownership: dict[str, str]
    report_key: str
    report_sha256: str
    s3_keys: tuple[str, ...]
    schema: str
    version: int
    writes_performed: bool


class _Snapshot(NamedTuple):
    aim_run_hashes: tuple[str, ...]
    s3_keys: tuple[str, ...]


class _ReportEvidence(NamedTuple):
    key: str
    ownership: dict[str, str]
    sha256: str


class AimReadSession(Protocol):
    def iter_runs(self) -> Iterator[Any]: ...


class AimDeleteSession(AimReadSession, Protocol):
    def delete_run(self, run_hash: str) -> bool: ...


class AimRepoGateway(Protocol):
    path: Path

    def open_read_only(self) -> Any: ...

    def open_write_delete(self) -> Any: ...


class TrustedAimRepoGateway:
    def __init__(self, path: Path) -> None:
        self.path = path

    @staticmethod
    def _require_updated(path: Path) -> None:
        from aim import Repo
        from aim.sdk.repo import RepoStatus

        if Repo.check_repo_status(str(path)) is not RepoStatus.UPDATED:
            raise RuntimeError(
                "Aim repository must be pre-upgraded before acceptance cleanup"
            )

    @contextmanager
    def open_read_only(self) -> Iterator[AimReadSession]:
        from aim import Repo

        self._require_updated(self.path)
        try:
            repo = Repo(str(self.path), read_only=True, init=False)
        except NotImplementedError as error:
            raise RuntimeError(
                "installed Aim cannot open the repository read-only; "
                "pre-upgrade Aim before cleanup"
            ) from error
        try:
            yield repo
        finally:
            repo.close()

    @contextmanager
    def open_write_delete(self) -> Iterator[AimDeleteSession]:
        from aim import Repo

        self._require_updated(self.path)
        repo = Repo(str(self.path), init=False)
        try:
            yield repo
        finally:
            repo.close()


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
    return first == second or first in second.parents or second in first.parents


def _validate_aim_boundary(
    control: FacilityControl,
    aim_repo: AimRepoGateway,
) -> Path:
    scratch = Path(control.aim.repo)
    main = Path(control.aim.main_repo)
    gateway_path = Path(aim_repo.path)
    if (
        scratch != ACCEPTANCE_AIM_SCRATCH
        or main != ACCEPTANCE_MAIN_REPO
        or gateway_path != ACCEPTANCE_AIM_SCRATCH
    ):
        raise ValueError("Aim paths must match audited acceptance-only constants")
    resolved_scratch = scratch.resolve()
    resolved_main = main.resolve()
    resolved_gateway = gateway_path.resolve()
    if resolved_gateway != resolved_scratch or _paths_overlap(
        resolved_scratch, resolved_main
    ):
        raise ValueError("Aim scratch and main repository paths overlap")
    return resolved_scratch


def _list_s3_keys(
    s3: Any,
    control: FacilityControl,
    experiment_id: str,
) -> tuple[str, ...]:
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


def _report_evidence(
    s3: Any,
    control: FacilityControl,
    experiment_id: str,
    keys: tuple[str, ...],
) -> _ReportEvidence:
    report_key = f"{control.prefix}/{experiment_id}/report.json"
    if report_key not in keys:
        raise ValueError("canonical report.json is missing")
    try:
        body = s3.get_object(Bucket=control.bucket, Key=report_key)["Body"].read()
        report = json.loads(body)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("canonical report.json is unreadable") from error
    if not isinstance(body, bytes) or not isinstance(report, Mapping):
        raise ValueError("canonical report.json must be a byte-backed object")
    metadata = report.get("experiment_metadata")
    ownership = {
        "experiment_name": report.get("experiment_name"),
        "purpose": metadata.get("purpose") if isinstance(metadata, Mapping) else None,
    }
    expected = {
        "experiment_name": "infra-brax-ppo-acceptance",
        "purpose": "infra-acceptance",
    }
    if ownership != expected:
        raise ValueError("canonical report.json metadata does not prove ownership")
    return _ReportEvidence(
        key=report_key,
        ownership=expected,
        sha256=hashlib.sha256(body).hexdigest(),
    )


def _list_aim_hashes(repo: AimReadSession, experiment_id: str) -> tuple[str, ...]:
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
    aim_repo: AimRepoGateway,
    experiment_id: str,
) -> tuple[str, ...]:
    with aim_repo.open_read_only() as repo:
        return _list_aim_hashes(repo, experiment_id)


def _snapshot(
    *,
    control: FacilityControl,
    s3: Any,
    aim_repo: AimRepoGateway,
    experiment_id: str,
) -> _Snapshot:
    return _Snapshot(
        aim_run_hashes=_read_aim_hashes(aim_repo, experiment_id),
        s3_keys=_list_s3_keys(s3, control, experiment_id),
    )


def _delete_s3_keys(
    s3: Any,
    control: FacilityControl,
    keys: tuple[str, ...],
) -> None:
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
    aim_repo: AimRepoGateway,
    hashes: tuple[str, ...],
) -> None:
    if not hashes:
        return
    with aim_repo.open_write_delete() as repo:
        for run_hash in hashes:
            if repo.delete_run(run_hash) is not True:
                raise RuntimeError(f"Aim refused exact run deletion: {run_hash}")


def _manifest_report(data: Any) -> CleanupReport:
    if not isinstance(data, Mapping) or set(data) != set(CleanupReport._fields):
        raise ValueError("cleanup manifest schema fields are invalid")
    try:
        return CleanupReport(
            aim_run_hashes=tuple(data["aim_run_hashes"]),
            expected_prefix=data["expected_prefix"],
            experiment_id=data["experiment_id"],
            ownership=dict(data["ownership"]),
            report_key=data["report_key"],
            report_sha256=data["report_sha256"],
            s3_keys=tuple(data["s3_keys"]),
            schema=data["schema"],
            version=data["version"],
            writes_performed=data["writes_performed"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("cleanup manifest values are invalid") from error


def _load_authorized_manifest(request: CleanupRequest) -> CleanupReport:
    if request.manifest is None or request.confirm_manifest_sha256 is None:
        raise ValueError("execute requires manifest path and SHA-256")
    if not _SHA256.fullmatch(request.confirm_manifest_sha256):
        raise ValueError("execute requires a lowercase SHA-256")
    try:
        payload = request.manifest.read_bytes()
    except OSError as error:
        raise ValueError("cleanup manifest cannot be read") from error
    actual_sha = hashlib.sha256(payload).hexdigest()
    if actual_sha != request.confirm_manifest_sha256:
        raise ValueError("cleanup manifest SHA-256 does not match exact file bytes")
    try:
        data = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("cleanup manifest JSON is invalid") from error
    canonical = (json.dumps(data, separators=(",", ":"), sort_keys=True) + "\n").encode()
    if payload != canonical:
        raise ValueError("cleanup manifest is not canonical JSON with one newline")
    report = _manifest_report(data)
    if (
        report.schema != MANIFEST_SCHEMA
        or report.version != MANIFEST_VERSION
        or type(report.version) is not int
        or not isinstance(report.expected_prefix, str)
        or not isinstance(report.experiment_id, str)
        or not isinstance(report.report_key, str)
        or report.writes_performed is not False
        or any(not isinstance(value, str) or not value for value in report.s3_keys)
        or any(
            not isinstance(value, str) or not value
            for value in report.aim_run_hashes
        )
        or tuple(sorted(report.s3_keys)) != report.s3_keys
        or tuple(sorted(report.aim_run_hashes)) != report.aim_run_hashes
        or len(set(report.s3_keys)) != len(report.s3_keys)
        or len(set(report.aim_run_hashes)) != len(report.aim_run_hashes)
        or report.report_key not in report.s3_keys
        or report.ownership
        != {
            "experiment_name": "infra-brax-ppo-acceptance",
            "purpose": "infra-acceptance",
        }
        or not _SHA256.fullmatch(report.report_sha256)
    ):
        raise ValueError("cleanup manifest authorization fields are invalid")
    return report


def _validate_live_subset(
    live: _Snapshot,
    authorized: CleanupReport,
) -> None:
    if not set(live.s3_keys).issubset(authorized.s3_keys):
        raise ValueError("live S3 set contains a new target")
    if not set(live.aim_run_hashes).issubset(authorized.aim_run_hashes):
        raise ValueError("live Aim set contains a new target")


def cleanup(
    request: CleanupRequest,
    *,
    control: FacilityControl,
    s3: Any,
    aim_repo: AimRepoGateway,
) -> CleanupReport:
    experiment_id = _canonical_experiment_id(request.experiment_id)
    _validate_aim_boundary(control, aim_repo)
    expected_prefix = f"s3://{control.bucket}/{control.prefix}/{experiment_id}/"
    assert_aim_scratch_inactive(control.aim)

    if not request.execute:
        planned = _snapshot(
            control=control,
            s3=s3,
            aim_repo=aim_repo,
            experiment_id=experiment_id,
        )
        evidence = _report_evidence(s3, control, experiment_id, planned.s3_keys)
        return CleanupReport(
            aim_run_hashes=planned.aim_run_hashes,
            expected_prefix=expected_prefix,
            experiment_id=experiment_id,
            ownership=evidence.ownership,
            report_key=evidence.key,
            report_sha256=evidence.sha256,
            s3_keys=planned.s3_keys,
            schema=MANIFEST_SCHEMA,
            version=MANIFEST_VERSION,
            writes_performed=False,
        )

    if request.confirm_prefix != expected_prefix:
        raise ValueError(f"execute requires confirm_prefix {expected_prefix}")
    authorized = _load_authorized_manifest(request)
    if (
        authorized.experiment_id != experiment_id
        or authorized.expected_prefix != expected_prefix
        or authorized.report_key
        != f"{control.prefix}/{experiment_id}/report.json"
    ):
        raise ValueError("cleanup manifest does not authorize this experiment and prefix")

    live = _snapshot(
        control=control,
        s3=s3,
        aim_repo=aim_repo,
        experiment_id=experiment_id,
    )
    _validate_live_subset(live, authorized)
    report_present = authorized.report_key in live.s3_keys
    if report_present:
        evidence = _report_evidence(s3, control, experiment_id, live.s3_keys)
        if (
            evidence.key != authorized.report_key
            or evidence.sha256 != authorized.report_sha256
            or evidence.ownership != authorized.ownership
        ):
            raise ValueError("live report does not match the authorized manifest")

    nonreport_keys = tuple(
        key for key in live.s3_keys if key != authorized.report_key
    )
    _delete_s3_keys(s3, control, nonreport_keys)
    _delete_aim_hashes(aim_repo, live.aim_run_hashes)

    remaining_keys = _list_s3_keys(s3, control, experiment_id)
    remaining_hashes = _read_aim_hashes(aim_repo, experiment_id)
    expected_remaining = (authorized.report_key,) if report_present else ()
    if remaining_keys != expected_remaining or remaining_hashes:
        raise RuntimeError("cleanup verification found remaining non-report targets")

    if report_present:
        _delete_s3_keys(s3, control, (authorized.report_key,))
    final_keys = _list_s3_keys(s3, control, experiment_id)
    final_hashes = _read_aim_hashes(aim_repo, experiment_id)
    if final_keys or final_hashes:
        raise RuntimeError("cleanup final verification found remaining targets")
    return authorized._replace(writes_performed=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run or exactly clean one proven infra acceptance experiment."
    )
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--confirm-prefix")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--confirm-manifest-sha256")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    control = load_facility_control(arguments.control)
    request = CleanupRequest(
        experiment_id=arguments.experiment_id,
        confirm_prefix=arguments.confirm_prefix,
        execute=arguments.execute,
        manifest=arguments.manifest,
        confirm_manifest_sha256=arguments.confirm_manifest_sha256,
    )
    session = boto3.Session(region_name=control.region)
    report = cleanup(
        request,
        control=control,
        s3=session.client("s3"),
        aim_repo=TrustedAimRepoGateway(ACCEPTANCE_AIM_SCRATCH),
    )
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
