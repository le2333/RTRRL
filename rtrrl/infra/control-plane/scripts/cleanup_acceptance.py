from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from typing import Any, NamedTuple, Protocol
import uuid

import boto3

from trainer_infra.aim_scratch import (
    assert_aim_scratch_inactive,
    open_facility_lock,
    open_trusted_directory,
)
from trainer_infra.facility_control import (
    AimScratchControl,
    FacilityControl,
    load_facility_control,
)


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

    def cleanup_operation(self, control: AimScratchControl) -> Any: ...


class _TreeEntry(NamedTuple):
    relative_path: str
    entry_type: str
    mode: int
    size: int
    sha256: str
    inode: int
    mtime_ns: int


def _tree_entry(
    *,
    relative_path: str,
    entry_type: str,
    file_stat: os.stat_result,
    size: int,
    sha256: str,
) -> _TreeEntry:
    return _TreeEntry(
        relative_path=relative_path,
        entry_type=entry_type,
        mode=stat.S_IMODE(file_stat.st_mode),
        size=size,
        sha256=sha256,
        inode=file_stat.st_ino,
        mtime_ns=file_stat.st_mtime_ns,
    )


def _source_tree_fingerprint(root: Path) -> tuple[_TreeEntry, ...]:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | nofollow
    try:
        root_fd = os.open(root, directory_flags)
    except OSError as error:
        raise ValueError(f"Aim tree root is not a safe directory: {root}") from error
    entries: list[_TreeEntry] = []

    def visit(directory_fd: int, relative: str) -> None:
        directory_stat = os.fstat(directory_fd)
        if not stat.S_ISDIR(directory_stat.st_mode):
            raise ValueError("Aim tree contains a special file")
        entries.append(
            _tree_entry(
                relative_path=relative,
                entry_type="directory",
                file_stat=directory_stat,
                size=0,
                sha256=hashlib.sha256(b"").hexdigest(),
            )
        )
        for name in sorted(os.listdir(directory_fd)):
            child_relative = name if relative == "." else f"{relative}/{name}"
            child_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISLNK(child_stat.st_mode):
                raise ValueError(f"Aim tree contains a symlink: {child_relative}")
            if stat.S_ISDIR(child_stat.st_mode):
                child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
                try:
                    opened_stat = os.fstat(child_fd)
                    if (
                        opened_stat.st_dev != child_stat.st_dev
                        or opened_stat.st_ino != child_stat.st_ino
                    ):
                        raise ValueError("Aim tree changed during directory inspection")
                    visit(child_fd, child_relative)
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(child_stat.st_mode):
                raise ValueError(f"Aim tree contains a special file: {child_relative}")
            file_fd = os.open(name, os.O_RDONLY | nofollow, dir_fd=directory_fd)
            try:
                opened_stat = os.fstat(file_fd)
                if (
                    opened_stat.st_dev != child_stat.st_dev
                    or opened_stat.st_ino != child_stat.st_ino
                ):
                    raise ValueError("Aim tree changed during file inspection")
                digest = hashlib.sha256()
                size = 0
                while chunk := os.read(file_fd, 1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
                final_stat = os.fstat(file_fd)
                if (
                    final_stat.st_ino != opened_stat.st_ino
                    or final_stat.st_size != opened_stat.st_size
                    or final_stat.st_mtime_ns != opened_stat.st_mtime_ns
                    or size != opened_stat.st_size
                ):
                    raise ValueError("Aim tree changed during file hashing")
                entries.append(
                    _tree_entry(
                        relative_path=child_relative,
                        entry_type="file",
                        file_stat=final_stat,
                        size=size,
                        sha256=digest.hexdigest(),
                    )
                )
            finally:
                os.close(file_fd)

    try:
        visit(root_fd, ".")
    finally:
        os.close(root_fd)
    return tuple(entries)


def _content_manifest(
    fingerprint: tuple[_TreeEntry, ...],
) -> tuple[tuple[str, str, int, int, str], ...]:
    return tuple(
        (
            entry.relative_path,
            entry.entry_type,
            entry.mode,
            entry.size,
            entry.sha256,
        )
        for entry in fingerprint
    )


def _require_lexical_directory(path: Path) -> Path:
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise ValueError("trusted Aim path must be absolute and lexically canonical")
    for component in reversed((path, *path.parents)):
        try:
            component_stat = component.lstat()
        except OSError as error:
            raise ValueError("trusted Aim path must exist") from error
        if stat.S_ISLNK(component_stat.st_mode):
            raise ValueError("trusted Aim path cannot contain a symlink")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ValueError("trusted Aim path must equal its strict canonical path")
    if not path.is_dir():
        raise ValueError("trusted Aim path must be a directory")
    return path


def _copy_aim_tree(source: Path, destination: Path) -> Path:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    source_fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY | nofollow)
    try:
        stable_source = Path("/proc/self/fd") / str(source_fd)
        return shutil.copytree(
            stable_source,
            destination,
            symlinks=True,
            copy_function=shutil.copy2,
        )
    finally:
        os.close(source_fd)


@contextmanager
def _verified_aim_snapshot(
    source_repo: Path,
    *,
    disk_usage: Any = shutil.disk_usage,
) -> Iterator[Path]:
    source_aim = source_repo / ".aim"
    before = _source_tree_fingerprint(source_aim)
    source_size = sum(entry.size for entry in before if entry.entry_type == "file")
    with tempfile.TemporaryDirectory(prefix="acceptance-aim-snapshot-") as temporary:
        required = 3 * source_size + 512 * 1024 * 1024
        if disk_usage(temporary).free < required:
            raise ValueError(f"insufficient temporary snapshot capacity: require {required} bytes")
        temporary_repo = Path(temporary) / "repo"
        temporary_repo.mkdir()
        copied_aim = temporary_repo / ".aim"
        _copy_aim_tree(source_aim, copied_aim)
        after = _source_tree_fingerprint(source_aim)
        copied = _source_tree_fingerprint(copied_aim)
        if before != after:
            raise ValueError("Aim source tree changed while snapshotting")
        if _content_manifest(before) != _content_manifest(copied):
            raise ValueError("Aim snapshot content manifest mismatch")
        yield temporary_repo


class TrustedAimRepoGateway:
    def __init__(self, path: Path, *, disk_usage: Any = shutil.disk_usage) -> None:
        self.path = path
        self._disk_usage = disk_usage
        self._control: AimScratchControl | None = None
        self._directory_fd: int | None = None
        self._lock_fd: int | None = None
        self._snapshot_context: Any = None
        self._snapshot_repo: Any = None

    def _source_path(self) -> Path:
        if self._directory_fd is None:
            raise RuntimeError("Aim gateway is outside an exclusive cleanup operation")
        return Path("/proc/self/fd") / str(self._directory_fd)

    def _assert_inactive(self) -> None:
        if self._control is None:
            raise RuntimeError("Aim gateway has no active facility control")
        secured = self._control.model_copy(update={"repo": self._source_path()})
        assert_aim_scratch_inactive(secured)

    def _close_snapshot(self) -> None:
        repo = self._snapshot_repo
        snapshot_context = self._snapshot_context
        self._snapshot_repo = None
        self._snapshot_context = None
        close_error: BaseException | None = None
        try:
            if repo is not None:
                repo.close()
                gc.collect()
        except BaseException as error:
            close_error = error
        finally:
            try:
                if snapshot_context is not None:
                    snapshot_context.__exit__(None, None, None)
            except BaseException as error:
                if close_error is None:
                    close_error = error
                else:
                    close_error.add_note(f"snapshot cleanup also failed: {error!r}")
        if close_error is not None:
            raise close_error

    @contextmanager
    def cleanup_operation(self, control: AimScratchControl) -> Iterator[None]:
        if self._directory_fd is not None:
            raise RuntimeError("Aim cleanup operation is already active")
        directory_fd = open_trusted_directory(self.path)
        try:
            lock_fd = open_facility_lock(directory_fd)
        except BaseException:
            os.close(directory_fd)
            raise
        self._control = control
        self._directory_fd = directory_fd
        self._lock_fd = lock_fd
        primary_error: BaseException | None = None
        close_errors: list[BaseException] = []
        try:
            self._assert_inactive()
            yield
        except BaseException as error:
            primary_error = error
        finally:
            try:
                self._close_snapshot()
            except BaseException as error:
                close_errors.append(error)
            self._control = None
            self._directory_fd = None
            self._lock_fd = None
            try:
                try:
                    os.close(lock_fd)
                except BaseException as error:
                    close_errors.append(error)
            finally:
                try:
                    os.close(directory_fd)
                except BaseException as error:
                    close_errors.append(error)
        if primary_error is not None:
            for error in close_errors:
                primary_error.add_note(f"cleanup close also failed: {error!r}")
                for note in getattr(error, "__notes__", ()):
                    primary_error.add_note(note)
            raise primary_error
        if close_errors:
            first, *additional = close_errors
            for error in additional:
                first.add_note(f"additional cleanup close failed: {error!r}")
            raise first

    @staticmethod
    def _require_updated(path: Path) -> None:
        from aim import Repo
        from aim.sdk.repo import RepoStatus

        if Repo.check_repo_status(str(path)) is not RepoStatus.UPDATED:
            raise RuntimeError("Aim repository must be pre-upgraded before acceptance cleanup")

    @contextmanager
    def open_read_only(self) -> Iterator[AimReadSession]:
        from aim import Repo
        from aim.sdk.repo import RepoStatus

        self._assert_inactive()
        if self._snapshot_repo is None:
            snapshot_context = _verified_aim_snapshot(
                self._source_path(),
                disk_usage=self._disk_usage,
            )
            snapshot = snapshot_context.__enter__()
            try:
                repo = Repo(str(snapshot), init=False)
                if Repo.check_repo_status(str(snapshot)) is not RepoStatus.UPDATED:
                    repo.close()
                    raise RuntimeError("temporary Aim snapshot could not be upgraded")
            except BaseException:
                snapshot_context.__exit__(*sys.exc_info())
                raise
            self._snapshot_context = snapshot_context
            self._snapshot_repo = repo
        yield self._snapshot_repo

    @contextmanager
    def open_write_delete(self) -> Iterator[AimDeleteSession]:
        from aim import Repo

        self._close_snapshot()
        self._assert_inactive()
        source = self._source_path()
        self._require_updated(source)
        repo = Repo(str(source), init=False)
        primary_error: BaseException | None = None
        close_error: BaseException | None = None
        try:
            self._assert_inactive()
            yield repo
        except BaseException as error:
            primary_error = error
        finally:
            try:
                repo.close()
            except BaseException as error:
                close_error = error
            finally:
                gc.collect()
        if primary_error is not None:
            if close_error is not None:
                primary_error.add_note(f"writable Aim repo close also failed: {close_error!r}")
            raise primary_error
        if close_error is not None:
            raise close_error


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
    _require_lexical_directory(scratch)
    resolved_scratch = scratch.resolve()
    resolved_main = main.resolve()
    resolved_gateway = gateway_path.resolve()
    if resolved_gateway != resolved_scratch or _paths_overlap(resolved_scratch, resolved_main):
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
        deleted_keys = {item.get("Key") for item in deleted if isinstance(item, Mapping)}
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
        lexical_stat = request.manifest.lstat()
    except OSError as error:
        raise ValueError("cleanup manifest cannot be read") from error
    if not stat.S_ISREG(lexical_stat.st_mode):
        raise ValueError("cleanup manifest must be a regular file, not a symlink")
    if lexical_stat.st_uid != os.getuid():
        raise ValueError("cleanup manifest must be owned by the current uid")
    if stat.S_IMODE(lexical_stat.st_mode) & 0o077:
        raise ValueError("cleanup manifest mode must not grant group or other access")
    try:
        manifest_fd = os.open(
            request.manifest,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            opened_stat = os.fstat(manifest_fd)
            if (
                opened_stat.st_dev != lexical_stat.st_dev
                or opened_stat.st_ino != lexical_stat.st_ino
                or not stat.S_ISREG(opened_stat.st_mode)
                or opened_stat.st_uid != os.getuid()
                or stat.S_IMODE(opened_stat.st_mode) & 0o077
            ):
                raise ValueError("cleanup manifest identity or mode changed while opening")
            chunks = []
            while chunk := os.read(manifest_fd, 1024 * 1024):
                chunks.append(chunk)
            payload = b"".join(chunks)
        finally:
            os.close(manifest_fd)
    except OSError as error:
        raise ValueError("cleanup manifest cannot be safely opened") from error
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
        or any(not isinstance(value, str) or not value for value in report.aim_run_hashes)
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


def _cleanup_locked(
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
        or authorized.report_key != f"{control.prefix}/{experiment_id}/report.json"
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

    nonreport_keys = tuple(key for key in live.s3_keys if key != authorized.report_key)
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


def cleanup(
    request: CleanupRequest,
    *,
    control: FacilityControl,
    s3: Any,
    aim_repo: AimRepoGateway,
) -> CleanupReport:
    _canonical_experiment_id(request.experiment_id)
    _validate_aim_boundary(control, aim_repo)
    with aim_repo.cleanup_operation(control.aim):
        assert_aim_scratch_inactive(control.aim)
        return _cleanup_locked(
            request,
            control=control,
            s3=s3,
            aim_repo=aim_repo,
        )


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
