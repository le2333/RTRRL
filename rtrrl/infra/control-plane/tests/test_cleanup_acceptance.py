from __future__ import annotations

from contextlib import contextmanager
from io import BytesIO
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any
from types import ModuleType
import uuid

import pytest

from trainer_infra.facility_control import load_facility_control


SCRIPT = Path(__file__).parents[1] / "scripts" / "cleanup_acceptance.py"
CONTROL = Path(__file__).parents[1] / "config" / "facility.yaml"
EXPERIMENT_ID = "12345678-1234-5678-9234-567812345678"
KEY_PREFIX = f"experiments/{EXPERIMENT_ID}/"
REPORT_KEY = f"{KEY_PREFIX}report.json"
RUN_KEY = f"{KEY_PREFIX}runs/run-1.json"
PREFIX = f"s3://rtrrl-artifacts-007122174918/{KEY_PREFIX}"
REPORT = {
    "experiment_name": "infra-brax-ppo-acceptance",
    "experiment_metadata": {"purpose": "infra-acceptance"},
}
REPORT_BYTES = json.dumps(REPORT, sort_keys=True).encode()
REPORT_SHA = hashlib.sha256(REPORT_BYTES).hexdigest()


def _load(monkeypatch: pytest.MonkeyPatch) -> Any:
    spec = importlib.util.spec_from_file_location("cleanup_acceptance", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "assert_aim_scratch_inactive",
        lambda _control: {"status": "inactive"},
    )
    return module


class FakeRun:
    def __init__(self, run_hash: str, experiment_id: str) -> None:
        self.hash = run_hash
        self.context = {"experiment_id": experiment_id}

    def get(self, key: str, default: object = None) -> object:
        return self.context if key == "context" else default


class AimHandle:
    def __init__(self, gateway: FakeAimGateway, *, writable: bool) -> None:
        self.gateway = gateway
        self.writable = writable

    def iter_runs(self):
        return iter(list(self.gateway.runs.values()))

    def delete_run(self, run_hash: str) -> bool:
        assert self.writable
        if self.gateway.fail_hash_once == run_hash:
            self.gateway.fail_hash_once = None
            return False
        del self.gateway.runs[run_hash]
        self.gateway.deleted.append(run_hash)
        return True


class FakeAimGateway:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.runs = {
            "exact-b": FakeRun("exact-b", EXPERIMENT_ID),
            "exact-a": FakeRun("exact-a", EXPERIMENT_ID),
            "other": FakeRun("other", str(uuid.uuid4())),
        }
        self.calls: list[str] = []
        self.deleted: list[str] = []
        self.fail_hash_once: str | None = None
        self.inject_after_delete: str | None = None

    @contextmanager
    def open_read_only(self):
        self.calls.append("read")
        yield AimHandle(self, writable=False)
        self.calls.append("read-close")

    @contextmanager
    def open_write_delete(self):
        self.calls.append("write")
        try:
            yield AimHandle(self, writable=True)
        finally:
            if self.inject_after_delete is not None:
                value = self.inject_after_delete
                self.runs[value] = FakeRun(value, EXPERIMENT_ID)
                self.inject_after_delete = None
            self.calls.append("write-close")


class FakeS3:
    def __init__(
        self,
        *,
        keys: list[str] | None = None,
        report_bytes: bytes = REPORT_BYTES,
    ) -> None:
        self.keys = set([REPORT_KEY, RUN_KEY] if keys is None else keys)
        self.report_bytes = report_bytes
        self.deleted: list[str] = []
        self.delete_batches: list[tuple[str, ...]] = []
        self.fail_nonreport_once = False
        self.fail_aim_stage_new_key: str | None = None
        self.fail_report_once = False
        self.remove_report_before_error = False

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["Bucket"] == "rtrrl-artifacts-007122174918"
        assert kwargs["Prefix"] == KEY_PREFIX
        return {
            "Contents": [{"Key": key} for key in sorted(self.keys)],
            "IsTruncated": False,
        }

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        assert kwargs == {
            "Bucket": "rtrrl-artifacts-007122174918",
            "Key": REPORT_KEY,
        }
        if REPORT_KEY not in self.keys:
            raise KeyError(REPORT_KEY)
        return {"Body": BytesIO(self.report_bytes)}

    def delete_objects(self, **kwargs: Any) -> dict[str, Any]:
        objects = tuple(item["Key"] for item in kwargs["Delete"]["Objects"])
        self.delete_batches.append(objects)
        if objects == (REPORT_KEY,) and self.fail_report_once:
            self.fail_report_once = False
            if self.remove_report_before_error:
                self.keys.remove(REPORT_KEY)
            raise TimeoutError("uncertain report deletion")
        failure = objects[-1] if REPORT_KEY not in objects and self.fail_nonreport_once else None
        self.fail_nonreport_once = False
        deleted = []
        errors = []
        for key in objects:
            if key == failure:
                errors.append({"Key": key, "Code": "InternalError"})
                continue
            self.keys.remove(key)
            self.deleted.append(key)
            deleted.append({"Key": key})
        return {
            "Deleted": deleted,
            "Errors": errors,
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }


def _gateway(control: Any) -> FakeAimGateway:
    return FakeAimGateway(control.aim.repo)


def _dry_request(module: Any) -> Any:
    return module.CleanupRequest(EXPERIMENT_ID, None, False, None, None)


def _save_manifest(
    module: Any,
    tmp_path: Path,
    report: Any,
) -> tuple[Path, str]:
    path = tmp_path / "cleanup-manifest.json"
    payload = (module.canonical_json(report) + "\n").encode()
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def _execute_request(module: Any, manifest: Path, digest: str) -> Any:
    return module.CleanupRequest(
        EXPERIMENT_ID,
        PREFIX,
        True,
        manifest,
        digest,
    )


def _manifest(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    s3: FakeS3,
    gateway: FakeAimGateway,
) -> tuple[Path, str, Any]:
    control = load_facility_control(CONTROL)
    report = module.cleanup(
        _dry_request(module),
        control=control,
        s3=s3,
        aim_repo=gateway,
    )
    path, digest = _save_manifest(module, tmp_path, report)
    return path, digest, report


def test_dry_run_emits_complete_canonical_recovery_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(monkeypatch)
    control = load_facility_control(CONTROL)
    s3 = FakeS3()
    gateway = _gateway(control)

    report = module.cleanup(
        _dry_request(module),
        control=control,
        s3=s3,
        aim_repo=gateway,
    )

    assert report.schema == "infra-acceptance-cleanup"
    assert report.version == 1
    assert report.experiment_id == EXPERIMENT_ID
    assert report.expected_prefix == PREFIX
    assert report.s3_keys == (REPORT_KEY, RUN_KEY)
    assert report.aim_run_hashes == ("exact-a", "exact-b")
    assert report.report_key == REPORT_KEY
    assert report.report_sha256 == REPORT_SHA
    assert report.ownership == {
        "experiment_name": "infra-brax-ppo-acceptance",
        "purpose": "infra-acceptance",
    }
    assert report.writes_performed is False
    assert gateway.calls == ["read", "read-close"]
    rendered = module.canonical_json(report)
    assert rendered == json.dumps(json.loads(rendered), separators=(",", ":"), sort_keys=True)


def test_cleanup_refuses_active_scratch_before_enumeration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(monkeypatch)
    monkeypatch.setattr(
        module,
        "assert_aim_scratch_inactive",
        lambda _control: (_ for _ in ()).throw(ValueError("Aim scratch is active")),
    )
    control = load_facility_control(CONTROL)
    gateway = _gateway(control)

    with pytest.raises(ValueError, match="active"):
        module.cleanup(
            _dry_request(module),
            control=control,
            s3=FakeS3(),
            aim_repo=gateway,
        )
    assert gateway.calls == []


def test_execute_requires_exact_manifest_path_hash_and_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load(monkeypatch)
    control = load_facility_control(CONTROL)
    s3 = FakeS3()
    gateway = _gateway(control)
    manifest, digest, _report = _manifest(module, monkeypatch, tmp_path, s3, gateway)

    for request in (
        module.CleanupRequest(EXPERIMENT_ID, PREFIX, True, None, digest),
        module.CleanupRequest(EXPERIMENT_ID, PREFIX, True, manifest, None),
        module.CleanupRequest(EXPERIMENT_ID, PREFIX, True, manifest, "0" * 64),
        module.CleanupRequest(EXPERIMENT_ID, PREFIX.rstrip("/"), True, manifest, digest),
    ):
        with pytest.raises(ValueError):
            module.cleanup(request, control=control, s3=s3, aim_repo=gateway)

    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256"):
        module.cleanup(
            _execute_request(module, manifest, digest),
            control=control,
            s3=s3,
            aim_repo=gateway,
        )


def test_execute_deletes_in_order_and_finally_verifies_both_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load(monkeypatch)
    control = load_facility_control(CONTROL)
    s3 = FakeS3()
    gateway = _gateway(control)
    manifest, digest, _report = _manifest(module, monkeypatch, tmp_path, s3, gateway)

    result = module.cleanup(
        _execute_request(module, manifest, digest),
        control=control,
        s3=s3,
        aim_repo=gateway,
    )

    assert result.writes_performed is True
    assert s3.delete_batches == [(RUN_KEY,), (REPORT_KEY,)]
    assert s3.keys == set()
    assert set(gateway.runs) == {"other"}
    assert gateway.calls.count("write") == 1
    assert gateway.calls[-2:] == ["read", "read-close"]


@pytest.mark.parametrize("new_target", ["s3", "aim"])
def test_manifest_rejects_every_new_live_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    new_target: str,
) -> None:
    module = _load(monkeypatch)
    control = load_facility_control(CONTROL)
    s3 = FakeS3()
    gateway = _gateway(control)
    manifest, digest, _report = _manifest(module, monkeypatch, tmp_path, s3, gateway)
    if new_target == "s3":
        s3.keys.add(f"{KEY_PREFIX}new.json")
    else:
        gateway.runs["new"] = FakeRun("new", EXPERIMENT_ID)

    with pytest.raises(ValueError, match="new"):
        module.cleanup(
            _execute_request(module, manifest, digest),
            control=control,
            s3=s3,
            aim_repo=gateway,
        )
    assert s3.deleted == []
    assert gateway.deleted == []


@pytest.mark.parametrize("report_present", [True, False])
def test_manifest_authorizes_subset_recovery_with_or_without_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    report_present: bool,
) -> None:
    module = _load(monkeypatch)
    control = load_facility_control(CONTROL)
    original_s3 = FakeS3()
    original_gateway = _gateway(control)
    manifest, digest, _report = _manifest(
        module,
        monkeypatch,
        tmp_path,
        original_s3,
        original_gateway,
    )
    s3 = FakeS3(keys=[REPORT_KEY] if report_present else [])
    gateway = _gateway(control)
    del gateway.runs["exact-a"]

    module.cleanup(
        _execute_request(module, manifest, digest),
        control=control,
        s3=s3,
        aim_repo=gateway,
    )
    assert s3.keys == set()
    assert set(gateway.runs) == {"other"}


def test_report_present_is_rehashed_and_ownership_revalidated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load(monkeypatch)
    control = load_facility_control(CONTROL)
    s3 = FakeS3()
    gateway = _gateway(control)
    manifest, digest, _report = _manifest(module, monkeypatch, tmp_path, s3, gateway)
    s3.report_bytes = json.dumps(
        {
            "experiment_name": "other",
            "experiment_metadata": {"purpose": "infra-acceptance"},
        }
    ).encode()

    with pytest.raises(ValueError, match="report"):
        module.cleanup(
            _execute_request(module, manifest, digest),
            control=control,
            s3=s3,
            aim_repo=gateway,
        )


def test_partial_nonreport_and_aim_failures_preserve_manifest_for_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load(monkeypatch)
    control = load_facility_control(CONTROL)
    s3 = FakeS3(keys=[REPORT_KEY, RUN_KEY, f"{KEY_PREFIX}other.json"])
    gateway = _gateway(control)
    manifest, digest, _report = _manifest(module, monkeypatch, tmp_path, s3, gateway)
    request = _execute_request(module, manifest, digest)
    s3.fail_nonreport_once = True

    with pytest.raises(RuntimeError, match="S3 deletion"):
        module.cleanup(request, control=control, s3=s3, aim_repo=gateway)
    assert manifest.exists()
    assert REPORT_KEY in s3.keys

    gateway.fail_hash_once = "exact-b"
    with pytest.raises(RuntimeError, match="Aim refused"):
        module.cleanup(request, control=control, s3=s3, aim_repo=gateway)
    assert manifest.exists()
    assert REPORT_KEY in s3.keys

    module.cleanup(request, control=control, s3=s3, aim_repo=gateway)
    assert s3.keys == set()


def test_uncertain_final_report_delete_recovers_from_missing_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load(monkeypatch)
    control = load_facility_control(CONTROL)
    s3 = FakeS3()
    gateway = _gateway(control)
    manifest, digest, _report = _manifest(module, monkeypatch, tmp_path, s3, gateway)
    request = _execute_request(module, manifest, digest)
    s3.fail_report_once = True
    s3.remove_report_before_error = True

    with pytest.raises(TimeoutError, match="uncertain"):
        module.cleanup(request, control=control, s3=s3, aim_repo=gateway)
    assert manifest.exists()
    assert s3.keys == set()
    module.cleanup(request, control=control, s3=s3, aim_repo=gateway)


@pytest.mark.parametrize("new_target", ["s3", "aim"])
def test_final_new_target_fails_and_same_manifest_keeps_rejecting_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    new_target: str,
) -> None:
    module = _load(monkeypatch)
    control = load_facility_control(CONTROL)
    s3 = FakeS3()
    gateway = _gateway(control)
    manifest, digest, _report = _manifest(module, monkeypatch, tmp_path, s3, gateway)
    request = _execute_request(module, manifest, digest)
    if new_target == "aim":
        gateway.inject_after_delete = "late"
    else:
        original = s3.list_objects_v2
        calls = 0

        def listing(**kwargs: Any) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            if calls == 3:
                s3.keys.add(f"{KEY_PREFIX}late.json")
            return original(**kwargs)

        s3.list_objects_v2 = listing  # type: ignore[method-assign]

    with pytest.raises((RuntimeError, ValueError), match="remaining|new"):
        module.cleanup(request, control=control, s3=s3, aim_repo=gateway)
    with pytest.raises(ValueError, match="new"):
        module.cleanup(request, control=control, s3=s3, aim_repo=gateway)


def test_public_cleanup_uses_gateway_not_repo_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(monkeypatch)
    assert "aim_repo" in module.cleanup.__annotations__
    assert "repo_factory" not in module.cleanup.__annotations__


def _fake_aim_modules(
    monkeypatch: pytest.MonkeyPatch,
    opened: list[Path],
) -> None:
    updated = object()

    class Repo:
        @staticmethod
        def check_repo_status(_path: str) -> object:
            return updated

        def __init__(
            self,
            path: str,
            *,
            read_only: bool | None = None,
            init: bool,
        ) -> None:
            assert read_only is None
            assert init is False
            opened.append(Path(path))

        def iter_runs(self):
            return iter(())

        def close(self) -> None:
            return None

    aim = ModuleType("aim")
    aim.Repo = Repo  # type: ignore[attr-defined]
    sdk = ModuleType("aim.sdk")
    repo_module = ModuleType("aim.sdk.repo")
    repo_module.RepoStatus = type("RepoStatus", (), {"UPDATED": updated})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aim", aim)
    monkeypatch.setitem(sys.modules, "aim.sdk", sdk)
    monkeypatch.setitem(sys.modules, "aim.sdk.repo", repo_module)


def _snapshot_source(tmp_path: Path) -> Path:
    source = tmp_path / "scratch"
    aim = source / ".aim"
    aim.mkdir(parents=True)
    (aim / "VERSION").write_text("3.28\n")
    data = aim / "data"
    data.mkdir()
    (data / "value").write_bytes(b"stable")
    return source


def test_trusted_gateway_opens_only_verified_temp_snapshot_and_cleans_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load(monkeypatch)
    source = _snapshot_source(tmp_path)
    before = module._source_tree_fingerprint(source / ".aim")
    opened: list[Path] = []
    _fake_aim_modules(monkeypatch, opened)

    gateway = module.TrustedAimRepoGateway(source)
    with gateway.open_read_only() as repo:
        assert list(repo.iter_runs()) == []
        assert opened[0] != source
        assert opened[0].parent != source
        assert (opened[0] / ".aim" / "data" / "value").read_bytes() == b"stable"
        temporary = opened[0]

    assert not temporary.exists()
    assert module._source_tree_fingerprint(source / ".aim") == before


@pytest.mark.parametrize(
    "failure",
    [
        "source-symlink",
        "source-special",
        "source-mutation",
        "copy-mismatch",
        "copy-symlink",
    ],
)
def test_snapshot_rejects_unsafe_or_changed_tree_and_fail_cleans_temp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
) -> None:
    module = _load(monkeypatch)
    source = _snapshot_source(tmp_path)
    opened: list[Path] = []
    _fake_aim_modules(monkeypatch, opened)
    copied: list[Path] = []
    real_copytree = shutil.copytree

    if failure == "source-symlink":
        (source / ".aim" / "link").symlink_to(source / ".aim" / "VERSION")
    elif failure == "source-special":
        os.mkfifo(source / ".aim" / "fifo")
    else:
        def changed_copytree(src: Path, dst: Path) -> Path:
            result = real_copytree(
                src,
                dst,
                symlinks=True,
                copy_function=shutil.copy2,
            )
            copied.append(Path(dst).parent)
            if failure == "source-mutation":
                (source / ".aim" / "data" / "value").write_bytes(b"mutated")
            elif failure == "copy-mismatch":
                (Path(dst) / "data" / "value").write_bytes(b"mismatch")
            else:
                (Path(dst) / "late-link").symlink_to(Path(dst) / "VERSION")
            return result

        monkeypatch.setattr(module, "_copy_aim_tree", changed_copytree)

    with pytest.raises(ValueError, match="symlink|special|changed|mismatch"):
        with module.TrustedAimRepoGateway(source).open_read_only():
            raise AssertionError("unsafe snapshot must never open")
    assert opened == []
    assert all(not path.exists() for path in copied)


def test_real_aim_328_snapshot_lists_runs_without_source_change(tmp_path: Path) -> None:
    from aim import Run

    module = importlib.util.module_from_spec(
        importlib.util.spec_from_file_location("cleanup_real_aim", SCRIPT)
    )
    spec = module.__spec__
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    source = tmp_path / "real-aim"
    run = Run(repo=str(source), experiment="acceptance")
    run["context"] = {"experiment_id": EXPERIMENT_ID}
    run_hash = run.hash
    run.close()
    before = module._source_tree_fingerprint(source / ".aim")

    with module.TrustedAimRepoGateway(source).open_read_only() as repo:
        assert run_hash in {item.hash for item in repo.iter_runs()}

    assert module._source_tree_fingerprint(source / ".aim") == before


@pytest.mark.parametrize(
    "experiment_id",
    ["", "..", "not-a-uuid", "12345678/1234-5678-9234-567812345678"],
)
def test_cleanup_rejects_noncanonical_ids_before_any_gateway_open(
    monkeypatch: pytest.MonkeyPatch,
    experiment_id: str,
) -> None:
    module = _load(monkeypatch)
    control = load_facility_control(CONTROL)
    gateway = _gateway(control)
    request = module.CleanupRequest(experiment_id, None, False, None, None)
    with pytest.raises(ValueError, match="canonical UUID"):
        module.cleanup(request, control=control, s3=FakeS3(), aim_repo=gateway)
    assert gateway.calls == []


def test_cleanup_rejects_untrusted_gateway_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(monkeypatch)
    control = load_facility_control(CONTROL)
    gateway = _gateway(control)
    gateway.path = control.aim.main_repo
    with pytest.raises(ValueError, match="audited"):
        module.cleanup(
            _dry_request(module),
            control=control,
            s3=FakeS3(),
            aim_repo=gateway,
        )
    assert gateway.calls == []


def test_cleanup_rejects_lexical_trusted_scratch_symlink_to_other_repo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load(monkeypatch)
    target = tmp_path / "other"
    (target / ".aim").mkdir(parents=True)
    lexical = tmp_path / "trusted-scratch"
    lexical.symlink_to(target, target_is_directory=True)
    control = load_facility_control(CONTROL)
    aim = control.aim.model_copy(update={"repo": lexical})
    control = control.model_copy(update={"aim": aim})
    monkeypatch.setattr(module, "ACCEPTANCE_AIM_SCRATCH", lexical)
    gateway = FakeAimGateway(lexical)

    with pytest.raises(ValueError, match="symlink|canonical"):
        module.cleanup(
            _dry_request(module),
            control=control,
            s3=FakeS3(),
            aim_repo=gateway,
        )
    assert gateway.calls == []
