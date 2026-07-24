from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import importlib.util
import json
from pathlib import Path
from typing import Any
import uuid

import pytest

from trainer_infra.facility_control import load_facility_control


SCRIPT = Path(__file__).parents[1] / "scripts" / "cleanup_acceptance.py"
CONTROL = Path(__file__).parents[1] / "config" / "facility.yaml"
EXPERIMENT_ID = "12345678-1234-5678-9234-567812345678"
KEY_PREFIX = f"experiments/{EXPERIMENT_ID}/"
REPORT_KEY = f"{KEY_PREFIX}report.json"
PREFIX = f"s3://rtrrl-artifacts-007122174918/{KEY_PREFIX}"
REPORT = {
    "experiment_name": "infra-brax-ppo-acceptance",
    "experiment_metadata": {"purpose": "infra-acceptance"},
}


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("cleanup_acceptance", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeS3:
    def __init__(
        self,
        *,
        keys: list[str] | None = None,
        report: dict[str, Any] | None = None,
        change_on_list: int | None = None,
        fail_nonreport_once: bool = False,
        fail_report_once: bool = False,
    ) -> None:
        self.keys = set(keys or [REPORT_KEY, f"{KEY_PREFIX}runs/run-1.json"])
        self.report = deepcopy(REPORT if report is None else report)
        self.change_on_list = change_on_list
        self.fail_nonreport_once = fail_nonreport_once
        self.fail_report_once = fail_report_once
        self.list_calls = 0
        self.deleted: list[str] = []
        self.delete_batches: list[tuple[str, ...]] = []

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        assert kwargs == {
            "Bucket": "rtrrl-artifacts-007122174918",
            "Prefix": KEY_PREFIX,
        }
        self.list_calls += 1
        if self.change_on_list == self.list_calls:
            self.keys.add(f"{KEY_PREFIX}late.json")
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
        return {"Body": BytesIO(json.dumps(self.report).encode("utf-8"))}

    def delete_objects(self, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["Bucket"] == "rtrrl-artifacts-007122174918"
        objects = tuple(item["Key"] for item in kwargs["Delete"]["Objects"])
        assert kwargs["Delete"]["Quiet"] is False
        self.delete_batches.append(objects)
        failure: str | None = None
        if REPORT_KEY in objects and self.fail_report_once:
            self.fail_report_once = False
            failure = REPORT_KEY
        elif REPORT_KEY not in objects and self.fail_nonreport_once:
            self.fail_nonreport_once = False
            failure = objects[-1]
        deleted = []
        errors = []
        for key in objects:
            if key == failure:
                errors.append({"Key": key, "Code": "InternalError", "Message": "injected"})
                continue
            assert key in self.keys
            self.keys.remove(key)
            self.deleted.append(key)
            deleted.append({"Key": key})
        return {
            "Deleted": deleted,
            "Errors": errors,
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }


class PagedS3:
    def __init__(self, pages: dict[str | None, dict[str, Any]]) -> None:
        self.pages = pages
        self.calls: list[str | None] = []

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["Bucket"] == "rtrrl-artifacts-007122174918"
        assert kwargs["Prefix"] == KEY_PREFIX
        token = kwargs.get("ContinuationToken")
        self.calls.append(token)
        return deepcopy(self.pages[token])


class FakeRun:
    def __init__(self, run_hash: str, experiment_id: str) -> None:
        self.hash = run_hash
        self._context = {"experiment_id": experiment_id}

    def get(self, key: str, default: object = None) -> object:
        return self._context if key == "context" else default


class AimState:
    def __init__(self) -> None:
        self.runs = {
            "exact-b": FakeRun("exact-b", EXPERIMENT_ID),
            "other": FakeRun("other", str(uuid.uuid4())),
            "exact-a": FakeRun("exact-a", EXPERIMENT_ID),
        }
        self.deleted: list[str] = []
        self.fail_hash_once: str | None = None


class FakeAimRepo:
    def __init__(self, path: Path, state: AimState, *, read_only: bool) -> None:
        self.path = path / ".aim"
        self.state = state
        self.read_only = read_only
        self.closed = 0

    def iter_runs(self):
        return iter(list(self.state.runs.values()))

    def delete_run(self, run_hash: str) -> bool:
        assert self.read_only is False
        if self.state.fail_hash_once == run_hash:
            self.state.fail_hash_once = None
            return False
        assert run_hash in self.state.runs
        del self.state.runs[run_hash]
        self.state.deleted.append(run_hash)
        return True

    def close(self) -> None:
        self.closed += 1


class RepoFactory:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.state = AimState()
        self.calls: list[tuple[str, bool, bool]] = []
        self.repos: list[FakeAimRepo] = []

    def __call__(self, path: str, *, read_only: bool, init: bool) -> FakeAimRepo:
        self.calls.append((path, read_only, init))
        repo = FakeAimRepo(self.path, self.state, read_only=read_only)
        self.repos.append(repo)
        return repo


def _request(cleanup_module: Any, *, execute: bool = False, confirm: str | None = None):
    return cleanup_module.CleanupRequest(
        experiment_id=EXPERIMENT_ID,
        confirm_prefix=confirm,
        execute=execute,
    )


def _factory(control: Any) -> RepoFactory:
    return RepoFactory(control.aim.repo)


def test_dry_run_is_canonical_read_only_closed_and_performs_zero_writes() -> None:
    cleanup_module = _load()
    control = load_facility_control(CONTROL)
    s3 = FakeS3()
    factory = _factory(control)

    report = cleanup_module.cleanup(
        _request(cleanup_module),
        control=control,
        s3=s3,
        repo_factory=factory,
    )

    assert report.expected_prefix == PREFIX
    assert report.s3_keys == (REPORT_KEY, f"{KEY_PREFIX}runs/run-1.json")
    assert report.aim_run_hashes == ("exact-a", "exact-b")
    assert report.writes_performed is False
    assert s3.deleted == []
    assert factory.state.deleted == []
    assert factory.calls == [(str(control.aim.repo), True, False)]
    assert [repo.closed for repo in factory.repos] == [1]
    rendered = cleanup_module.canonical_json(report)
    assert rendered == json.dumps(json.loads(rendered), separators=(",", ":"), sort_keys=True)


@pytest.mark.parametrize(
    "experiment_id",
    ["", "/", "abc/def", "..", "../escape", f"{EXPERIMENT_ID}/"],
)
def test_cleanup_rejects_noncanonical_or_traversing_ids(experiment_id: str) -> None:
    cleanup_module = _load()
    control = load_facility_control(CONTROL)
    with pytest.raises(ValueError, match="canonical UUID"):
        cleanup_module.cleanup(
            cleanup_module.CleanupRequest(experiment_id, None, False),
            control=control,
            s3=FakeS3(),
            repo_factory=_factory(control),
        )


@pytest.mark.parametrize(
    "report",
    [
        {},
        {"experiment_name": "other", "experiment_metadata": {"purpose": "infra-acceptance"}},
        {
            "experiment_name": "infra-brax-ppo-acceptance",
            "experiment_metadata": {"purpose": "other"},
        },
    ],
)
def test_cleanup_rejects_missing_or_mismatched_canonical_report(
    report: dict[str, Any],
) -> None:
    cleanup_module = _load()
    control = load_facility_control(CONTROL)
    with pytest.raises(ValueError, match="canonical report"):
        cleanup_module.cleanup(
            _request(cleanup_module),
            control=control,
            s3=FakeS3(report=report),
            repo_factory=_factory(control),
        )


def test_control_cannot_self_authorize_a_different_aim_repo() -> None:
    cleanup_module = _load()
    control = load_facility_control(CONTROL)
    forged = control.model_copy(
        update={
            "aim": control.aim.model_copy(
                update={
                    "repo": Path("/tmp/forged-scratch"),
                    "main_repo": Path("/tmp/forged-main"),
                }
            )
        }
    )
    factory = RepoFactory(forged.aim.repo)

    with pytest.raises(ValueError, match="audited acceptance"):
        cleanup_module.cleanup(
            _request(cleanup_module),
            control=forged,
            s3=FakeS3(),
            repo_factory=factory,
        )
    assert factory.calls == []


@pytest.mark.parametrize("relation", ["same", "ancestor", "descendant", "aim-ancestor"])
def test_aim_trust_rejects_lexical_and_dot_aim_overlap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relation: str,
) -> None:
    cleanup_module = _load()
    control = load_facility_control(CONTROL)
    if relation == "same":
        scratch = main = tmp_path / "repo"
    elif relation == "ancestor":
        scratch, main = tmp_path / "repo", tmp_path / "repo" / "main"
    elif relation == "descendant":
        main, scratch = tmp_path / "repo", tmp_path / "repo" / "scratch"
    else:
        main, scratch = tmp_path / "main", tmp_path / ".aim" / "scratch"
    monkeypatch.setattr(cleanup_module, "ACCEPTANCE_AIM_SCRATCH", scratch)
    monkeypatch.setattr(cleanup_module, "ACCEPTANCE_MAIN_REPO", main)
    forged = control.model_copy(
        update={"aim": control.aim.model_copy(update={"repo": scratch, "main_repo": main})}
    )

    with pytest.raises(ValueError, match="overlap"):
        cleanup_module.cleanup(
            _request(cleanup_module),
            control=forged,
            s3=FakeS3(),
            repo_factory=RepoFactory(scratch),
        )


def test_aim_trust_rejects_symlink_resolved_overlap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cleanup_module = _load()
    control = load_facility_control(CONTROL)
    main = tmp_path / "main"
    main.mkdir()
    nested = main / "nested"
    nested.mkdir()
    scratch = tmp_path / "scratch-link"
    scratch.symlink_to(nested, target_is_directory=True)
    monkeypatch.setattr(cleanup_module, "ACCEPTANCE_AIM_SCRATCH", scratch)
    monkeypatch.setattr(cleanup_module, "ACCEPTANCE_MAIN_REPO", main)
    forged = control.model_copy(
        update={"aim": control.aim.model_copy(update={"repo": scratch, "main_repo": main})}
    )

    with pytest.raises(ValueError, match="overlap"):
        cleanup_module.cleanup(
            _request(cleanup_module),
            control=forged,
            s3=FakeS3(),
            repo_factory=RepoFactory(scratch),
        )


def test_s3_pagination_accepts_multiple_pages_and_empty_middle_page() -> None:
    cleanup_module = _load()
    control = load_facility_control(CONTROL)
    s3 = PagedS3(
        {
            None: {
                "Contents": [{"Key": f"{KEY_PREFIX}z.json"}],
                "IsTruncated": True,
                "NextContinuationToken": "a",
            },
            "a": {
                "Contents": [],
                "IsTruncated": True,
                "NextContinuationToken": "b",
            },
            "b": {
                "Contents": [{"Key": REPORT_KEY}],
                "IsTruncated": False,
            },
        }
    )

    assert cleanup_module._list_s3_keys(s3, control, EXPERIMENT_ID) == (
        REPORT_KEY,
        f"{KEY_PREFIX}z.json",
    )
    assert s3.calls == [None, "a", "b"]


def test_s3_pagination_rejects_repeated_or_cyclic_continuation_token() -> None:
    cleanup_module = _load()
    control = load_facility_control(CONTROL)
    s3 = PagedS3(
        {
            None: {"Contents": [], "IsTruncated": True, "NextContinuationToken": "a"},
            "a": {"Contents": [], "IsTruncated": True, "NextContinuationToken": "a"},
        }
    )

    with pytest.raises(ValueError, match="continuation token"):
        cleanup_module._list_s3_keys(s3, control, EXPERIMENT_ID)
    assert s3.calls == [None, "a"]


def test_execute_refuses_changed_snapshot_before_any_write() -> None:
    cleanup_module = _load()
    control = load_facility_control(CONTROL)
    s3 = FakeS3(change_on_list=2)
    factory = _factory(control)

    with pytest.raises(RuntimeError, match="changed"):
        cleanup_module.cleanup(
            _request(cleanup_module, execute=True, confirm=PREFIX),
            control=control,
            s3=s3,
            repo_factory=factory,
        )
    assert s3.deleted == []
    assert factory.state.deleted == []
    assert all(read_only for _path, read_only, _init in factory.calls)
    assert all(repo.closed == 1 for repo in factory.repos)


def test_nonreport_s3_partial_failure_preserves_report_and_retry_succeeds() -> None:
    cleanup_module = _load()
    control = load_facility_control(CONTROL)
    s3 = FakeS3(
        keys=[REPORT_KEY, f"{KEY_PREFIX}a.json", f"{KEY_PREFIX}b.json"],
        fail_nonreport_once=True,
    )
    factory = _factory(control)
    request = _request(cleanup_module, execute=True, confirm=PREFIX)

    with pytest.raises(RuntimeError, match="S3 deletion"):
        cleanup_module.cleanup(request, control=control, s3=s3, repo_factory=factory)
    assert REPORT_KEY in s3.keys
    assert factory.state.deleted == []

    report = cleanup_module.cleanup(
        request,
        control=control,
        s3=s3,
        repo_factory=factory,
    )
    assert report.writes_performed is True
    assert s3.keys == set()


def test_aim_partial_failure_preserves_report_and_retry_uses_remaining_hashes() -> None:
    cleanup_module = _load()
    control = load_facility_control(CONTROL)
    s3 = FakeS3()
    factory = _factory(control)
    factory.state.fail_hash_once = "exact-b"
    request = _request(cleanup_module, execute=True, confirm=PREFIX)

    with pytest.raises(RuntimeError, match="Aim refused"):
        cleanup_module.cleanup(request, control=control, s3=s3, repo_factory=factory)
    assert REPORT_KEY in s3.keys
    assert factory.state.deleted == ["exact-a"]

    cleanup_module.cleanup(request, control=control, s3=s3, repo_factory=factory)
    assert s3.keys == set()
    assert set(factory.state.runs) == {"other"}


def test_final_report_delete_failure_preserves_sentinel_and_is_safely_retryable() -> None:
    cleanup_module = _load()
    control = load_facility_control(CONTROL)
    s3 = FakeS3(fail_report_once=True)
    factory = _factory(control)
    request = _request(cleanup_module, execute=True, confirm=PREFIX)

    with pytest.raises(RuntimeError, match="S3 deletion"):
        cleanup_module.cleanup(request, control=control, s3=s3, repo_factory=factory)
    assert s3.keys == {REPORT_KEY}
    assert set(factory.state.runs) == {"other"}

    cleanup_module.cleanup(request, control=control, s3=s3, repo_factory=factory)
    assert s3.keys == set()


def test_execute_uses_short_writable_repo_only_after_stable_read_only_snapshots() -> None:
    cleanup_module = _load()
    control = load_facility_control(CONTROL)
    factory = _factory(control)
    s3 = FakeS3()

    cleanup_module.cleanup(
        _request(cleanup_module, execute=True, confirm=PREFIX),
        control=control,
        s3=s3,
        repo_factory=factory,
    )

    assert factory.calls == [
        (str(control.aim.repo), True, False),
        (str(control.aim.repo), True, False),
        (str(control.aim.repo), False, False),
        (str(control.aim.repo), True, False),
    ]
    assert all(repo.closed == 1 for repo in factory.repos)
    assert s3.delete_batches == [
        (f"{KEY_PREFIX}runs/run-1.json",),
        (REPORT_KEY,),
    ]


def test_real_aim_read_only_open_does_not_change_repository_files(tmp_path: Path) -> None:
    from aim import Run

    cleanup_module = _load()
    repo_path = tmp_path / "aim-scratch"
    run = Run(repo=str(repo_path), experiment="infra")
    run["context"] = {"experiment_id": EXPERIMENT_ID}
    run.close()

    def snapshot() -> dict[str, tuple[bytes, int]]:
        return {
            str(path.relative_to(repo_path)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in repo_path.rglob("*")
            if path.is_file()
        }

    before = snapshot()
    repo = cleanup_module._default_repo_factory(
        str(repo_path),
        read_only=True,
        init=False,
    )
    try:
        assert [item.hash for item in repo.iter_runs()]
    finally:
        repo.close()
    assert snapshot() == before


def test_cleanup_source_exposes_no_shared_resource_or_recursive_deletion() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in (
        "delete_bucket",
        "delete_repository",
        "deregister_job_definition",
        "delete_job_queue",
        "delete_compute_environment",
        "shutil.rmtree",
        "query_runs",
    ):
        assert forbidden not in source


def test_cli_error_writes_nothing_to_canonical_stdout(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_module = _load()

    class Session:
        def __init__(self, *, region_name: str) -> None:
            assert region_name == "eu-north-1"

        def client(self, name: str) -> FakeS3:
            assert name == "s3"
            return FakeS3()

    monkeypatch.setattr(cleanup_module.boto3, "Session", Session)
    with pytest.raises(ValueError, match="canonical UUID"):
        cleanup_module.main(
            [
                "--control",
                str(CONTROL),
                "--experiment-id",
                "invalid",
            ]
        )
    assert capsys.readouterr().out == ""
