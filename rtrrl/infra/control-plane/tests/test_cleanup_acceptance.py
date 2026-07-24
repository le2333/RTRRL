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
PREFIX = (
    "s3://rtrrl-artifacts-007122174918/experiments/"
    f"{EXPERIMENT_ID}/"
)
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
    ) -> None:
        self.keys = set(
            keys
            or [
                f"experiments/{EXPERIMENT_ID}/report.json",
                f"experiments/{EXPERIMENT_ID}/runs/run-1.json",
            ]
        )
        self.report = deepcopy(REPORT if report is None else report)
        self.change_on_list = change_on_list
        self.list_calls = 0
        self.deleted: list[str] = []

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        assert kwargs == {
            "Bucket": "rtrrl-artifacts-007122174918",
            "Prefix": f"experiments/{EXPERIMENT_ID}/",
        }
        self.list_calls += 1
        if self.change_on_list == self.list_calls:
            self.keys.add(f"experiments/{EXPERIMENT_ID}/late.json")
        return {
            "Contents": [{"Key": key} for key in sorted(self.keys)],
            "IsTruncated": False,
        }

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        assert kwargs == {
            "Bucket": "rtrrl-artifacts-007122174918",
            "Key": f"experiments/{EXPERIMENT_ID}/report.json",
        }
        if kwargs["Key"] not in self.keys:
            raise KeyError(kwargs["Key"])
        return {"Body": BytesIO(json.dumps(self.report).encode("utf-8"))}

    def delete_object(self, **kwargs: Any) -> dict[str, Any]:
        key = kwargs["Key"]
        assert kwargs["Bucket"] == "rtrrl-artifacts-007122174918"
        assert key in self.keys
        self.keys.remove(key)
        self.deleted.append(key)
        return {}


class FakeRun:
    def __init__(self, run_hash: str, experiment_id: str) -> None:
        self.hash = run_hash
        self._context = {"experiment_id": experiment_id}

    def get(self, key: str, default: object = None) -> object:
        return self._context if key == "context" else default


class FakeAimRepo:
    def __init__(self, path: Path, runs: list[FakeRun]) -> None:
        self.path = path
        self.runs = {run.hash: run for run in runs}
        self.deleted: list[str] = []
        self.iter_calls = 0

    def iter_runs(self):
        self.iter_calls += 1
        return iter(list(self.runs.values()))

    def delete_run(self, run_hash: str) -> bool:
        assert run_hash in self.runs
        del self.runs[run_hash]
        self.deleted.append(run_hash)
        return True


def _request(cleanup_module: Any, *, execute: bool = False, confirm: str | None = None):
    return cleanup_module.CleanupRequest(
        experiment_id=EXPERIMENT_ID,
        confirm_prefix=confirm,
        execute=execute,
    )


def _repo(control: Any) -> FakeAimRepo:
    return FakeAimRepo(
        control.aim.repo,
        [
            FakeRun("exact-b", EXPERIMENT_ID),
            FakeRun("other", str(uuid.uuid4())),
            FakeRun("exact-a", EXPERIMENT_ID),
        ],
    )


def test_dry_run_is_canonical_deterministic_and_performs_zero_writes() -> None:
    cleanup_module = _load()
    control = load_facility_control(CONTROL)
    s3 = FakeS3()
    aim_repo = _repo(control)

    report = cleanup_module.cleanup(
        _request(cleanup_module),
        control=control,
        s3=s3,
        aim_repo=aim_repo,
    )

    assert report.expected_prefix == PREFIX
    assert report.s3_keys == (
        f"experiments/{EXPERIMENT_ID}/report.json",
        f"experiments/{EXPERIMENT_ID}/runs/run-1.json",
    )
    assert report.aim_run_hashes == ("exact-a", "exact-b")
    assert report.writes_performed is False
    assert s3.deleted == []
    assert aim_repo.deleted == []
    first = cleanup_module.canonical_json(report)
    second = cleanup_module.canonical_json(report)
    assert first == second
    assert first == json.dumps(json.loads(first), separators=(",", ":"), sort_keys=True)


@pytest.mark.parametrize(
    "experiment_id",
    ["", "/", "abc/def", "..", "../escape", "12345678-1234-5678-9234-567812345678/"],
)
def test_cleanup_rejects_noncanonical_or_traversing_ids(experiment_id: str) -> None:
    cleanup_module = _load()
    control = load_facility_control(CONTROL)
    with pytest.raises(ValueError, match="canonical UUID"):
        cleanup_module.cleanup(
            cleanup_module.CleanupRequest(experiment_id, None, False),
            control=control,
            s3=FakeS3(),
            aim_repo=_repo(control),
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
            aim_repo=_repo(control),
        )


def test_cleanup_rejects_missing_report_and_main_aim_repo() -> None:
    cleanup_module = _load()
    control = load_facility_control(CONTROL)
    report_key = f"experiments/{EXPERIMENT_ID}/report.json"
    with pytest.raises(ValueError, match="canonical report"):
        cleanup_module.cleanup(
            _request(cleanup_module),
            control=control,
            s3=FakeS3(keys=[f"experiments/{EXPERIMENT_ID}/artifact.json"]),
            aim_repo=_repo(control),
        )
    with pytest.raises(ValueError, match="main Aim"):
        cleanup_module.cleanup(
            _request(cleanup_module),
            control=control,
            s3=FakeS3(keys=[report_key]),
            aim_repo=FakeAimRepo(control.aim.main_repo, []),
        )


def test_execute_requires_exact_prefix_and_refuses_changed_snapshot() -> None:
    cleanup_module = _load()
    control = load_facility_control(CONTROL)
    with pytest.raises(ValueError, match="confirm_prefix"):
        cleanup_module.cleanup(
            _request(cleanup_module, execute=True, confirm=PREFIX.rstrip("/")),
            control=control,
            s3=FakeS3(),
            aim_repo=_repo(control),
        )
    s3 = FakeS3(change_on_list=2)
    aim_repo = _repo(control)
    with pytest.raises(RuntimeError, match="changed"):
        cleanup_module.cleanup(
            _request(cleanup_module, execute=True, confirm=PREFIX),
            control=control,
            s3=s3,
            aim_repo=aim_repo,
        )
    assert s3.deleted == []
    assert aim_repo.deleted == []


def test_execute_deletes_only_exact_keys_and_hashes_then_postverifies_empty() -> None:
    cleanup_module = _load()
    control = load_facility_control(CONTROL)
    s3 = FakeS3()
    aim_repo = _repo(control)

    report = cleanup_module.cleanup(
        _request(cleanup_module, execute=True, confirm=PREFIX),
        control=control,
        s3=s3,
        aim_repo=aim_repo,
    )

    assert report.writes_performed is True
    assert s3.deleted == list(report.s3_keys)
    assert aim_repo.deleted == list(report.aim_run_hashes)
    assert set(aim_repo.runs) == {"other"}
    assert s3.list_calls == 3
    assert aim_repo.iter_calls == 3


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


def test_real_aim_repo_can_enumerate_context_without_querying_another_repo(
    tmp_path: Path,
) -> None:
    from aim import Repo, Run

    cleanup_module = _load()
    control = load_facility_control(CONTROL)
    repo_path = tmp_path / "isolated-aim"
    run = Run(repo=str(repo_path), experiment="infra")
    run_hash = run.hash
    run["context"] = {"experiment_id": EXPERIMENT_ID}
    run.close()
    repo = Repo(str(repo_path))
    isolated_control = control.model_copy(
        update={
            "aim": control.aim.model_copy(
                update={"repo": repo_path, "main_repo": tmp_path / "main-aim"}
            )
        }
    )

    report = cleanup_module.cleanup(
        _request(cleanup_module),
        control=isolated_control,
        s3=FakeS3(),
        aim_repo=repo,
    )

    assert report.aim_run_hashes == (run_hash,)
