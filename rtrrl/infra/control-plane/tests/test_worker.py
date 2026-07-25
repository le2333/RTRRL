from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from training_sdk.execution import JobBundle, RunBundle, canonical_json
from tests.test_execution import IMAGE, make_run_bundle

WORKER_PATH = Path(__file__).parents[2] / "worker" / "worker.py"
SPEC = importlib.util.spec_from_file_location("trainer_worker", WORKER_PATH)
assert SPEC is not None and SPEC.loader is not None
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)

BUNDLE_URI = "s3://bucket/experiments/exp-1/jobs/bundle-1/bundle.json"


class FakeStore:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.put_json_calls: list[tuple[str, Any]] = []
        self.put_file_calls: list[tuple[str, bytes]] = []
        self.put_file_error: Exception | None = None
        self.put_file_fail_at = 1
        self.put_json_error: Exception | None = None

    def get_bytes(self, uri: str, *, expected_sha256: str | None = None) -> bytes:
        data = self.objects[uri]
        if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ValueError("SHA-256 mismatch")
        return data

    def put_json(self, uri: str, value: Any) -> str:
        if self.put_json_error is not None:
            raise self.put_json_error
        self.put_json_calls.append((uri, value))
        return hashlib.sha256(canonical_json(value).encode()).hexdigest()

    def put_file(self, uri: str, path: Path) -> str:
        if (
            self.put_file_error is not None
            and len(self.put_file_calls) + 1 == self.put_file_fail_at
        ):
            raise self.put_file_error
        data = path.read_bytes()
        self.put_file_calls.append((uri, data))
        return hashlib.sha256(data).hexdigest()


def make_worker_run(tmp_path: Path, number: int) -> RunBundle:
    base = make_run_bundle(resource_profile="g6x")
    run_id = f"experiment-123:shared:{number:04d}"
    context = base.model_dump(mode="json")["run_context"]
    context["run_id"] = run_id
    context["artifact_directory"] = str(tmp_path / f"artifacts-{number}")
    return RunBundle(
        run_id=run_id,
        argv=("python", "train.py", "--config", "{config_path}"),
        image_digest=IMAGE,
        resource_profile="g6x",
        config_yaml=base.config_yaml,
        config_sha256=base.config_sha256,
        run_context=context,
        run_context_sha256=hashlib.sha256(canonical_json(context).encode()).hexdigest(),
        artifact_prefix=f"experiments/exp-1/groups/shared/runs/run-{number}/input/",
    )


def make_store(tmp_path: Path, runs: tuple[RunBundle, ...]) -> tuple[FakeStore, JobBundle]:
    bundle = JobBundle(
        job_id="bundle-1",
        image_digest=IMAGE,
        resource_profile="g6x",
        runs=runs,
    )
    objects = {BUNDLE_URI: bundle.to_json().encode()}
    for run in runs:
        base = f"s3://bucket/{run.artifact_prefix}"
        objects[f"{base}config.yaml"] = run.config_yaml.encode()
        objects[f"{base}run-context.json"] = canonical_json(run.run_context).encode()
    return FakeStore(objects), bundle


def test_worker_verifies_inputs_injects_context_and_uploads_artifacts(
    tmp_path: Path,
) -> None:
    runs = (make_worker_run(tmp_path, 1), make_worker_run(tmp_path, 2))
    store, _ = make_store(tmp_path, runs)
    calls: list[tuple[list[str], dict[str, str], bool]] = []
    temporary_paths: list[Path] = []
    runtime_artifact_paths: list[Path] = []

    def run_child(
        argv: list[str], *, env: dict[str, str], shell: bool, check: bool
    ) -> subprocess.CompletedProcess[str]:
        calls.append((argv, env, shell))
        assert check is False
        context_path = Path(env["TRAINER_RUN_CONTEXT_PATH"])
        config_path = Path(argv[-1])
        temporary_paths.extend((context_path, config_path))
        assert config_path.read_text() == runs[len(calls) - 1].config_yaml
        context = json.loads(context_path.read_text())
        run_number = context["run_number"]
        artifact_directory = Path(context["artifact_directory"])
        runtime_artifact_paths.append(artifact_directory)
        assert artifact_directory != Path(
            runs[len(calls) - 1].run_context["artifact_directory"]
        )
        (artifact_directory / "aim-buffer").mkdir(parents=True)
        (artifact_directory / "rerun").mkdir()
        (artifact_directory / "checkpoints").mkdir()
        (artifact_directory / "aim-buffer" / "events.log").write_text(f"aim-{run_number}")
        (artifact_directory / "rerun" / "eval.rrd").write_text(f"rerun-{run_number}")
        (artifact_directory / "checkpoints" / "latest").write_text(f"ckpt-{run_number}")
        return subprocess.CompletedProcess(argv, 0)

    result = worker.execute_bundle(
        BUNDLE_URI,
        store,
        run_command=run_child,
        now=lambda: datetime(2026, 7, 21, 8, 0, tzinfo=timezone.utc),
    )

    assert result == 0
    assert all(call[0][-1] != "{config_path}" for call in calls)
    assert all(shell is False for _, _, shell in calls)
    assert all("TRAINER_RUN_CONTEXT_PATH" in env for _, env, _ in calls)
    assert all(not path.exists() for path in temporary_paths)
    assert all(not path.exists() for path in runtime_artifact_paths)
    assert len(store.put_json_calls) == 2
    assert all(uri.endswith("/status/attempt-0.json") for uri, _ in store.put_json_calls)
    assert [value["exit_code"] for _, value in store.put_json_calls] == [0, 0]
    assert [uri for uri, _ in store.put_file_calls] == sorted(
        uri for uri, _ in store.put_file_calls
    )
    assert {uri.rsplit("/", 2)[-2] for uri, _ in store.put_file_calls} >= {
        "aim-buffer",
        "rerun",
        "checkpoints",
    }


def test_worker_stops_after_first_nonzero_child_and_uploads_failed_marker(
    tmp_path: Path,
) -> None:
    runs = (make_worker_run(tmp_path, 1), make_worker_run(tmp_path, 2))
    store, _ = make_store(tmp_path, runs)
    calls: list[list[str]] = []

    def fail_first(
        argv: list[str], *, env: dict[str, str], shell: bool, check: bool
    ) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 23)

    assert worker.execute_bundle(BUNDLE_URI, store, run_command=fail_first) == 23
    assert len(calls) == 1
    assert len(store.put_json_calls) == 1
    assert store.put_json_calls[0][1]["exit_code"] == 23


def test_worker_rejects_tampered_input_before_starting_child(tmp_path: Path) -> None:
    run = make_worker_run(tmp_path, 1)
    store, _ = make_store(tmp_path, (run,))
    store.objects[f"s3://bucket/{run.artifact_prefix}config.yaml"] = b"tampered"
    called = False

    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal called
        called = True

    with pytest.raises(ValueError, match="SHA-256"):
        worker.execute_bundle(BUNDLE_URI, store, run_command=unexpected)
    assert called is False
    assert store.put_json_calls == []


def test_worker_rejects_noncanonical_bundle_transport(tmp_path: Path) -> None:
    run = make_worker_run(tmp_path, 1)
    store, bundle = make_store(tmp_path, (run,))
    store.objects[BUNDLE_URI] = json.dumps(bundle.model_dump(mode="json"), indent=2).encode()

    with pytest.raises(ValueError, match="canonical"):
        worker.execute_bundle(BUNDLE_URI, store)


def test_artifact_upload_failure_writes_failed_marker_and_stops(tmp_path: Path) -> None:
    runs = (make_worker_run(tmp_path, 1), make_worker_run(tmp_path, 2))
    store, _ = make_store(tmp_path, runs)
    store.put_file_error = RuntimeError("artifact AWS failure")
    calls = 0

    def child(
        argv: list[str], *, env: dict[str, str], shell: bool, check: bool
    ) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        context = json.loads(Path(env["TRAINER_RUN_CONTEXT_PATH"]).read_text())
        artifact = Path(context["artifact_directory"]) / "rerun"
        artifact.mkdir(parents=True)
        (artifact / "eval.rrd").write_bytes(b"artifact")
        return subprocess.CompletedProcess(argv, 0)

    error = store.put_file_error
    with pytest.raises(RuntimeError) as caught:
        worker.execute_bundle(BUNDLE_URI, store, run_command=child)
    assert caught.value is error
    assert calls == 1
    marker = store.put_json_calls[0][1]
    assert marker["exit_code"] == 0
    assert "artifact AWS failure" in marker["error"]


def test_marker_failure_after_artifact_failure_remains_visible(tmp_path: Path) -> None:
    run = make_worker_run(tmp_path, 1)
    store, _ = make_store(tmp_path, (run,))
    store.put_file_error = RuntimeError("artifact failed")
    marker_error = RuntimeError("marker AWS failure")
    store.put_json_error = marker_error

    def child(
        argv: list[str], *, env: dict[str, str], shell: bool, check: bool
    ) -> subprocess.CompletedProcess[str]:
        context = json.loads(Path(env["TRAINER_RUN_CONTEXT_PATH"]).read_text())
        artifact = Path(context["artifact_directory"]) / "rerun"
        artifact.mkdir(parents=True)
        (artifact / "eval.rrd").write_bytes(b"artifact")
        return subprocess.CompletedProcess(argv, 7)

    with pytest.raises(RuntimeError) as caught:
        worker.execute_bundle(BUNDLE_URI, store, run_command=child)
    assert caught.value is store.put_file_error
    assert caught.value.__cause__ is marker_error


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_worker_rejects_symlink_and_nonregular_artifacts(
    tmp_path: Path, kind: str
) -> None:
    run = make_worker_run(tmp_path, 1)
    store, _ = make_store(tmp_path, (run,))

    def child(
        argv: list[str], *, env: dict[str, str], shell: bool, check: bool
    ) -> subprocess.CompletedProcess[str]:
        context = json.loads(Path(env["TRAINER_RUN_CONTEXT_PATH"]).read_text())
        artifact = Path(context["artifact_directory"]) / "checkpoints"
        artifact.mkdir(parents=True)
        path = artifact / "unsafe"
        if kind == "symlink":
            path.symlink_to(tmp_path / "outside")
        else:
            os.mkfifo(path)
        return subprocess.CompletedProcess(argv, 0)

    with pytest.raises((OSError, ValueError)):
        worker.execute_bundle(BUNDLE_URI, store, run_command=child)
    assert "artifact" in store.put_json_calls[0][1]["error"]


def test_second_artifact_failure_marker_keeps_first_uploaded_uri(tmp_path: Path) -> None:
    run = make_worker_run(tmp_path, 1)
    store, _ = make_store(tmp_path, (run,))
    error = RuntimeError("second upload failed")
    store.put_file_error = error
    store.put_file_fail_at = 2

    def child(
        argv: list[str], *, env: dict[str, str], shell: bool, check: bool
    ) -> subprocess.CompletedProcess[str]:
        context = json.loads(Path(env["TRAINER_RUN_CONTEXT_PATH"]).read_text())
        artifact = Path(context["artifact_directory"]) / "rerun"
        artifact.mkdir(parents=True)
        (artifact / "a.rrd").write_bytes(b"first")
        (artifact / "b.rrd").write_bytes(b"second")
        return subprocess.CompletedProcess(argv, 0)

    with pytest.raises(RuntimeError) as caught:
        worker.execute_bundle(BUNDLE_URI, store, run_command=child)
    assert caught.value is error
    assert len(store.put_file_calls) == 1
    marker = store.put_json_calls[0][1]
    assert marker["artifacts"] == [store.put_file_calls[0][0]]
    assert "second upload failed" in marker["error"]


def test_artifact_replaced_by_symlink_before_open_is_never_uploaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = make_worker_run(tmp_path, 1)
    store, _ = make_store(tmp_path, (run,))
    outside = tmp_path / "outside"
    outside.write_bytes(b"secret")
    original_open = worker.os.open
    replaced = False

    def racing_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal replaced
        if (
            path == "victim"
            and flags & worker.os.O_NOFOLLOW
            and not flags & worker.os.O_DIRECTORY
            and not replaced
        ):
            parent_fd = kwargs["dir_fd"]
            worker.os.unlink(path, dir_fd=parent_fd)
            worker.os.symlink(str(outside), path, dir_fd=parent_fd)
            replaced = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(worker.os, "open", racing_open)

    def child(
        argv: list[str], *, env: dict[str, str], shell: bool, check: bool
    ) -> subprocess.CompletedProcess[str]:
        context = json.loads(Path(env["TRAINER_RUN_CONTEXT_PATH"]).read_text())
        artifact = Path(context["artifact_directory"]) / "rerun"
        artifact.mkdir(parents=True)
        (artifact / "victim").write_bytes(b"safe")
        return subprocess.CompletedProcess(argv, 0)

    with pytest.raises(OSError):
        worker.execute_bundle(BUNDLE_URI, store, run_command=child)
    assert replaced is True
    assert store.put_file_calls == []
    assert "artifact upload failed" in store.put_json_calls[0][1]["error"]


def test_worker_imports_with_sdk_only_and_trainer_infra_blocked() -> None:
    sdk_src = Path(__file__).parents[4] / "training-sdk" / "src"
    code = f"""
import importlib.abc, importlib.util, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'trainer_infra' or fullname.startswith('trainer_infra.'):
            raise ImportError('trainer_infra is forbidden')
        return None
sys.meta_path.insert(0, Block())
sys.path.insert(0, {str(sdk_src)!r})
spec = importlib.util.spec_from_file_location('isolated_worker', {str(WORKER_PATH)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert 'trainer_infra' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    "uri",
    [
        "s3://bucket/other/experiments/exp-1/jobs/bundle-1/bundle.json",
        "s3://bucket/experiments/exp-1/jobs/../bundle.json",
        BUNDLE_URI + "?version=x",
        BUNDLE_URI + "#fragment",
    ],
)
def test_worker_rejects_noncanonical_bundle_uri(tmp_path: Path, uri: str) -> None:
    run = make_worker_run(tmp_path, 1)
    store, _ = make_store(tmp_path, (run,))
    with pytest.raises(ValueError, match="bundle URI|S3 URI"):
        worker.execute_bundle(uri, store)


def test_worker_rejects_run_input_prefix_outside_bundle_namespace(tmp_path: Path) -> None:
    run = make_worker_run(tmp_path, 1).model_copy(
        update={"artifact_prefix": "experiments/other/groups/g/runs/r/input/"}
    )
    store, _ = make_store(tmp_path, (run,))
    with pytest.raises(ValueError, match="prefix"):
        worker.execute_bundle(BUNDLE_URI, store)
