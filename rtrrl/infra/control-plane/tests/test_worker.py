from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from trainer_infra.execution import JobBundle, RunBundle
from trainer_infra.identities import canonical_json
from test_execution import IMAGE, make_run_bundle

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

    def get_bytes(self, uri: str, *, expected_sha256: str | None = None) -> bytes:
        data = self.objects[uri]
        if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ValueError("SHA-256 mismatch")
        return data

    def put_json(self, uri: str, value: Any) -> str:
        self.put_json_calls.append((uri, value))
        return hashlib.sha256(canonical_json(value).encode()).hexdigest()

    def put_file(self, uri: str, path: Path) -> str:
        data = path.read_bytes()
        self.put_file_calls.append((uri, data))
        return hashlib.sha256(data).hexdigest()


def make_worker_run(tmp_path: Path, number: int) -> RunBundle:
    base = make_run_bundle(resource_profile="g6x")
    run_id = f"experiment-123:shared:{number:04d}"
    context = base.model_dump(mode="json")["run_context"]
    context["run_id"] = run_id
    context["artifact_directory"] = str(tmp_path / f"artifacts-{number}")
    config_path = tmp_path / f"config-{number}.yaml"
    return RunBundle(
        run_id=run_id,
        argv=("python", "train.py", "--config", str(config_path)),
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

    def run_child(
        argv: list[str], *, env: dict[str, str], shell: bool, check: bool
    ) -> subprocess.CompletedProcess[str]:
        calls.append((argv, env, shell))
        assert check is False
        context_path = Path(env["TRAINER_RUN_CONTEXT_PATH"])
        context = json.loads(context_path.read_text())
        run_number = context["run_number"]
        artifact_directory = Path(context["artifact_directory"])
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
    assert [call[0] for call in calls] == [list(run.argv) for run in runs]
    assert all(shell is False for _, _, shell in calls)
    assert all("TRAINER_RUN_CONTEXT_PATH" in env for _, env, _ in calls)
    assert [Path(run.argv[-1]).read_text() for run in runs] == [run.config_yaml for run in runs]
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
    assert calls == [list(runs[0].argv)]
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
