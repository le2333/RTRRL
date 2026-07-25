import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from training_sdk import objects

from trainer_infra.backends.base import JobResult
from trainer_infra.backends.local import LocalBackend
from trainer_infra.launch import build_run_config, config_uri, create_launch
from tests.helpers import EXAMPLE, make_plan

WHEN = datetime(2026, 7, 25, 5, 14, tzinfo=UTC)
ENTRY = "brax_ppo_acceptance"


def write_catalog(tmp_path: Path, body: str) -> Path:
    child = tmp_path / "child.py"
    child.write_text(body, encoding="utf-8")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "contract": 2,
                "entries": {
                    ENTRY: {
                        "command": [sys.executable, str(child)],
                        "source_hash": "sha256:0",
                        "metrics": ["m"],
                        "space": {"total_steps": [1]},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return catalog


def publish_sleeping_manifest(s3_base: str, tmp_path: Path) -> str:
    """A manifest the worker can actually start, whose child never finishes."""
    launch = create_launch(make_plan(s3_base), tmp_path / "archive", EXAMPLE, WHEN)
    config = build_run_config(launch, 0, {"total_steps": 1, "learning_rate": 1e-4})
    objects.put_bytes(config_uri(launch, 0), config.model_dump_json().encode())
    manifest = f"{launch.prefix}/rounds/round-000/job-0.json"
    objects.put_bytes(manifest, json.dumps({"runs": [config_uri(launch, 0)]}).encode())
    return manifest


def _is_sleep_process(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        return False
    return "sleep" in cmdline and "600" in cmdline


def _wait_until(predicate, *, timeout: float = 5.0, interval: float = 0.05) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("condition not met before timeout")


def test_failed_worker_is_reported_with_a_readable_log(
    tmp_path: Path, s3_base: str
) -> None:
    backend = LocalBackend(tmp_path, write_catalog(tmp_path, "import sys; sys.exit(3)"))
    job_id = backend.submit_raw(f"{s3_base}/missing-manifest.json", "job-0")
    results = backend.wait([job_id])
    assert len(results) == 1 and results[0].succeeded is False
    returncode = backend._processes[job_id].returncode
    assert results[0].reason == f"exit code {returncode}"
    assert "worker failed" in backend.log_tail(results[0], 50)


def test_successful_worker_is_reported(tmp_path: Path, s3_base: str) -> None:
    ok_child = """
import json, os
scratch = os.environ["TRAINER_SCRATCH"]
with open(os.path.join(scratch, "metrics.jsonl"), "a") as handle:
    for step in (0, 4):
        handle.write(json.dumps({"step": step, "metrics": {"episode_return": 2.0}}) + "\\n")
        handle.flush()
"""
    backend = LocalBackend(tmp_path, write_catalog(tmp_path, ok_child))
    manifest = publish_sleeping_manifest(s3_base, tmp_path)
    job_id = backend.submit_raw(manifest, "job-ok")
    results = backend.wait([job_id])
    assert len(results) == 1 and results[0].succeeded is True
    assert results[0].reason is None


def test_log_tail_returns_only_the_last_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "job-noisy" / "worker.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("\n".join(f"line-{i}" for i in range(100)) + "\n", encoding="utf-8")
    backend = LocalBackend(tmp_path, write_catalog(tmp_path, "pass"))
    result = JobResult(
        job_id="local-job-noisy-0",
        name="job-noisy",
        succeeded=False,
        log_stream=str(log_path),
    )
    tail = backend.log_tail(result, 5).splitlines()
    assert len(tail) == 5
    assert tail[-1] == "line-99"
    assert tail[0] == "line-95"
    assert "line-0" not in tail


def test_terminate_stops_a_running_job(tmp_path: Path, s3_base: str) -> None:
    pid_file = tmp_path / "grandchild.pid"
    sleeping_child = f"""
import json, os, subprocess, time
from pathlib import Path
scratch = os.environ["TRAINER_SCRATCH"]
with open(os.path.join(scratch, "metrics.jsonl"), "a") as handle:
    handle.write(json.dumps({{"step": 0, "metrics": {{"episode_return": 2.0}}}}) + "\\n")
    handle.flush()
proc = subprocess.Popen(["sleep", "600"])
Path("{pid_file}").write_text(str(proc.pid))
time.sleep(600)
"""
    backend = LocalBackend(tmp_path, write_catalog(tmp_path, sleeping_child))
    job_id = backend.submit_raw(publish_sleeping_manifest(s3_base, tmp_path), "job-0")

    def grandchild_is_running() -> bool:
        if not pid_file.exists():
            return False
        grandchild_pid = int(pid_file.read_text(encoding="utf-8"))
        return _is_sleep_process(grandchild_pid)

    _wait_until(grandchild_is_running)
    backend.terminate([job_id])
    results = backend.wait([job_id])
    assert results[0].succeeded is False
    grandchild_pid = int(pid_file.read_text(encoding="utf-8"))
    assert not _is_sleep_process(grandchild_pid)
