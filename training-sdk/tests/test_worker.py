import json
import os
import sys
import time
from pathlib import Path

import pytest

from training_sdk import objects
from training_sdk.worker import WorkerError, _Heartbeat, main, run_manifest
from tests.test_reporter import make_config

CHILD = """
import json, os, sys, time
config = json.loads(open(os.environ["TRAINER_RUN_CONFIG"]).read())
scratch = os.environ["TRAINER_SCRATCH"]
mode = os.environ.get("CHILD_MODE", "ok")
if mode == "crash":
    sys.exit(3)
if mode == "grandchild":
    import subprocess
    proc = subprocess.Popen(["sleep", "600"])
    with open(os.environ["GRANDCHILD_PID_FILE"], "w") as handle:
        handle.write(str(proc.pid))
    with open(os.path.join(scratch, "metrics.jsonl"), "a") as handle:
        handle.write(json.dumps({"step": 0, "metrics": {"episode_return": 2.0}}) + "\\n")
        handle.flush()
    time.sleep(600)
with open(os.path.join(scratch, "metrics.jsonl"), "a") as handle:
    for step in (0, 4):
        handle.write(json.dumps({"step": step, "metrics": {"episode_return": 2.0}}) + "\\n")
        handle.flush()
        if mode == "slow":
            time.sleep(0.5)
if mode == "hang":
    time.sleep(600)
if mode == "pause_after_first":
    with open(os.path.join(scratch, "metrics.jsonl"), "a") as handle:
        handle.write(json.dumps({"step": 0, "metrics": {"episode_return": 2.0}}) + "\\n")
        handle.flush()
    time.sleep(3)
    with open(os.path.join(scratch, "metrics.jsonl"), "a") as handle:
        handle.write(json.dumps({"step": 4, "metrics": {"episode_return": 2.0}}) + "\\n")
        handle.flush()
if mode == "empty_metrics":
    open(os.path.join(scratch, "metrics.jsonl"), "w").close()
"""


@pytest.fixture
def catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    child = tmp_path / "child.py"
    child.write_text(CHILD, encoding="utf-8")
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {
                "contract": 2,
                "entries": {
                    "e": {
                        "command": [sys.executable, str(child)],
                        "source_hash": "sha256:0",
                        "metrics": ["episode_return"],
                        "space": {"total_steps": [4]},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRAINER_CATALOG", str(path))
    return path


def publish(s3_base: str, trial: int, *, entry: str = "e") -> str:
    config = make_config().model_copy(
        update={
            "trial": trial,
            "run_id": f"smoke-20260725-000000-t{trial}",
            "entry": entry,
            "score": make_config().score.model_copy(
                update={"s3": f"{s3_base}/trials/t{trial}/score.json"}
            ),
        }
    )
    uri = f"{s3_base}/trials/t{trial}/config.json"
    objects.put_bytes(uri, config.model_dump_json().encode())
    return uri


def write_manifest(s3_base: str, uris: list[str]) -> str:
    manifest = f"{s3_base}/rounds/round-000/job-0.json"
    objects.put_bytes(manifest, json.dumps({"runs": uris}).encode())
    return manifest


def test_every_run_is_executed_and_scored(s3_base: str, tmp_path: Path, catalog: Path) -> None:
    manifest = write_manifest(s3_base, [publish(s3_base, 0), publish(s3_base, 1)])
    run_manifest(manifest, tmp_path, startup_seconds=60, stall_factor=10)
    for trial in (0, 1):
        payload = json.loads(objects.get_bytes(f"{s3_base}/trials/t{trial}/score.json"))
        assert payload["value"] == 2.0
        assert payload["run_id"] == f"smoke-20260725-000000-t{trial}"


def test_scratch_is_removed_between_runs(s3_base: str, tmp_path: Path, catalog: Path) -> None:
    manifest = write_manifest(s3_base, [publish(s3_base, 0), publish(s3_base, 1)])
    run_manifest(manifest, tmp_path, startup_seconds=60, stall_factor=10)
    assert list(tmp_path.glob("*/metrics.jsonl")) == []


def test_crashing_run_stops_the_manifest(
    s3_base: str, tmp_path: Path, catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHILD_MODE", "crash")
    manifest = write_manifest(
        s3_base, [publish(s3_base, 100), publish(s3_base, 101)]
    )
    with pytest.raises(WorkerError, match="exit code 3"):
        run_manifest(manifest, tmp_path, startup_seconds=60, stall_factor=10)
    assert objects.exists(f"{s3_base}/trials/t100/score.json") is False
    assert objects.exists(f"{s3_base}/trials/t101/score.json") is False


def test_stalled_run_is_killed(
    s3_base: str, tmp_path: Path, catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHILD_MODE", "hang")
    manifest = write_manifest(s3_base, [publish(s3_base, 0)])
    with pytest.raises(WorkerError, match="stalled"):
        run_manifest(
            manifest,
            tmp_path,
            startup_seconds=1.0,
            stall_factor=1,
            poll_seconds=0.05,
        )


def test_startup_grace_survives_long_pause_after_first_report(
    s3_base: str, tmp_path: Path, catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHILD_MODE", "pause_after_first")
    manifest = write_manifest(s3_base, [publish(s3_base, 0)])
    run_manifest(
        manifest,
        tmp_path,
        startup_seconds=30,
        stall_factor=10,
        poll_seconds=0.2,
    )
    payload = json.loads(objects.get_bytes(f"{s3_base}/trials/t0/score.json"))
    assert payload["value"] == 2.0


def test_slow_healthy_run_is_not_killed(
    s3_base: str, tmp_path: Path, catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHILD_MODE", "slow")
    manifest = write_manifest(s3_base, [publish(s3_base, 0)])
    run_manifest(
        manifest,
        tmp_path,
        startup_seconds=1.0,
        stall_factor=1,
        poll_seconds=0.05,
    )
    payload = json.loads(objects.get_bytes(f"{s3_base}/trials/t0/score.json"))
    assert payload["value"] == 2.0


def test_heartbeat_limit_holds_startup_grace_after_first_report(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text('{"step": 0}\n', encoding="utf-8")
    watcher = _Heartbeat(metrics, startup_seconds=900, stall_factor=10)
    watcher._poll()
    assert watcher.limit() == 900


def test_the_gap_inside_one_epoch_does_not_become_the_expected_cadence(
    tmp_path: Path,
) -> None:
    # An epoch reports its training metrics and then, seconds later, its
    # evaluation metrics; the next epoch is minutes away. Reading that burst as
    # the run's cadence is what killed a healthy two million step run after its
    # first epoch.
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text('{"step": 0}\n', encoding="utf-8")
    watcher = _Heartbeat(metrics, startup_seconds=900, stall_factor=10)
    watcher._poll()
    time.sleep(0.2)
    metrics.write_text('{"step": 0}\n{"step": 0}\n', encoding="utf-8")
    watcher._poll()
    assert watcher.limit() == 900


def test_a_reporter_slower_than_the_grace_period_raises_the_limit(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text('{"step": 0}\n', encoding="utf-8")
    watcher = _Heartbeat(metrics, startup_seconds=0.1, stall_factor=10)
    watcher._poll()
    time.sleep(0.2)
    metrics.write_text('{"step": 0}\n{"step": 1}\n', encoding="utf-8")
    watcher._poll()
    assert watcher.limit() > 1.0


def test_the_widest_silence_sets_the_limit_not_the_typical_one(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text("0\n", encoding="utf-8")
    watcher = _Heartbeat(metrics, startup_seconds=0.01, stall_factor=10)
    watcher._poll()
    for index, pause in enumerate((0.02, 0.4, 0.02), start=1):
        time.sleep(pause)
        metrics.write_text(f"{index}\n", encoding="utf-8")
        watcher._poll()
    # Three quick reports and one long pause is one run, and the pause is the
    # part that says how long silence may legitimately last.
    assert watcher.limit() > 3.0


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


def test_kill_terminates_grandchild_processes(
    s3_base: str,
    tmp_path: Path,
    catalog: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid_file = tmp_path / "grandchild.pid"
    monkeypatch.setenv("CHILD_MODE", "grandchild")
    monkeypatch.setenv("GRANDCHILD_PID_FILE", str(pid_file))
    manifest = write_manifest(s3_base, [publish(s3_base, 0)])
    with pytest.raises(WorkerError, match="stalled"):
        run_manifest(
            manifest,
            tmp_path,
            startup_seconds=1.0,
            stall_factor=1,
            poll_seconds=0.05,
        )
    grandchild_pid = int(pid_file.read_text(encoding="utf-8"))
    assert not _is_sleep_process(grandchild_pid)


def test_catalog_contract_mismatch(
    s3_base: str, tmp_path: Path, catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog.write_text(
        json.dumps(
            {
                "contract": 1,
                "entries": {
                    "e": {
                        "command": ["true"],
                        "source_hash": "sha256:0",
                        "metrics": ["episode_return"],
                        "space": {"total_steps": [4]},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = write_manifest(s3_base, [publish(s3_base, 0)])
    with pytest.raises(WorkerError, match="image catalog declares contract 1"):
        run_manifest(manifest, tmp_path, startup_seconds=60, stall_factor=10)


def test_run_config_contract_mismatch(
    s3_base: str, tmp_path: Path, catalog: Path
) -> None:
    config = make_config().model_copy(
        update={
            "contract": 1,
            "score": make_config().score.model_copy(
                update={"s3": f"{s3_base}/trials/t0/score.json"}
            ),
        }
    )
    uri = f"{s3_base}/trials/t0/config.json"
    objects.put_bytes(uri, config.model_dump_json().encode())
    manifest = write_manifest(s3_base, [uri])
    with pytest.raises(WorkerError, match="run configuration declares contract 1"):
        run_manifest(manifest, tmp_path, startup_seconds=60, stall_factor=10)


def test_missing_catalog_entry(s3_base: str, tmp_path: Path, catalog: Path) -> None:
    manifest = write_manifest(s3_base, [publish(s3_base, 0, entry="missing")])
    with pytest.raises(WorkerError, match="image catalog does not declare entry 'missing'"):
        run_manifest(manifest, tmp_path, startup_seconds=60, stall_factor=10)


def test_score_computation_failure_stops_manifest(
    s3_base: str, tmp_path: Path, catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from training_sdk.score import ScoreError

    monkeypatch.setenv("CHILD_MODE", "empty_metrics")
    manifest = write_manifest(
        s3_base, [publish(s3_base, 200), publish(s3_base, 201)]
    )
    with pytest.raises(ScoreError, match="no reported value for metric"):
        run_manifest(manifest, tmp_path, startup_seconds=60, stall_factor=10)
    assert objects.exists(f"{s3_base}/trials/t200/score.json") is False
    assert objects.exists(f"{s3_base}/trials/t201/score.json") is False


def test_main_reports_a_missing_manifest_variable(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The image's default command is this entry point, so it must explain itself.

    A Batch job started without the manifest is misconfigured by the control
    plane, and the only trace left behind is what lands in CloudWatch.
    """
    monkeypatch.delenv("TRAINER_MANIFEST", raising=False)

    assert main() == 1
    assert "TRAINER_MANIFEST is not set" in capsys.readouterr().err
