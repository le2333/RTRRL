from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace
from typing import Any, get_type_hints

import pytest

import trainer_infra.aim_scratch as aim_scratch
from trainer_infra.facility_control import AimScratchControl


ProcessSnapshot = aim_scratch.ProcessSnapshot
launch_aim_scratch = aim_scratch.launch_aim_scratch
validate_aim_scratch = aim_scratch.validate_aim_scratch


START_SCRIPT = Path(__file__).parents[1] / "scripts" / "start_facility_aim.py"
CONTROL = Path(__file__).parents[1] / "config" / "facility.yaml"


def _control(tmp_path: Path) -> AimScratchControl:
    repo = tmp_path / "scratch"
    repo.mkdir()
    (repo / ".aim").mkdir()
    main = tmp_path / "main"
    main.mkdir()
    return AimScratchControl(
        repo=repo,
        main_repo=main,
        host="127.0.0.1",
        port=53801,
        metadata_file=repo / "aim-server-53801.json",
        pid_file=repo / "aim-server-53801.pid",
        log_file=repo / "aim-server-53801.log",
    )


def _recorded(
    control: AimScratchControl,
    *,
    pid: int = 321,
    start_time_ticks: int = 12345,
) -> tuple[str, ...]:
    command = (
        "/venv/bin/aim",
        "server",
        "--host",
        control.host,
        "--port",
        str(control.port),
        "--repo",
        str(control.repo.resolve()),
        "-y",
    )
    control.metadata_file.write_text(
        json.dumps(
            {
                "command": list(command),
                "cwd": str(control.repo.resolve()),
                "endpoint": control.endpoint,
                "pid": pid,
                "port": control.port,
                "repo": str(control.repo.resolve()),
                "start_time_ticks": start_time_ticks,
                "started_at_utc": "2026-07-23T18:00:00Z",
            }
        )
    )
    control.pid_file.write_text(f"{pid}\n")
    return command


def test_launch_writes_pid_and_reproducible_runtime_metadata(tmp_path: Path) -> None:
    control = _control(tmp_path)
    calls: list[dict[str, Any]] = []

    def popen(command: list[str], **kwargs: Any) -> Any:
        calls.append({"command": command, **kwargs})
        return SimpleNamespace(pid=321)

    result = launch_aim_scratch(
        control,
        aim_executable="/venv/bin/aim",
        popen=popen,
        health_probe=lambda _host, _port: True,
        discover_process=lambda _control: ProcessSnapshot(
            pid=321,
            cmdline=(
                "/venv/bin/python",
                "/venv/bin/aim",
                "server",
                "--host",
                "127.0.0.1",
                "--port",
                "53801",
                "--repo",
                str(control.repo),
                "-y",
            ),
            cwd=control.repo,
            start_time_ticks=12345,
        ),
        now=lambda: datetime(2026, 7, 23, 18, 0, tzinfo=timezone.utc),
    )

    metadata = json.loads(control.metadata_file.read_text())
    assert result["status"] == "started"
    assert control.pid_file.read_text() == "321\n"
    assert metadata == {
        "command": [
            "/venv/bin/python",
            "/venv/bin/aim",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            "53801",
            "--repo",
            str(control.repo),
            "-y",
        ],
        "cwd": str(control.repo),
        "endpoint": "aim://127.0.0.1:53801",
        "pid": 321,
        "port": 53801,
        "repo": str(control.repo),
        "start_time_ticks": 12345,
        "started_at_utc": "2026-07-23T18:00:00Z",
    }
    assert calls[0]["cwd"] == control.repo
    assert calls[0]["start_new_session"] is True


def test_launch_resumes_valid_recorded_process_without_duplicate(
    tmp_path: Path,
) -> None:
    control = _control(tmp_path)
    control.metadata_file.write_text('{"pid":321}')
    control.pid_file.write_text("321\n")
    popen_calls: list[object] = []

    result = launch_aim_scratch(
        control,
        aim_executable="/venv/bin/aim",
        popen=lambda *_args, **_kwargs: popen_calls.append(object()),
        runtime_validator=lambda _control: {
            "pid": 321,
            "status": "ready",
        },
    )

    assert result == {"pid": 321, "status": "resumed"}
    assert popen_calls == []


def test_preflight_validates_metadata_pid_cmdline_cwd_port_and_repo(
    tmp_path: Path,
) -> None:
    control = _control(tmp_path)
    metadata = {
        "command": [
            "/venv/bin/aim",
            "server",
            "--host",
            control.host,
            "--port",
            str(control.port),
            "--repo",
            str(control.repo),
            "-y",
        ],
        "cwd": str(control.repo),
        "endpoint": control.endpoint,
        "pid": 321,
        "port": control.port,
        "repo": str(control.repo),
        "start_time_ticks": 12345,
        "started_at_utc": "2026-07-23T18:00:00Z",
    }
    control.metadata_file.write_text(json.dumps(metadata))
    control.pid_file.write_text("321\n")

    result = validate_aim_scratch(
        control,
        inspect_process=lambda _pid: ProcessSnapshot(
            pid=321,
            cmdline=tuple(metadata["command"]),
            cwd=control.repo,
            start_time_ticks=12345,
        ),
        health_probe=lambda host, port: host == "127.0.0.1" and port == 53801,
    )

    assert result["status"] == "ready"
    assert result["pid"] == 321
    assert result["cmdline_matches"] is True
    assert result["cwd_matches"] is True
    assert result["repo_matches"] is True
    assert result["port_matches"] is True


@pytest.mark.parametrize("drift", ["cmdline", "cwd", "pid"])
def test_preflight_rejects_aim_runtime_drift(tmp_path: Path, drift: str) -> None:
    control = _control(tmp_path)
    command = (
        "/venv/bin/aim",
        "server",
        "--host",
        control.host,
        "--port",
        str(control.port),
        "--repo",
        str(control.repo),
        "-y",
    )
    control.metadata_file.write_text(
        json.dumps(
            {
                "command": list(command),
                "cwd": str(control.repo),
                "endpoint": control.endpoint,
                "pid": 321,
                "port": control.port,
                "repo": str(control.repo),
                "start_time_ticks": 12345,
                "started_at_utc": "2026-07-23T18:00:00Z",
            }
        )
    )
    control.pid_file.write_text("999\n" if drift == "pid" else "321\n")
    snapshot = ProcessSnapshot(
        pid=321,
        cmdline=("aim", "server", "--repo", "/wrong")
        if drift == "cmdline"
        else command,
        cwd=control.main_repo if drift == "cwd" else control.repo,
        start_time_ticks=12345,
    )

    with pytest.raises(ValueError, match=drift):
        validate_aim_scratch(
            control,
            inspect_process=lambda _pid: snapshot,
            health_probe=lambda _host, _port: True,
        )


def test_inactive_accepts_stale_files_but_rejects_live_or_occupied_endpoint(
    tmp_path: Path,
) -> None:
    control = _control(tmp_path)
    command = _recorded(control)
    stale = aim_scratch.assert_aim_scratch_inactive(
        control,
        process_enumerator=lambda: (),
        health_probe=lambda _host, _port: False,
    )
    assert stale["status"] == "inactive"
    assert control.metadata_file.exists()
    assert control.pid_file.exists()

    with pytest.raises(ValueError, match="active"):
        aim_scratch.assert_aim_scratch_inactive(
            control,
            process_enumerator=lambda: (
                ProcessSnapshot(321, command, control.repo, 12345),
            ),
            health_probe=lambda _host, _port: True,
        )
    metadata = json.loads(control.metadata_file.read_text())
    metadata["endpoint"] = "stale"
    control.metadata_file.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="active"):
        aim_scratch.assert_aim_scratch_inactive(
            control,
            process_enumerator=lambda: (
                ProcessSnapshot(321, command, control.repo, 12345),
            ),
            health_probe=lambda _host, _port: False,
        )
    control.metadata_file.unlink()
    control.pid_file.unlink()
    with pytest.raises(ValueError, match="occupied"):
        aim_scratch.assert_aim_scratch_inactive(
            control,
            health_probe=lambda _host, _port: True,
        )


def test_stop_validates_exact_identity_then_terms_only_recorded_process(
    tmp_path: Path,
) -> None:
    control = _control(tmp_path)
    command = _recorded(control)
    unrelated = control.repo / "unrelated"
    unrelated.write_text("keep")
    inspections = iter(
        [
            ProcessSnapshot(321, command, control.repo, 12345),
            ProcessSnapshot(321, command, control.repo, 12345),
            FileNotFoundError(),
            FileNotFoundError(),
        ]
    )
    signals: list[tuple[int, int]] = []
    health = iter([True, False])

    def inspect(_pid: int) -> ProcessSnapshot:
        result = next(inspections)
        if isinstance(result, BaseException):
            raise result
        return result

    report = aim_scratch.stop_aim_scratch(
        control,
        inspect_process=inspect,
        health_probe=lambda _host, _port: next(health),
        send_signal=lambda pid, value: signals.append((pid, value)),
        monotonic=iter([0.0, 0.1]).__next__,
        sleep=lambda _seconds: None,
    )

    assert report == {"pid": 321, "status": "stopped"}
    assert signals == [(321, signal.SIGTERM)]
    assert not control.metadata_file.exists()
    assert not control.pid_file.exists()
    assert unrelated.read_text() == "keep"


@pytest.mark.parametrize("drift", ["cmdline", "cwd", "repo", "port", "endpoint"])
def test_stop_rejects_wrong_identity_without_signaling(
    tmp_path: Path,
    drift: str,
) -> None:
    control = _control(tmp_path)
    command = list(_recorded(control))
    metadata = json.loads(control.metadata_file.read_text())
    snapshot = ProcessSnapshot(321, tuple(command), control.repo, 12345)
    if drift == "cmdline":
        snapshot = ProcessSnapshot(321, ("other",), control.repo, 12345)
    elif drift == "cwd":
        snapshot = ProcessSnapshot(321, tuple(command), control.main_repo, 12345)
    elif drift == "repo":
        metadata["repo"] = str(control.main_repo)
    elif drift == "port":
        metadata["port"] = 1
    else:
        metadata["endpoint"] = "aim://127.0.0.1:1"
    control.metadata_file.write_text(json.dumps(metadata))
    signals: list[tuple[int, int]] = []

    with pytest.raises(ValueError, match="mismatch"):
        aim_scratch.stop_aim_scratch(
            control,
            inspect_process=lambda _pid: snapshot,
            health_probe=lambda _host, _port: True,
            send_signal=lambda pid, value: signals.append((pid, value)),
        )
    assert signals == []
    assert control.metadata_file.exists()
    assert control.pid_file.exists()


def test_stop_rejects_stale_pid_and_pid_reuse_without_other_signal(
    tmp_path: Path,
) -> None:
    control = _control(tmp_path)
    command = _recorded(control)
    signals: list[tuple[int, int]] = []
    with pytest.raises(ValueError, match="stale"):
        aim_scratch.stop_aim_scratch(
            control,
            inspect_process=lambda _pid: (_ for _ in ()).throw(FileNotFoundError()),
            health_probe=lambda _host, _port: False,
            send_signal=lambda pid, value: signals.append((pid, value)),
        )
    assert signals == []

    inspections = iter(
        [
            ProcessSnapshot(321, command, control.repo, 12345),
            ProcessSnapshot(321, command, control.repo, 12345),
            ProcessSnapshot(321, command, control.repo, 99999),
        ]
    )
    with pytest.raises(RuntimeError, match="PID reuse"):
        aim_scratch.stop_aim_scratch(
            control,
            inspect_process=lambda _pid: next(inspections),
            health_probe=lambda _host, _port: True,
            send_signal=lambda pid, value: signals.append((pid, value)),
            monotonic=iter([0.0, 0.1]).__next__,
            sleep=lambda _seconds: None,
        )
    assert signals == [(321, signal.SIGTERM)]
    assert control.metadata_file.exists()
    assert control.pid_file.exists()


def test_stop_timeout_never_sigkills_or_removes_evidence(tmp_path: Path) -> None:
    control = _control(tmp_path)
    command = _recorded(control)
    signals: list[tuple[int, int]] = []
    times = iter([0.0, 0.1, 2.0])

    with pytest.raises(TimeoutError, match="SIGTERM"):
        aim_scratch.stop_aim_scratch(
            control,
            inspect_process=lambda pid: ProcessSnapshot(
                pid,
                command,
                control.repo,
                12345,
            ),
            health_probe=lambda _host, _port: True,
            send_signal=lambda pid, value: signals.append((pid, value)),
            timeout=1.0,
            monotonic=times.__next__,
            sleep=lambda _seconds: None,
        )
    assert signals == [(321, signal.SIGTERM)]
    assert signal.SIGKILL not in [value for _pid, value in signals]
    assert control.metadata_file.exists()
    assert control.pid_file.exists()


def test_stop_rechecks_generation_immediately_before_signal(tmp_path: Path) -> None:
    control = _control(tmp_path)
    command = _recorded(control)
    inspections = iter(
        [
            ProcessSnapshot(321, command, control.repo, 12345),
            ProcessSnapshot(321, command, control.repo, 99999),
        ]
    )
    signals: list[tuple[int, int]] = []

    with pytest.raises(RuntimeError, match="before SIGTERM"):
        aim_scratch.stop_aim_scratch(
            control,
            inspect_process=lambda _pid: next(inspections),
            health_probe=lambda _host, _port: True,
            send_signal=lambda pid, value: signals.append((pid, value)),
        )
    assert signals == []
    assert control.metadata_file.exists()
    assert control.pid_file.exists()


def test_start_script_stop_flag_calls_only_validated_stop(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = importlib.util.spec_from_file_location("start_facility_aim_test", START_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls: list[Path] = []
    monkeypatch.setattr(
        module,
        "stop_aim_scratch",
        lambda control: calls.append(control.repo) or {"pid": 321, "status": "stopped"},
    )
    monkeypatch.setattr(
        module,
        "launch_aim_scratch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not start")),
    )

    assert module.main(["--control", str(CONTROL), "--stop"]) == 0
    assert calls == [Path("/home/ubuntu/trainer/task7-aim-scratch")]
    assert json.loads(capsys.readouterr().out) == {"pid": 321, "status": "stopped"}


def test_process_snapshot_records_proc_start_generation() -> None:
    assert get_type_hints(ProcessSnapshot)["start_time_ticks"] is int
    snapshot = aim_scratch.inspect_process(os.getpid())
    stat_text = Path("/proc/self/stat").read_text()
    fields_after_command = stat_text[stat_text.rfind(")") + 2 :].split()
    assert snapshot.start_time_ticks == int(fields_after_command[19])


def test_inactive_scans_all_processes_without_metadata_or_endpoint_health(
    tmp_path: Path,
) -> None:
    control = _control(tmp_path)
    command = (
        "python",
        "server",
        "--repo",
        ".",
        "--port",
        "9999",
    )

    with pytest.raises(ValueError, match="active"):
        aim_scratch.assert_aim_scratch_inactive(
            control,
            process_enumerator=lambda: (
                ProcessSnapshot(777, command, control.repo, 12345),
            ),
            health_probe=lambda _host, _port: False,
        )


def test_validate_fails_closed_when_generation_metadata_is_missing(
    tmp_path: Path,
) -> None:
    control = _control(tmp_path)
    command = _recorded(control)
    metadata = json.loads(control.metadata_file.read_text())
    del metadata["start_time_ticks"]
    control.metadata_file.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="start_time_ticks"):
        validate_aim_scratch(
            control,
            inspect_process=lambda _pid: ProcessSnapshot(
                321,
                command,
                control.repo,
                12345,
            ),
            health_probe=lambda _host, _port: True,
        )
