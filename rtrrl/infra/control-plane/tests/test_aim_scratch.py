from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, get_type_hints

import pytest

import trainer_infra.aim_scratch as aim_scratch
from trainer_infra.facility_control import AimScratchControl


ProcessSnapshot = aim_scratch.ProcessSnapshot
launch_aim_scratch = aim_scratch.launch_aim_scratch
validate_aim_scratch = aim_scratch.validate_aim_scratch
REAL_PROCESS_REPO_IDENTITY = aim_scratch._process_repo_identity


START_SCRIPT = Path(__file__).parents[1] / "scripts" / "start_facility_aim.py"
CONTROL = Path(__file__).parents[1] / "config" / "facility.yaml"


@pytest.fixture(autouse=True)
def _resolve_fake_process_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    def identity(snapshot: ProcessSnapshot, _argument: str) -> tuple[int, int]:
        file_stat = snapshot.cwd.stat()
        return file_stat.st_dev, file_stat.st_ino

    monkeypatch.setattr(aim_scratch, "_process_repo_identity", identity)


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


def _gated_target(call: dict[str, Any]) -> tuple[str, ...]:
    command = call["command"]
    return tuple(command[command.index("--") + 1 :])


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
        "/proc/self/fd/99",
        "-y",
    )
    root_stat = control.repo.stat()
    control.metadata_file.write_text(
        json.dumps(
            {
                "command": list(command),
                "cwd": str(control.repo.resolve()),
                "endpoint": control.endpoint,
                "pid": pid,
                "port": control.port,
                "repo_fd": 99,
                "repo_root_dev": root_stat.st_dev,
                "repo_root_ino": root_stat.st_ino,
                "start_time_ticks": start_time_ticks,
                "started_at_utc": "2026-07-23T18:00:00Z",
                "trusted_repo": str(control.repo),
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
        discover_process=lambda _control, **_kwargs: ProcessSnapshot(
            pid=321,
            cmdline=_gated_target(calls[0]),
            cwd=control.repo,
            start_time_ticks=12345,
        ),
        now=lambda: datetime(2026, 7, 23, 18, 0, tzinfo=timezone.utc),
        pidfd_open=lambda _pid: 91,
        pidfd_send_signal=lambda _fd, _signal: None,
        wait_pidfd=lambda _fd, _timeout: True,
        close_fd=lambda _fd: None,
    )

    metadata = json.loads(control.metadata_file.read_text())
    assert result["status"] == "started"
    assert control.pid_file.read_text() == "321\n"
    assert metadata == {
        "command": list(_gated_target(calls[0])),
        "cwd": str(control.repo),
        "endpoint": "aim://127.0.0.1:53801",
        "pid": 321,
        "port": 53801,
        "repo_fd": calls[0]["pass_fds"][2],
        "repo_root_dev": control.repo.stat().st_dev,
        "repo_root_ino": control.repo.stat().st_ino,
        "start_time_ticks": 12345,
        "started_at_utc": "2026-07-23T18:00:00Z",
        "trusted_repo": str(control.repo),
    }
    assert str(calls[0]["cwd"]).startswith("/proc/self/fd/")
    assert calls[0]["start_new_session"] is True
    assert len(calls[0]["pass_fds"]) == 3
    assert (control.repo / ".trainer-aim-scratch.lock").is_file()


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
        runtime_validator=lambda _control, **_kwargs: {
            "pid": 321,
            "status": "ready",
        },
    )

    assert result == {"pid": 321, "status": "resumed"}
    assert popen_calls == []


def test_launch_rejects_exclusive_cleanup_lock_without_starting(
    tmp_path: Path,
) -> None:
    control = _control(tmp_path)
    directory_fd = aim_scratch.open_trusted_directory(control.repo)
    lock_fd = aim_scratch.create_facility_lock(directory_fd)
    starts: list[object] = []
    try:
        with pytest.raises(ValueError, match="lock"):
            launch_aim_scratch(
                control,
                aim_executable="/venv/bin/aim",
                popen=lambda *_args, **_kwargs: starts.append(object()),
            )
    finally:
        os.close(lock_fd)
        os.close(directory_fd)
    assert starts == []


def test_launch_health_timeout_pidfd_terminates_child_before_releasing_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _control(tmp_path)
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(
        aim_scratch.time,
        "monotonic",
        iter([0.0, 11.0]).__next__,
    )
    with pytest.raises(TimeoutError, match="healthy"):
        launch_aim_scratch(
            control,
            aim_executable="/venv/bin/aim",
            popen=lambda *_args, **_kwargs: SimpleNamespace(pid=321),
            health_probe=lambda _host, _port: False,
            sleep=lambda _seconds: None,
            pidfd_open=lambda _pid: 91,
            pidfd_send_signal=lambda fd, value: sent.append((fd, value)),
            wait_pidfd=lambda _fd, _timeout: True,
            close_fd=lambda _fd: None,
        )
    assert sent == [(91, signal.SIGTERM)]
    directory_fd = aim_scratch.open_trusted_directory(control.repo)
    lock_fd = aim_scratch.open_facility_lock(directory_fd)
    os.close(lock_fd)
    os.close(directory_fd)


def test_real_gate_pidfd_open_failure_leaves_no_child_or_lock(
    tmp_path: Path,
) -> None:
    control = _control(tmp_path)
    processes: list[subprocess.Popen[bytes]] = []

    def popen(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        process = subprocess.Popen(*args, **kwargs)
        processes.append(process)
        return process

    with pytest.raises(RuntimeError, match="pidfd"):
        launch_aim_scratch(
            control,
            aim_executable=sys.executable,
            popen=popen,
            pidfd_open=lambda _pid: (_ for _ in ()).throw(OSError("injected")),
        )
    assert processes[0].poll() == 0
    directory_fd = aim_scratch.open_trusted_directory(control.repo)
    lock_fd = aim_scratch.open_facility_lock(directory_fd)
    os.close(lock_fd)
    os.close(directory_fd)


def test_launch_keeps_pinned_inode_when_lexical_repo_is_swapped(
    tmp_path: Path,
) -> None:
    control = _control(tmp_path)
    calls: list[dict[str, Any]] = []
    moved = tmp_path / "original"

    def popen(command: list[str], **kwargs: Any) -> Any:
        calls.append({"command": command, **kwargs})
        return SimpleNamespace(pid=321)

    def health(_host: str, _port: int) -> bool:
        if control.repo.exists():
            control.repo.rename(moved)
            control.repo.mkdir()
        return True

    result = launch_aim_scratch(
        control,
        aim_executable="/venv/bin/aim",
        popen=popen,
        health_probe=health,
        discover_process=lambda _control, **_kwargs: ProcessSnapshot(
            321,
            _gated_target(calls[0]),
            moved,
            12345,
        ),
        pidfd_open=lambda _pid: 91,
        pidfd_send_signal=lambda _fd, _value: None,
        wait_pidfd=lambda _fd, _timeout: True,
        close_fd=lambda _fd: None,
    )

    assert result["repo_root_ino"] == moved.stat().st_ino
    assert (moved / control.metadata_file.name).is_file()
    assert not (control.repo / control.metadata_file.name).exists()


def test_cleanup_lock_open_never_creates_missing_file(tmp_path: Path) -> None:
    control = _control(tmp_path)
    directory_fd = aim_scratch.open_trusted_directory(control.repo)
    before = tuple(sorted(path.name for path in control.repo.iterdir()))
    try:
        with pytest.raises(ValueError, match="existing"):
            aim_scratch.open_facility_lock(directory_fd)
    finally:
        os.close(directory_fd)
    assert tuple(sorted(path.name for path in control.repo.iterdir())) == before


def test_real_child_inherits_exclusive_flock_until_exit(tmp_path: Path) -> None:
    control = _control(tmp_path)
    directory_fd = aim_scratch.open_trusted_directory(control.repo)
    lock_fd = aim_scratch.create_facility_lock(directory_fd)
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        pass_fds=(lock_fd,),
    )
    os.close(lock_fd)
    try:
        with pytest.raises(ValueError, match="held"):
            aim_scratch.open_facility_lock(directory_fd)
    finally:
        child.terminate()
        child.wait(timeout=5)
    reacquired = aim_scratch.open_facility_lock(directory_fd)
    os.close(reacquired)
    os.close(directory_fd)


def test_process_self_fd_argument_is_resolved_in_target_pid_namespace(
    tmp_path: Path,
) -> None:
    control = _control(tmp_path)
    directory_fd = aim_scratch.open_trusted_directory(control.repo)
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        pass_fds=(directory_fd,),
    )
    try:
        snapshot = ProcessSnapshot(child.pid, ("python",), control.repo, 1)
        assert REAL_PROCESS_REPO_IDENTITY(
            snapshot,
            f"/proc/self/fd/{directory_fd}",
        ) == (control.repo.stat().st_dev, control.repo.stat().st_ino)
    finally:
        child.terminate()
        child.wait(timeout=5)
        os.close(directory_fd)


def test_real_locked_child_validates_and_resumes_without_second_start(
    tmp_path: Path,
) -> None:
    control = _control(tmp_path)
    directory_fd = aim_scratch.open_trusted_directory(control.repo)
    lock_fd = aim_scratch.create_facility_lock(directory_fd)
    pinned = f"/proc/self/fd/{directory_fd}"
    command = [
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
        "server",
        "--host",
        control.host,
        "--port",
        str(control.port),
        "--repo",
        pinned,
        "-y",
    ]
    child = subprocess.Popen(
        command,
        cwd=pinned,
        pass_fds=(lock_fd, directory_fd),
    )
    os.close(lock_fd)
    root_stat = control.repo.stat()
    try:
        snapshot = aim_scratch.inspect_process(child.pid)
        metadata = {
            "command": list(snapshot.cmdline),
            "cwd": str(snapshot.cwd),
            "endpoint": control.endpoint,
            "pid": child.pid,
            "port": control.port,
            "repo_fd": directory_fd,
            "repo_root_dev": root_stat.st_dev,
            "repo_root_ino": root_stat.st_ino,
            "start_time_ticks": snapshot.start_time_ticks,
            "started_at_utc": "2026-07-24T00:00:00Z",
            "trusted_repo": str(control.repo),
        }
        control.metadata_file.write_text(json.dumps(metadata))
        control.pid_file.write_text(f"{child.pid}\n")
        result = launch_aim_scratch(
            control,
            aim_executable="/must/not/start",
            popen=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not start")),
            health_probe=lambda _host, _port: True,
        )
        assert result["status"] == "resumed"
        assert result["pid"] == child.pid
    finally:
        child.terminate()
        child.wait(timeout=5)
        os.close(directory_fd)


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
            "/proc/self/fd/99",
            "-y",
        ],
        "cwd": str(control.repo),
        "endpoint": control.endpoint,
        "pid": 321,
        "port": control.port,
        "repo_fd": 99,
        "repo_root_dev": control.repo.stat().st_dev,
        "repo_root_ino": control.repo.stat().st_ino,
        "start_time_ticks": 12345,
        "started_at_utc": "2026-07-23T18:00:00Z",
        "trusted_repo": str(control.repo),
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
        "/proc/self/fd/99",
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
                "repo_fd": 99,
                "repo_root_dev": control.repo.stat().st_dev,
                "repo_root_ino": control.repo.stat().st_ino,
                "start_time_ticks": 12345,
                "started_at_utc": "2026-07-23T18:00:00Z",
                "trusted_repo": str(control.repo),
            }
        )
    )
    control.pid_file.write_text("999\n" if drift == "pid" else "321\n")
    snapshot = ProcessSnapshot(
        pid=321,
        cmdline=("aim", "server", "--repo", "/wrong") if drift == "cmdline" else command,
        cwd=control.main_repo if drift == "cwd" else control.repo,
        start_time_ticks=12345,
    )

    expected_error = "cwd|inode" if drift == "cwd" else drift
    with pytest.raises(ValueError, match=expected_error):
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
            process_enumerator=lambda: (ProcessSnapshot(321, command, control.repo, 12345),),
            health_probe=lambda _host, _port: True,
        )
    metadata = json.loads(control.metadata_file.read_text())
    metadata["endpoint"] = "stale"
    control.metadata_file.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="active"):
        aim_scratch.assert_aim_scratch_inactive(
            control,
            process_enumerator=lambda: (ProcessSnapshot(321, command, control.repo, 12345),),
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
        pidfd_open=lambda _pid: 91,
        pidfd_send_signal=lambda fd, value: signals.append((fd, value)),
        wait_pidfd=lambda _fd, _timeout: True,
        close_fd=lambda _fd: None,
    )

    assert report == {"pid": 321, "status": "stopped"}
    assert signals == [(91, signal.SIGTERM)]
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
        metadata["repo_root_ino"] += 1
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
            pidfd_open=lambda _pid: 91,
            pidfd_send_signal=lambda fd, value: signals.append((fd, value)),
            wait_pidfd=lambda _fd, _timeout: True,
            close_fd=lambda _fd: None,
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
            pidfd_open=lambda _pid: 91,
            pidfd_send_signal=lambda fd, value: signals.append((fd, value)),
            wait_pidfd=lambda _fd, _timeout: True,
            close_fd=lambda _fd: None,
        )
    assert signals == []

    inspections = iter(
        [
            ProcessSnapshot(321, command, control.repo, 12345),
            ProcessSnapshot(321, command, control.repo, 99999),
        ]
    )
    with pytest.raises(RuntimeError, match="PID reuse"):
        aim_scratch.stop_aim_scratch(
            control,
            inspect_process=lambda _pid: next(inspections),
            health_probe=lambda _host, _port: True,
            pidfd_open=lambda _pid: 91,
            pidfd_send_signal=lambda fd, value: signals.append((fd, value)),
            wait_pidfd=lambda _fd, _timeout: True,
            close_fd=lambda _fd: None,
        )
    assert signals == []
    assert control.metadata_file.exists()
    assert control.pid_file.exists()


def test_stop_timeout_never_sigkills_or_removes_evidence(tmp_path: Path) -> None:
    control = _control(tmp_path)
    command = _recorded(control)
    signals: list[tuple[int, int]] = []
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
            pidfd_open=lambda _pid: 91,
            pidfd_send_signal=lambda fd, value: signals.append((fd, value)),
            wait_pidfd=lambda _fd, _timeout: False,
            close_fd=lambda _fd: None,
            timeout=1.0,
        )
    assert signals == [(91, signal.SIGTERM)]
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
            pidfd_open=lambda _pid: 91,
            pidfd_send_signal=lambda fd, value: signals.append((fd, value)),
            wait_pidfd=lambda _fd, _timeout: True,
            close_fd=lambda _fd: None,
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
            process_enumerator=lambda: (ProcessSnapshot(777, command, control.repo, 12345),),
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


def test_stop_uses_exact_pidfd_and_never_pid_signal(tmp_path: Path) -> None:
    control = _control(tmp_path)
    command = _recorded(control)
    sent: list[tuple[int, int]] = []
    closed: list[int] = []

    result = aim_scratch.stop_aim_scratch(
        control,
        inspect_process=lambda _pid: ProcessSnapshot(
            321,
            command,
            control.repo,
            12345,
        ),
        health_probe=lambda _host, _port, values=iter([True, False]): next(values),
        pidfd_open=lambda pid: 91 if pid == 321 else -1,
        pidfd_send_signal=lambda fd, value: sent.append((fd, value)),
        wait_pidfd=lambda fd, timeout: fd == 91 and timeout == 10.0,
        close_fd=lambda fd: closed.append(fd),
    )

    assert result == {"pid": 321, "status": "stopped"}
    assert sent == [(91, signal.SIGTERM)]
    assert closed == [91]


def test_stop_pidfd_generation_reuse_before_signal_never_signals(
    tmp_path: Path,
) -> None:
    control = _control(tmp_path)
    command = _recorded(control)
    snapshots = iter(
        [
            ProcessSnapshot(321, command, control.repo, 99999),
        ]
    )
    sent: list[tuple[int, int]] = []

    with pytest.raises(ValueError, match="start_time_ticks"):
        aim_scratch.stop_aim_scratch(
            control,
            inspect_process=lambda _pid: next(snapshots),
            health_probe=lambda _host, _port: True,
            pidfd_open=lambda _pid: 91,
            pidfd_send_signal=lambda fd, value: sent.append((fd, value)),
            wait_pidfd=lambda _fd, _timeout: True,
            close_fd=lambda _fd: None,
        )
    assert sent == []


def test_stop_fails_closed_without_pidfd_support_and_source_has_no_os_kill(
    tmp_path: Path,
) -> None:
    control = _control(tmp_path)
    _recorded(control)
    with pytest.raises(RuntimeError, match="pidfd"):
        aim_scratch.stop_aim_scratch(
            control,
            pidfd_open=lambda _pid: (_ for _ in ()).throw(NotImplementedError()),
        )
    source = (Path(__file__).parents[1] / "src" / "trainer_infra" / "aim_scratch.py").read_text()
    assert "os.kill" not in source


def test_stop_rejects_replaced_metadata_and_does_not_unlink_new_file(
    tmp_path: Path,
) -> None:
    control = _control(tmp_path)
    command = _recorded(control)
    probes = 0

    def health(_host: str, _port: int) -> bool:
        nonlocal probes
        probes += 1
        if probes == 2:
            control.metadata_file.unlink()
            control.metadata_file.write_text('{"replacement":true}')
        return probes == 1

    with pytest.raises(RuntimeError, match="evidence changed"):
        aim_scratch.stop_aim_scratch(
            control,
            inspect_process=lambda _pid: ProcessSnapshot(321, command, control.repo, 12345),
            health_probe=health,
            pidfd_open=lambda _pid: 91,
            pidfd_send_signal=lambda _fd, _value: None,
            wait_pidfd=lambda _fd, _timeout: True,
            close_fd=lambda _fd: None,
        )
    assert control.metadata_file.read_text() == '{"replacement":true}'
    assert control.pid_file.exists()
