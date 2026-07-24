from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import socket
import stat
import subprocess
import time
from typing import Any

from trainer_infra.facility_control import AimScratchControl


FACILITY_LOCK_NAME = ".trainer-aim-scratch.lock"


@dataclass(frozen=True)
class ProcessSnapshot:
    pid: int
    cmdline: tuple[str, ...]
    cwd: Path
    start_time_ticks: int


def inspect_process(pid: int) -> ProcessSnapshot:
    root = Path("/proc") / str(pid)
    command = tuple(
        item.decode()
        for item in (root / "cmdline").read_bytes().split(b"\0")
        if item
    )
    stat_text = (root / "stat").read_text(encoding="utf-8")
    command_end = stat_text.rfind(")")
    fields = stat_text[command_end + 2 :].split()
    if command_end < 0 or len(fields) <= 19:
        raise ValueError(f"malformed process stat for PID {pid}")
    start_time_ticks = int(fields[19])
    return ProcessSnapshot(
        pid=pid,
        cmdline=command,
        cwd=(root / "cwd").resolve(),
        start_time_ticks=start_time_ticks,
    )


def enumerate_processes(
    *,
    inspector: Callable[[int], ProcessSnapshot] = inspect_process,
) -> Iterator[ProcessSnapshot]:
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            yield inspector(int(item.name))
        except (FileNotFoundError, OSError, UnicodeDecodeError, ValueError):
            continue


def open_trusted_directory(path: Path) -> int:
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise ValueError("trusted Aim path must be absolute and lexically canonical")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    current_fd = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            lexical_stat = os.stat(
                component,
                dir_fd=current_fd,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(lexical_stat.st_mode):
                raise ValueError("trusted Aim path contains a symlink or non-directory")
            child_fd = os.open(component, flags, dir_fd=current_fd)
            opened_stat = os.fstat(child_fd)
            if (
                opened_stat.st_dev != lexical_stat.st_dev
                or opened_stat.st_ino != lexical_stat.st_ino
            ):
                os.close(child_fd)
                raise ValueError("trusted Aim path changed while opening")
            os.close(current_fd)
            current_fd = child_fd
        final_lexical = path.lstat()
        final_opened = os.fstat(current_fd)
        if (
            final_opened.st_dev != final_lexical.st_dev
            or final_opened.st_ino != final_lexical.st_ino
        ):
            raise ValueError("trusted Aim path identity mismatch")
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def open_facility_lock(directory_fd: int, *, exclusive: bool) -> int:
    lock_fd = os.open(
        FACILITY_LOCK_NAME,
        os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    lock_stat = os.fstat(lock_fd)
    if (
        not stat.S_ISREG(lock_stat.st_mode)
        or lock_stat.st_uid != os.getuid()
        or stat.S_IMODE(lock_stat.st_mode) & 0o077
    ):
        os.close(lock_fd)
        raise ValueError("Aim scratch facility lock file is not private and regular")
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    try:
        fcntl.flock(lock_fd, operation | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(lock_fd)
        raise ValueError("Aim scratch facility lock is already held") from error
    return lock_fd


def probe_health(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def _open_pidfd(pid: int) -> int:
    opener = getattr(os, "pidfd_open", None)
    if opener is None:
        raise RuntimeError("Linux pidfd_open is required to stop Aim safely")
    return opener(pid)


def _send_pidfd_signal(pidfd: int, signal_number: int) -> None:
    sender = getattr(signal, "pidfd_send_signal", None)
    if sender is None:
        raise RuntimeError("Linux pidfd_send_signal is required to stop Aim safely")
    sender(pidfd, signal_number)


def _wait_for_pidfd(pidfd: int, timeout: float) -> bool:
    poller = select.poll()
    poller.register(pidfd, select.POLLIN)
    return bool(poller.poll(max(0, int(timeout * 1000))))


def _argument(command: tuple[str, ...], name: str) -> str | None:
    try:
        index = command.index(name)
        return command[index + 1]
    except (ValueError, IndexError):
        return None


def _recorded_identity(
    control: AimScratchControl,
    *,
    inspect: Callable[[int], ProcessSnapshot],
) -> tuple[dict[str, Any], ProcessSnapshot, bytes, bytes]:
    try:
        metadata_bytes = control.metadata_file.read_bytes()
        pid_bytes = control.pid_file.read_bytes()
        metadata = json.loads(metadata_bytes)
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeDecodeError) as error:
        raise ValueError("Aim scratch metadata is stale or unreadable") from error
    if not isinstance(metadata, dict):
        raise ValueError("Aim scratch metadata mismatch")
    pid_text = pid_bytes.decode("utf-8").strip()
    if not pid_text.isdigit() or int(pid_text) != metadata.get("pid"):
        raise ValueError("pid metadata mismatch")
    pid = int(pid_text)
    try:
        snapshot = inspect(pid)
    except (FileNotFoundError, ProcessLookupError) as error:
        raise ValueError("Aim scratch PID is stale") from error
    repo = control.repo.resolve()
    start_time_ticks = metadata.get("start_time_ticks")
    if (
        not isinstance(start_time_ticks, int)
        or isinstance(start_time_ticks, bool)
        or start_time_ticks <= 0
    ):
        raise ValueError("start_time_ticks metadata mismatch")
    if snapshot.start_time_ticks != start_time_ticks:
        raise ValueError("start_time_ticks mismatch")
    command = snapshot.cmdline
    expected_command = tuple(metadata.get("command", ()))
    if command != expected_command or "server" not in command:
        raise ValueError("cmdline mismatch")
    if snapshot.cwd.resolve() != repo or metadata.get("cwd") != str(repo):
        raise ValueError("cwd mismatch")
    if _argument(command, "--repo") != str(repo) or metadata.get("repo") != str(repo):
        raise ValueError("repo argument mismatch")
    if (
        _argument(command, "--host") != control.host
        or _argument(command, "--port") != str(control.port)
        or metadata.get("port") != control.port
    ):
        raise ValueError("port argument mismatch")
    if metadata.get("endpoint") != control.endpoint:
        raise ValueError("endpoint mismatch")
    return metadata, snapshot, metadata_bytes, pid_bytes


def discover_aim_process(control: AimScratchControl) -> ProcessSnapshot:
    matches = []
    for snapshot in enumerate_processes():
        if (
            "server" in snapshot.cmdline
            and _argument(snapshot.cmdline, "--repo") == str(control.repo.resolve())
            and _argument(snapshot.cmdline, "--port") == str(control.port)
            and snapshot.cwd.resolve() == control.repo.resolve()
        ):
            matches.append(snapshot)
    if not matches:
        raise ValueError("no matching Aim scratch process was found")
    return max(matches, key=lambda item: item.pid)


def validate_aim_scratch(
    control: AimScratchControl,
    *,
    inspect_process: Callable[[int], ProcessSnapshot] = inspect_process,
    health_probe: Callable[[str, int], bool] = probe_health,
) -> dict[str, Any]:
    repo = control.repo.resolve()
    main = control.main_repo.resolve()
    if repo == main or repo in main.parents or main in repo.parents:
        raise ValueError("repo isolation mismatch")
    _metadata, snapshot, _metadata_bytes, _pid_bytes = _recorded_identity(
        control,
        inspect=inspect_process,
    )
    if not health_probe(control.host, control.port):
        raise ValueError("health endpoint is unavailable")
    return {
        "cmdline_matches": True,
        "cwd_matches": True,
        "endpoint": control.endpoint,
        "healthy": True,
        "isolated": True,
        "metadata_file": str(control.metadata_file),
        "pid": snapshot.pid,
        "pid_file": str(control.pid_file),
        "port_matches": True,
        "repo": str(repo),
        "repo_matches": True,
        "status": "ready",
    }


def assert_aim_scratch_inactive(
    control: AimScratchControl,
    *,
    process_enumerator: Callable[[], Iterable[ProcessSnapshot]] = enumerate_processes,
    health_probe: Callable[[str, int], bool] = probe_health,
) -> dict[str, Any]:
    repo = control.repo.resolve(strict=True)
    for snapshot in process_enumerator():
        command = snapshot.cmdline
        repo_argument = _argument(command, "--repo")
        if "server" not in command or repo_argument is None:
            continue
        try:
            process_repo_path = Path(repo_argument)
            if not process_repo_path.is_absolute():
                process_repo_path = snapshot.cwd / process_repo_path
            process_repo = process_repo_path.resolve(strict=True)
        except OSError:
            continue
        if process_repo == repo:
            raise ValueError("exact Aim scratch server is active")
    if health_probe(control.host, control.port):
        raise ValueError("Aim scratch endpoint is occupied")
    return {"endpoint": control.endpoint, "status": "inactive"}


def stop_aim_scratch(
    control: AimScratchControl,
    *,
    inspect_process: Callable[[int], ProcessSnapshot] = inspect_process,
    health_probe: Callable[[str, int], bool] = probe_health,
    pidfd_open: Callable[[int], int] = _open_pidfd,
    pidfd_send_signal: Callable[[int, int], None] = _send_pidfd_signal,
    wait_pidfd: Callable[[int, float], bool] = _wait_for_pidfd,
    close_fd: Callable[[int], None] = os.close,
    timeout: float = 10.0,
) -> dict[str, Any]:
    try:
        pid_text = control.pid_file.read_text(encoding="utf-8").strip()
        metadata = json.loads(control.metadata_file.read_bytes())
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Aim scratch metadata is stale or unreadable") from error
    if (
        not pid_text.isdigit()
        or not isinstance(metadata, dict)
        or int(pid_text) != metadata.get("pid")
    ):
        raise ValueError("pid metadata mismatch")
    pid = int(pid_text)
    try:
        pidfd = pidfd_open(pid)
    except (AttributeError, NotImplementedError) as error:
        raise RuntimeError("Linux pidfd support is required to stop Aim safely") from error
    try:
        _metadata, snapshot, metadata_bytes, pid_bytes = _recorded_identity(
            control,
            inspect=inspect_process,
        )
        if not health_probe(control.host, control.port):
            raise ValueError("recorded Aim scratch endpoint is unavailable")
        try:
            before_signal = inspect_process(snapshot.pid)
        except (FileNotFoundError, ProcessLookupError) as error:
            raise RuntimeError("Aim scratch generation vanished before SIGTERM") from error
        original_identity = (
            snapshot.cmdline,
            snapshot.cwd.resolve(),
            snapshot.start_time_ticks,
        )
        before_signal_identity = (
            before_signal.cmdline,
            before_signal.cwd.resolve(),
            before_signal.start_time_ticks,
        )
        if before_signal_identity != original_identity:
            raise RuntimeError("PID reuse detected before SIGTERM")
        pidfd_send_signal(pidfd, signal.SIGTERM)
        if not wait_pidfd(pidfd, timeout):
            raise TimeoutError("Aim scratch did not stop after SIGTERM")
        if health_probe(control.host, control.port):
            raise RuntimeError("Aim scratch endpoint remains occupied after termination")
        if (
            control.metadata_file.read_bytes() != metadata_bytes
            or control.pid_file.read_bytes() != pid_bytes
        ):
            raise RuntimeError("Aim scratch evidence changed while stopping")
        control.metadata_file.unlink()
        control.pid_file.unlink()
        return {"pid": snapshot.pid, "status": "stopped"}
    finally:
        close_fd(pidfd)


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _launch_aim_scratch_locked(
    control: AimScratchControl,
    *,
    inherited_lock_fd: int,
    aim_executable: str,
    popen: Callable[..., Any] = subprocess.Popen,
    health_probe: Callable[[str, int], bool] = probe_health,
    discover_process: Callable[[AimScratchControl], ProcessSnapshot] = discover_aim_process,
    runtime_validator: Callable[[AimScratchControl], dict[str, Any]] = validate_aim_scratch,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    repo = control.repo.resolve()
    if not (repo / ".aim").is_dir():
        raise ValueError(f"Aim repository metadata is missing under {repo}")
    if control.metadata_file.is_file() and control.pid_file.is_file():
        try:
            current = runtime_validator(control)
        except FileNotFoundError:
            if health_probe(control.host, control.port):
                raise ValueError(
                    "Aim endpoint is occupied but recorded process is missing"
                ) from None
        else:
            return {**current, "status": "resumed"}
    command = [
        aim_executable,
        "server",
        "--host",
        control.host,
        "--port",
        str(control.port),
        "--repo",
        str(repo),
        "-y",
    ]
    with control.log_file.open("ab") as log:
        popen(
            command,
            cwd=repo,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            pass_fds=(inherited_lock_fd,),
        )
    deadline = time.monotonic() + 10
    while not health_probe(control.host, control.port):
        if time.monotonic() >= deadline:
            raise TimeoutError("Aim scratch did not become healthy")
        sleep(0.1)
    snapshot = discover_process(control)
    if snapshot.cwd.resolve() != repo:
        raise ValueError("launched Aim cwd mismatch")
    metadata = {
        "command": list(snapshot.cmdline),
        "cwd": str(snapshot.cwd.resolve()),
        "endpoint": control.endpoint,
        "pid": snapshot.pid,
        "port": control.port,
        "repo": str(repo),
        "start_time_ticks": snapshot.start_time_ticks,
        "started_at_utc": _utc_text(now()),
    }
    temporary = control.metadata_file.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(metadata, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(control.metadata_file)
    control.pid_file.write_text(f"{snapshot.pid}\n", encoding="utf-8")
    return {**metadata, "status": "started"}


def launch_aim_scratch(
    control: AimScratchControl,
    *,
    aim_executable: str,
    popen: Callable[..., Any] = subprocess.Popen,
    health_probe: Callable[[str, int], bool] = probe_health,
    discover_process: Callable[[AimScratchControl], ProcessSnapshot] = discover_aim_process,
    runtime_validator: Callable[[AimScratchControl], dict[str, Any]] = validate_aim_scratch,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    directory_fd = open_trusted_directory(control.repo)
    try:
        lock_fd = open_facility_lock(directory_fd, exclusive=False)
        try:
            return _launch_aim_scratch_locked(
                control,
                inherited_lock_fd=lock_fd,
                aim_executable=aim_executable,
                popen=popen,
                health_probe=health_probe,
                discover_process=discover_process,
                runtime_validator=runtime_validator,
                now=now,
                sleep=sleep,
            )
        finally:
            os.close(lock_fd)
    finally:
        os.close(directory_fd)
