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
import sys
import time
from typing import Any

from trainer_infra.facility_control import AimScratchControl


FACILITY_LOCK_NAME = ".trainer-aim-scratch.lock"
_GATE_ENVIRONMENT_KEYS = ("HOME", "LANG", "LC_ALL", "PATH", "TMPDIR")


@dataclass(frozen=True)
class ProcessSnapshot:
    pid: int
    cmdline: tuple[str, ...]
    cwd: Path
    start_time_ticks: int


@dataclass(frozen=True)
class FileEvidence:
    name: str
    device: int
    inode: int
    content: bytes


def inspect_process(pid: int) -> ProcessSnapshot:
    root = Path("/proc") / str(pid)
    command = tuple(item.decode() for item in (root / "cmdline").read_bytes().split(b"\0") if item)
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


def _validate_and_lock_facility_file(lock_fd: int) -> int:
    lock_stat = os.fstat(lock_fd)
    if (
        not stat.S_ISREG(lock_stat.st_mode)
        or lock_stat.st_uid != os.getuid()
        or stat.S_IMODE(lock_stat.st_mode) & 0o077
    ):
        os.close(lock_fd)
        raise ValueError("Aim scratch facility lock file is not private and regular")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(lock_fd)
        raise ValueError("Aim scratch facility lock is already held") from error
    return lock_fd


def create_facility_lock(directory_fd: int) -> int:
    flags = os.O_RDWR | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        lock_fd = os.open(
            FACILITY_LOCK_NAME,
            flags | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory_fd,
        )
    except FileExistsError:
        lock_fd = os.open(FACILITY_LOCK_NAME, flags, dir_fd=directory_fd)
    return _validate_and_lock_facility_file(lock_fd)


def open_facility_lock(directory_fd: int) -> int:
    try:
        lock_fd = os.open(
            FACILITY_LOCK_NAME,
            os.O_RDWR | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
    except FileNotFoundError as error:
        raise ValueError("existing Aim scratch facility lock is required") from error
    return _validate_and_lock_facility_file(lock_fd)


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


def _process_repo_path(snapshot: ProcessSnapshot, argument: str) -> Path:
    self_fd_prefix = "/proc/self/fd/"
    if argument.startswith(self_fd_prefix):
        fd_text = argument.removeprefix(self_fd_prefix)
        if not fd_text.isdigit():
            raise ValueError("process repo fd argument is invalid")
        return Path("/proc") / str(snapshot.pid) / "fd" / fd_text
    path = Path(argument)
    return path if path.is_absolute() else snapshot.cwd / path


def _process_repo_identity(
    snapshot: ProcessSnapshot,
    argument: str,
) -> tuple[int, int]:
    repo_stat = _process_repo_path(snapshot, argument).stat()
    return repo_stat.st_dev, repo_stat.st_ino


def _read_regular_file_at(directory_fd: int, name: str) -> FileEvidence:
    if not name or "/" in name:
        raise ValueError("Aim evidence file must be directly under trusted scratch")
    lexical = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(lexical.st_mode):
        raise ValueError("Aim evidence must be a regular file")
    file_fd = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        opened = os.fstat(file_fd)
        if opened.st_dev != lexical.st_dev or opened.st_ino != lexical.st_ino:
            raise ValueError("Aim evidence identity changed while opening")
        chunks = []
        while chunk := os.read(file_fd, 1024 * 1024):
            chunks.append(chunk)
        return FileEvidence(name, opened.st_dev, opened.st_ino, b"".join(chunks))
    finally:
        os.close(file_fd)


def _evidence_unchanged(directory_fd: int, evidence: FileEvidence) -> bool:
    try:
        current = _read_regular_file_at(directory_fd, evidence.name)
    except (FileNotFoundError, OSError, ValueError):
        return False
    return current == evidence


def _remove_exact_evidence(
    directory_fd: int,
    evidence_items: tuple[FileEvidence, ...],
    *,
    nonce: int,
) -> None:
    moved: list[tuple[FileEvidence, str]] = []
    try:
        for evidence in evidence_items:
            quarantine = f".{evidence.name}.remove-{nonce}"
            os.rename(
                evidence.name,
                quarantine,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            moved.append((evidence, quarantine))
        for evidence, quarantine in moved:
            quarantined = _read_regular_file_at(directory_fd, quarantine)
            if (
                quarantined.device != evidence.device
                or quarantined.inode != evidence.inode
                or quarantined.content != evidence.content
            ):
                raise RuntimeError("Aim scratch evidence was replaced during removal")
        for _evidence, quarantine in moved:
            os.unlink(quarantine, dir_fd=directory_fd)
    except BaseException:
        for evidence, quarantine in reversed(moved):
            try:
                os.stat(evidence.name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                try:
                    os.rename(
                        quarantine,
                        evidence.name,
                        src_dir_fd=directory_fd,
                        dst_dir_fd=directory_fd,
                    )
                except FileNotFoundError:
                    pass
        raise


def _recorded_identity(
    control: AimScratchControl,
    *,
    inspect: Callable[[int], ProcessSnapshot],
    directory_fd: int | None = None,
) -> tuple[dict[str, Any], ProcessSnapshot, bytes, bytes]:
    try:
        if control.metadata_file.parent != control.repo or control.pid_file.parent != control.repo:
            raise ValueError("Aim evidence files must be under trusted scratch")
        if directory_fd is None:
            metadata_bytes = control.metadata_file.read_bytes()
            pid_bytes = control.pid_file.read_bytes()
        else:
            metadata_bytes = _read_regular_file_at(directory_fd, control.metadata_file.name).content
            pid_bytes = _read_regular_file_at(directory_fd, control.pid_file.name).content
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
    root_dev = metadata.get("repo_root_dev")
    root_ino = metadata.get("repo_root_ino")
    repo_fd = metadata.get("repo_fd")
    trusted_repo = metadata.get("trusted_repo")
    if (
        not isinstance(root_dev, int)
        or not isinstance(root_ino, int)
        or not isinstance(repo_fd, int)
        or trusted_repo != str(control.repo)
    ):
        raise ValueError("repo root identity metadata mismatch")
    repo_argument = _argument(command, "--repo")
    if repo_argument != f"/proc/self/fd/{repo_fd}":
        raise ValueError("repo fd argument mismatch")
    if _process_repo_identity(snapshot, repo_argument) != (root_dev, root_ino):
        raise ValueError("repo root inode mismatch")
    cwd_stat = snapshot.cwd.stat()
    if (cwd_stat.st_dev, cwd_stat.st_ino) != (root_dev, root_ino):
        raise ValueError("cwd mismatch")
    if (
        _argument(command, "--host") != control.host
        or _argument(command, "--port") != str(control.port)
        or metadata.get("port") != control.port
    ):
        raise ValueError("port argument mismatch")
    if metadata.get("endpoint") != control.endpoint:
        raise ValueError("endpoint mismatch")
    return metadata, snapshot, metadata_bytes, pid_bytes


def discover_aim_process(
    control: AimScratchControl,
    *,
    root_identity: tuple[int, int] | None = None,
) -> ProcessSnapshot:
    if root_identity is None:
        repo_stat = control.repo.stat()
        root_identity = (repo_stat.st_dev, repo_stat.st_ino)
    matches = []
    for snapshot in enumerate_processes():
        repo_argument = _argument(snapshot.cmdline, "--repo")
        if "server" not in snapshot.cmdline or repo_argument is None:
            continue
        try:
            process_identity = _process_repo_identity(snapshot, repo_argument)
        except OSError:
            continue
        if process_identity == root_identity and _argument(snapshot.cmdline, "--port") == str(
            control.port
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
    directory_fd: int | None = None,
) -> dict[str, Any]:
    own_directory_fd = directory_fd is None
    if directory_fd is None:
        directory_fd = open_trusted_directory(control.repo)
    try:
        repo = control.repo.resolve()
        main = control.main_repo.resolve()
        if repo == main or repo in main.parents or main in repo.parents:
            raise ValueError("repo isolation mismatch")
        _metadata, snapshot, _metadata_bytes, _pid_bytes = _recorded_identity(
            control,
            inspect=inspect_process,
            directory_fd=directory_fd,
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
            "repo_root_dev": _metadata["repo_root_dev"],
            "repo_root_ino": _metadata["repo_root_ino"],
            "repo": str(repo),
            "repo_matches": True,
            "start_time_ticks": snapshot.start_time_ticks,
            "status": "ready",
        }
    finally:
        if own_directory_fd:
            os.close(directory_fd)


def assert_aim_scratch_inactive(
    control: AimScratchControl,
    *,
    process_enumerator: Callable[[], Iterable[ProcessSnapshot]] = enumerate_processes,
    health_probe: Callable[[str, int], bool] = probe_health,
) -> dict[str, Any]:
    repo_stat = control.repo.stat()
    repo_identity = (repo_stat.st_dev, repo_stat.st_ino)
    for snapshot in process_enumerator():
        command = snapshot.cmdline
        repo_argument = _argument(command, "--repo")
        if "server" not in command or repo_argument is None:
            continue
        try:
            process_identity = _process_repo_identity(snapshot, repo_argument)
        except OSError:
            continue
        if process_identity == repo_identity:
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
    if control.metadata_file.parent != control.repo or control.pid_file.parent != control.repo:
        raise ValueError("Aim evidence files must be under trusted scratch")
    directory_fd = open_trusted_directory(control.repo)
    try:
        try:
            metadata_evidence = _read_regular_file_at(directory_fd, control.metadata_file.name)
            pid_evidence = _read_regular_file_at(directory_fd, control.pid_file.name)
            metadata = json.loads(metadata_evidence.content)
            pid_text = pid_evidence.content.decode("utf-8").strip()
        except (
            FileNotFoundError,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as error:
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
            _metadata, snapshot, _metadata_bytes, _pid_bytes = _recorded_identity(
                control,
                inspect=inspect_process,
                directory_fd=directory_fd,
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
            if not _evidence_unchanged(directory_fd, metadata_evidence) or not _evidence_unchanged(
                directory_fd, pid_evidence
            ):
                raise RuntimeError("Aim scratch evidence changed while stopping")
            _remove_exact_evidence(
                directory_fd,
                (metadata_evidence, pid_evidence),
                nonce=snapshot.pid,
            )
            return {"pid": snapshot.pid, "status": "stopped"}
        finally:
            close_fd(pidfd)
    finally:
        os.close(directory_fd)


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _gate_environment() -> dict[str, str]:
    return {name: os.environ[name] for name in _GATE_ENVIRONMENT_KEYS if name in os.environ}


def _expected_resume_identity(
    control: AimScratchControl,
    directory_fd: int,
) -> tuple[int, int, int, int]:
    if control.metadata_file.parent != control.repo or control.pid_file.parent != control.repo:
        raise ValueError("Aim evidence files must be under trusted scratch")
    metadata = json.loads(_read_regular_file_at(directory_fd, control.metadata_file.name).content)
    pid_text = (
        _read_regular_file_at(directory_fd, control.pid_file.name).content.decode("utf-8").strip()
    )
    if not isinstance(metadata, dict) or not pid_text.isdigit():
        raise ValueError("Aim resume metadata is invalid")
    identity = (
        metadata.get("pid"),
        metadata.get("start_time_ticks"),
        metadata.get("repo_root_dev"),
        metadata.get("repo_root_ino"),
    )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in identity
    ):
        raise ValueError("Aim resume identity metadata is invalid")
    if int(pid_text) != identity[0]:
        raise ValueError("Aim resume PID metadata mismatch")
    root_stat = os.fstat(directory_fd)
    if identity[2:] != (root_stat.st_dev, root_stat.st_ino):
        raise ValueError("Aim resume root identity metadata mismatch")
    return identity


def _resume_existing_process(
    control: AimScratchControl,
    *,
    directory_fd: int,
    runtime_validator: Callable[..., dict[str, Any]],
    pidfd_open: Callable[[int], int],
    wait_pidfd: Callable[[int, float], bool],
    close_fd: Callable[[int], None],
) -> dict[str, Any]:
    expected = _expected_resume_identity(control, directory_fd)
    resume_pidfd = pidfd_open(expected[0])
    try:
        if wait_pidfd(resume_pidfd, 0):
            raise ValueError("recorded Aim process exited before runtime validation")
        current = runtime_validator(control, directory_fd=directory_fd)
        observed = (
            current.get("pid"),
            current.get("start_time_ticks"),
            current.get("repo_root_dev"),
            current.get("repo_root_ino"),
        )
        if (
            any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in observed
            )
            or observed != expected
        ):
            raise ValueError("Aim resume runtime identity mismatch")
        if wait_pidfd(resume_pidfd, 0):
            raise ValueError("recorded Aim process exited during runtime validation")
        return {**current, "status": "resumed"}
    finally:
        close_fd(resume_pidfd)


def _launch_aim_scratch_locked(
    control: AimScratchControl,
    *,
    directory_fd: int,
    inherited_lock_fd: int,
    aim_executable: str,
    popen: Callable[..., Any] = subprocess.Popen,
    health_probe: Callable[[str, int], bool] = probe_health,
    discover_process: Callable[..., ProcessSnapshot] = discover_aim_process,
    runtime_validator: Callable[..., dict[str, Any]] = validate_aim_scratch,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    sleep: Callable[[float], None] = time.sleep,
    pidfd_open: Callable[[int], int] = _open_pidfd,
    pidfd_send_signal: Callable[[int, int], None] = _send_pidfd_signal,
    wait_pidfd: Callable[[int, float], bool] = _wait_for_pidfd,
    close_fd: Callable[[int], None] = os.close,
) -> dict[str, Any]:
    if (
        control.metadata_file.parent != control.repo
        or control.pid_file.parent != control.repo
        or control.log_file.parent != control.repo
    ):
        raise ValueError("Aim lifecycle files must be under trusted scratch")
    pinned_repo = Path("/proc/self/fd") / str(directory_fd)
    root_stat = os.fstat(directory_fd)
    if not (pinned_repo / ".aim").is_dir():
        raise ValueError(f"Aim repository metadata is missing under {control.repo}")
    if control.metadata_file.is_file() and control.pid_file.is_file():
        try:
            return _resume_existing_process(
                control,
                directory_fd=directory_fd,
                runtime_validator=runtime_validator,
                pidfd_open=pidfd_open,
                wait_pidfd=wait_pidfd,
                close_fd=close_fd,
            )
        except FileNotFoundError:
            if health_probe(control.host, control.port):
                raise ValueError(
                    "Aim endpoint is occupied but recorded process is missing"
                ) from None
    command = [
        aim_executable,
        "server",
        "--host",
        control.host,
        "--port",
        str(control.port),
        "--repo",
        str(pinned_repo),
        "-y",
    ]
    gate_read_fd, gate_write_fd = os.pipe2(os.O_CLOEXEC)
    parent_fds = {gate_read_fd, gate_write_fd}
    try:
        gate_command = [
            sys.executable,
            "-m",
            "trainer_infra.aim_process_gate",
            "--gate-fd",
            str(gate_read_fd),
            "--lock-fd",
            str(inherited_lock_fd),
            "--repo-fd",
            str(directory_fd),
            "--",
            *command,
        ]
        log_fd = os.open(
            control.log_file.name,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC,
            0o600,
            dir_fd=directory_fd,
        )
        parent_fds.add(log_fd)
        with os.fdopen(log_fd, "ab", closefd=True) as log:
            parent_fds.discard(log_fd)
            process = popen(
                gate_command,
                cwd=pinned_repo,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                pass_fds=(gate_read_fd, inherited_lock_fd, directory_fd),
                env=_gate_environment(),
            )
        try:
            child_pidfd = pidfd_open(process.pid)
        except BaseException:
            os.close(gate_write_fd)
            parent_fds.discard(gate_write_fd)
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    "gated Aim wrapper did not exit after pidfd_open failure"
                ) from error
            raise RuntimeError("could not bind launched Aim child to a pidfd") from None
        try:
            try:
                os.write(gate_write_fd, b"G")
                os.close(gate_write_fd)
                parent_fds.discard(gate_write_fd)
                os.close(gate_read_fd)
                parent_fds.discard(gate_read_fd)
                deadline = time.monotonic() + 10
                while not health_probe(control.host, control.port):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Aim scratch did not become healthy")
                    sleep(0.1)
                snapshot = discover_process(
                    control,
                    root_identity=(root_stat.st_dev, root_stat.st_ino),
                )
                if snapshot.pid != process.pid:
                    raise ValueError("launched Aim process PID mismatch")
                repo_argument = _argument(snapshot.cmdline, "--repo")
                if repo_argument is None or _process_repo_identity(snapshot, repo_argument) != (
                    root_stat.st_dev,
                    root_stat.st_ino,
                ):
                    raise ValueError("launched Aim repo inode mismatch")
                metadata = {
                    "command": list(snapshot.cmdline),
                    "cwd": str(snapshot.cwd.resolve()),
                    "endpoint": control.endpoint,
                    "pid": snapshot.pid,
                    "port": control.port,
                    "repo_fd": directory_fd,
                    "repo_root_dev": root_stat.st_dev,
                    "repo_root_ino": root_stat.st_ino,
                    "start_time_ticks": snapshot.start_time_ticks,
                    "started_at_utc": _utc_text(now()),
                    "trusted_repo": str(control.repo),
                }
                metadata_name = control.metadata_file.name
                metadata_temporary = f".{metadata_name}.tmp-{process.pid}"
                metadata_fd = os.open(
                    metadata_temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                    0o600,
                    dir_fd=directory_fd,
                )
                try:
                    os.write(
                        metadata_fd,
                        (
                            json.dumps(metadata, separators=(",", ":"), sort_keys=True) + "\n"
                        ).encode(),
                    )
                finally:
                    os.close(metadata_fd)
                os.replace(
                    metadata_temporary,
                    metadata_name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                )
                pid_name = control.pid_file.name
                pid_temporary = f".{pid_name}.tmp-{process.pid}"
                pid_fd = os.open(
                    pid_temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                    0o600,
                    dir_fd=directory_fd,
                )
                try:
                    os.write(pid_fd, f"{snapshot.pid}\n".encode())
                finally:
                    os.close(pid_fd)
                os.replace(
                    pid_temporary,
                    pid_name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                )
                return {**metadata, "status": "started"}
            except BaseException:
                pidfd_send_signal(child_pidfd, signal.SIGTERM)
                if not wait_pidfd(child_pidfd, 10.0):
                    raise RuntimeError(
                        "launched Aim child could not be terminated safely"
                    ) from None
                raise
        finally:
            close_fd(child_pidfd)
    finally:
        for parent_fd in parent_fds:
            try:
                os.close(parent_fd)
            except OSError:
                pass


def launch_aim_scratch(
    control: AimScratchControl,
    *,
    aim_executable: str,
    popen: Callable[..., Any] = subprocess.Popen,
    health_probe: Callable[[str, int], bool] = probe_health,
    discover_process: Callable[..., ProcessSnapshot] = discover_aim_process,
    runtime_validator: Callable[..., dict[str, Any]] = validate_aim_scratch,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    sleep: Callable[[float], None] = time.sleep,
    pidfd_open: Callable[[int], int] = _open_pidfd,
    pidfd_send_signal: Callable[[int, int], None] = _send_pidfd_signal,
    wait_pidfd: Callable[[int, float], bool] = _wait_for_pidfd,
    close_fd: Callable[[int], None] = os.close,
) -> dict[str, Any]:
    directory_fd = open_trusted_directory(control.repo)
    try:
        try:
            lock_fd = create_facility_lock(directory_fd)
        except ValueError as lock_error:
            if "already held" not in str(lock_error):
                raise
            try:
                resumed = _resume_existing_process(
                    control,
                    directory_fd=directory_fd,
                    runtime_validator=runtime_validator,
                    pidfd_open=pidfd_open,
                    wait_pidfd=wait_pidfd,
                    close_fd=close_fd,
                )
            except BaseException as validation_error:
                occupied = ValueError(
                    "Aim scratch facility lock is occupied and holder validation failed"
                )
                occupied.add_note(f"lock acquisition failed: {lock_error!r}")
                raise occupied from validation_error
            return resumed
        try:
            return _launch_aim_scratch_locked(
                control,
                directory_fd=directory_fd,
                inherited_lock_fd=lock_fd,
                aim_executable=aim_executable,
                popen=popen,
                health_probe=health_probe,
                discover_process=discover_process,
                runtime_validator=runtime_validator,
                now=now,
                sleep=sleep,
                pidfd_open=pidfd_open,
                pidfd_send_signal=pidfd_send_signal,
                wait_pidfd=wait_pidfd,
                close_fd=close_fd,
            )
        finally:
            os.close(lock_fd)
    finally:
        os.close(directory_fd)
