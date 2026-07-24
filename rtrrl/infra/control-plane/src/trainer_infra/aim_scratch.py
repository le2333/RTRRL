from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from typing import Any

from trainer_infra.facility_control import AimScratchControl


@dataclass(frozen=True)
class ProcessSnapshot:
    pid: int
    cmdline: tuple[str, ...]
    cwd: Path


def inspect_process(pid: int) -> ProcessSnapshot:
    root = Path("/proc") / str(pid)
    command = tuple(
        item.decode()
        for item in (root / "cmdline").read_bytes().split(b"\0")
        if item
    )
    return ProcessSnapshot(pid=pid, cmdline=command, cwd=(root / "cwd").resolve())


def probe_health(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


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
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            snapshot = inspect_process(int(item.name))
        except (FileNotFoundError, OSError, UnicodeDecodeError):
            continue
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
    inspect_process: Callable[[int], ProcessSnapshot] = inspect_process,
    health_probe: Callable[[str, int], bool] = probe_health,
) -> dict[str, Any]:
    exact_process_active = False
    if control.metadata_file.exists() and control.pid_file.exists():
        try:
            _recorded_identity(control, inspect=inspect_process)
        except (ValueError, OSError):
            pass
        else:
            exact_process_active = True
    if exact_process_active:
        raise ValueError("exact Aim scratch server is active")
    if health_probe(control.host, control.port):
        raise ValueError("Aim scratch endpoint is occupied")
    return {"endpoint": control.endpoint, "status": "inactive"}


def stop_aim_scratch(
    control: AimScratchControl,
    *,
    inspect_process: Callable[[int], ProcessSnapshot] = inspect_process,
    health_probe: Callable[[str, int], bool] = probe_health,
    send_signal: Callable[[int, int], None] = os.kill,
    timeout: float = 10.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    _metadata, snapshot, metadata_bytes, pid_bytes = _recorded_identity(
        control,
        inspect=inspect_process,
    )
    if not health_probe(control.host, control.port):
        raise ValueError("recorded Aim scratch endpoint is unavailable")
    original_identity = (snapshot.cmdline, snapshot.cwd.resolve())
    send_signal(snapshot.pid, signal.SIGTERM)
    deadline = monotonic() + timeout
    while True:
        try:
            current = inspect_process(snapshot.pid)
        except (FileNotFoundError, ProcessLookupError):
            if not health_probe(control.host, control.port):
                break
        else:
            current_identity = (current.cmdline, current.cwd.resolve())
            if current_identity != original_identity:
                raise RuntimeError("PID reuse detected after SIGTERM")
        if monotonic() >= deadline:
            raise TimeoutError("Aim scratch did not stop after SIGTERM")
        sleep(0.1)

    try:
        replacement = inspect_process(snapshot.pid)
    except (FileNotFoundError, ProcessLookupError):
        pass
    else:
        replacement_identity = (replacement.cmdline, replacement.cwd.resolve())
        if replacement_identity != original_identity:
            raise RuntimeError("PID reuse detected before evidence removal")
        raise RuntimeError("Aim scratch process remained after endpoint stopped")
    if (
        control.metadata_file.read_bytes() != metadata_bytes
        or control.pid_file.read_bytes() != pid_bytes
    ):
        raise RuntimeError("Aim scratch evidence changed while stopping")
    control.metadata_file.unlink()
    control.pid_file.unlink()
    return {"pid": snapshot.pid, "status": "stopped"}


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


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
