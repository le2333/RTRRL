from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from trainer_infra.aim_scratch import (
    ProcessSnapshot,
    launch_aim_scratch,
    validate_aim_scratch,
)
from trainer_infra.facility_control import AimScratchControl


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
        "started_at_utc": "2026-07-23T18:00:00Z",
    }
    assert calls[0]["cwd"] == control.repo
    assert calls[0]["start_new_session"] is True


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
    )

    with pytest.raises(ValueError, match=drift):
        validate_aim_scratch(
            control,
            inspect_process=lambda _pid: snapshot,
            health_probe=lambda _host, _port: True,
        )
