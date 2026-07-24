from __future__ import annotations

import os
import stat

import pytest

from trainer_infra import aim_process_gate


def test_gate_eof_exits_without_exec(monkeypatch: pytest.MonkeyPatch) -> None:
    reads = iter([b""])
    monkeypatch.setattr(os, "read", lambda _fd, _size: next(reads))
    monkeypatch.setattr(os, "close", lambda _fd: None)
    monkeypatch.setattr(
        aim_process_gate,
        "_validate_fds",
        lambda _gate, _lock, _repo: None,
    )
    monkeypatch.setattr(
        os,
        "execvp",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not exec")),
    )
    assert (
        aim_process_gate.main(["--gate-fd", "3", "--lock-fd", "4", "--repo-fd", "5", "--", "aim"])
        == 0
    )


def test_gate_releases_only_one_exact_byte_to_exec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(os, "read", lambda _fd, _size: b"G")
    monkeypatch.setattr(os, "close", lambda _fd: None)
    monkeypatch.setattr(
        aim_process_gate,
        "_validate_fds",
        lambda _gate, _lock, _repo: None,
    )
    monkeypatch.setattr(
        os,
        "execvp",
        lambda executable, args: executed.append((executable, args)),
    )
    assert (
        aim_process_gate.main(
            [
                "--gate-fd",
                "3",
                "--lock-fd",
                "4",
                "--repo-fd",
                "5",
                "--",
                "/venv/bin/aim",
                "server",
            ]
        )
        == 0
    )
    assert executed == [("/venv/bin/aim", ["/venv/bin/aim", "server"])]


def test_gate_rejects_duplicate_or_wrong_fd_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="distinct"):
        aim_process_gate._validate_fds(3, 3, 5)

    monkeypatch.setattr(
        os,
        "fstat",
        lambda fd: type(
            "S",
            (),
            {"st_mode": (stat.S_IFIFO if fd == 3 else stat.S_IFREG)},
        )(),
    )
    with pytest.raises(ValueError, match="directory"):
        aim_process_gate._validate_fds(3, 4, 5)
