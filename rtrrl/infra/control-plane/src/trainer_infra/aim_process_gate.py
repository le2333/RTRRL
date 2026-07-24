from __future__ import annotations

import argparse
import os
import stat


def _validate_fds(gate_fd: int, lock_fd: int, repo_fd: int) -> None:
    if min(gate_fd, lock_fd, repo_fd) < 3 or len({gate_fd, lock_fd, repo_fd}) != 3:
        raise ValueError("gate, lock, and repo fds must be distinct descriptors")
    gate_stat = os.fstat(gate_fd)
    lock_stat = os.fstat(lock_fd)
    repo_stat = os.fstat(repo_fd)
    if not stat.S_ISFIFO(gate_stat.st_mode):
        raise ValueError("gate fd must be a pipe")
    if not stat.S_ISREG(lock_stat.st_mode):
        raise ValueError("lock fd must be a regular file")
    if not stat.S_ISDIR(repo_stat.st_mode):
        raise ValueError("repo fd must be a directory")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gate-fd", required=True, type=int)
    parser.add_argument("--lock-fd", required=True, type=int)
    parser.add_argument("--repo-fd", required=True, type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    command = arguments.command
    if command and command[0] == "--":
        command = command[1:]
    if not command or not command[0]:
        raise ValueError("gate requires an exact executable and argv")
    _validate_fds(arguments.gate_fd, arguments.lock_fd, arguments.repo_fd)
    release = os.read(arguments.gate_fd, 1)
    os.close(arguments.gate_fd)
    if not release:
        return 0
    if release != b"G":
        raise ValueError("gate release byte is invalid")
    os.execvp(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
