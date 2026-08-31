from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from trainer_infra.scheduler import Scheduler, StudyTask, TaskStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="study-scheduler")
    parser.add_argument("--state", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add")
    add.add_argument("config", type=Path)
    add.add_argument("--catalog", type=Path, required=True)
    add.add_argument("--database", type=Path, required=True)
    commands.add_parser("list")
    capacity = commands.add_parser("capacity")
    capacity.add_argument("value", type=int)
    launch_interval = commands.add_parser("launch-interval")
    launch_interval.add_argument("seconds", type=float)
    run = commands.add_parser("run")
    run.add_argument("--max-concurrent", type=int, default=4)
    run.add_argument("--poll-seconds", type=float, default=5.0)
    run.add_argument("--launch-interval-seconds", type=float, default=30.0)
    return parser


def _experiment(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("name"), str):
        raise TypeError(f"{path} must contain a string experiment name")
    return document


def _payload(task: StudyTask) -> dict[str, Any]:
    return {
        "id": task.id,
        "name": _experiment(task.config)["name"],
        "config": str(task.config),
        "catalog": str(task.catalog),
        "database": str(task.database),
        "state": task.state.value,
        "pid": task.pid,
        "exit_code": task.exit_code,
        "reason": task.reason,
    }


def _launcher(task: StudyTask) -> subprocess.Popen[bytes]:
    environment = dict(os.environ)
    environment.pop("AWS_PROFILE", None)
    environment["AWS_CONFIG_FILE"] = "/dev/null"
    environment["AWS_SHARED_CREDENTIALS_FILE"] = "/dev/null"
    command = (
        str(Path(sys.executable).with_name("trainerctl")),
        "run",
        str(task.config),
        "--backend",
        "batch",
        "--catalog",
        str(task.catalog),
        "--database",
        str(task.database),
        "--poll-seconds",
        "20",
    )
    return subprocess.Popen(command, env=environment)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        arguments = parser.parse_args(argv)
        store = TaskStore(arguments.state)
        if arguments.command == "add":
            if not arguments.config.is_file():
                raise ValueError(f"configuration does not exist: {arguments.config}")
            if not arguments.catalog.is_file():
                raise ValueError(f"catalog does not exist: {arguments.catalog}")
            _experiment(arguments.config)
            if arguments.state.resolve() == arguments.database.resolve():
                raise ValueError("scheduler and study databases must differ")
            print(json.dumps(_payload(store.add(arguments.config, arguments.catalog, arguments.database))))
            return 0
        if arguments.command == "list":
            tasks = store.list()
            states = [task.state.value for task in tasks]
            print(json.dumps({
                "capacity": store.ensure_capacity(4),
                "launch_interval_seconds": store.ensure_launch_interval(30.0),
                "running": states.count("running"),
                "queued": states.count("queued"),
                "tasks": [_payload(task) for task in tasks],
            }))
            return 0
        if arguments.command == "capacity":
            store.set_capacity(arguments.value)
            print(json.dumps({"capacity": store.capacity()}))
            return 0
        if arguments.command == "launch-interval":
            store.set_launch_interval(arguments.seconds)
            print(json.dumps({"launch_interval_seconds": store.launch_interval()}))
            return 0
        Scheduler(
            store,
            _launcher,
            max_concurrent=arguments.max_concurrent,
            poll_seconds=arguments.poll_seconds,
            launch_interval_seconds=arguments.launch_interval_seconds,
        ).run()
        return 0
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        print(f"study-scheduler: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
