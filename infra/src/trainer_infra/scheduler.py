from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Protocol


class TaskState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class StudyTask:
    id: int
    config: Path
    catalog: Path
    database: Path
    state: TaskState
    pid: int | None
    created_at: str
    started_at: str | None
    finished_at: str | None
    exit_code: int | None
    reason: str | None


class Process(Protocol):
    pid: int

    def poll(self) -> int | None: ...


class TaskStore:
    """Persistent ownership and lifecycle state for study controllers."""

    def __init__(self, database: Path) -> None:
        self.database = Path(database).resolve()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY,
                    config TEXT NOT NULL,
                    catalog TEXT NOT NULL,
                    database TEXT NOT NULL,
                    state TEXT NOT NULL,
                    pid INTEGER,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    exit_code INTEGER,
                    reason TEXT,
                    UNIQUE(config, database)
                )
                """
            )

    def add(self, config: Path, catalog: Path, database: Path) -> StudyTask:
        values = (
            str(Path(config).resolve()),
            str(Path(catalog).resolve()),
            str(Path(database).resolve()),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO tasks(config, catalog, database, state, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (*values, TaskState.QUEUED.value, _now()),
            )
            row = connection.execute(
                "SELECT * FROM tasks WHERE config = ? AND database = ?",
                (values[0], values[2]),
            ).fetchone()
        return _task(row)

    def list(self) -> tuple[StudyTask, ...]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM tasks ORDER BY id").fetchall()
        return tuple(_task(row) for row in rows)

    def claim(self, limit: int) -> tuple[StudyTask, ...]:
        if limit < 1:
            return ()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT * FROM tasks WHERE state = ? ORDER BY id LIMIT ?",
                (TaskState.QUEUED.value, limit),
            ).fetchall()
            if rows:
                placeholders = ",".join("?" for _ in rows)
                connection.execute(
                    f"UPDATE tasks SET state = ?, started_at = ? WHERE id IN ({placeholders})",
                    (TaskState.RUNNING.value, _now(), *(row["id"] for row in rows)),
                )
                rows = connection.execute(
                    f"SELECT * FROM tasks WHERE id IN ({placeholders}) ORDER BY id",
                    tuple(row["id"] for row in rows),
                ).fetchall()
        return tuple(_task(row) for row in rows)

    def finish(self, task_id: int, exit_code: int, reason: str | None) -> None:
        state = TaskState.SUCCEEDED if exit_code == 0 else TaskState.FAILED
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE tasks
                SET state = ?, finished_at = ?, exit_code = ?, reason = ?, pid = NULL
                WHERE id = ?
                """,
                (state.value, _now(), exit_code, reason, task_id),
            )

    def record_pid(self, task_id: int, pid: int) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE tasks SET pid = ? WHERE id = ?", (pid, task_id))

    def interrupt_orphans(
        self, live: Callable[[StudyTask], bool]
    ) -> tuple[StudyTask, ...]:
        interrupted: list[StudyTask] = []
        for task in self.list():
            if task.state is TaskState.RUNNING and not live(task):
                self.finish(task.id, -1, "controller interrupted")
                interrupted.append(task)
        return tuple(interrupted)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=5.0)
        connection.row_factory = sqlite3.Row
        return connection


class Scheduler:
    """Keep a bounded set of independent study controllers alive."""

    def __init__(self, store: TaskStore, launch: Callable[[StudyTask], Process], *, max_concurrent: int = 4, poll_seconds: float = 5.0) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be positive")
        self.store = store
        self.launch = launch
        self.max_concurrent = max_concurrent
        self.poll_seconds = poll_seconds
        self.processes: dict[int, Process] = {}

    def tick(self) -> None:
        for task_id, process in tuple(self.processes.items()):
            exit_code = process.poll()
            if exit_code is None:
                continue
            reason = None if exit_code == 0 else f"trainerctl exited with code {exit_code}"
            self.store.finish(task_id, exit_code, reason)
            del self.processes[task_id]
        open_slots = self.max_concurrent - len(self.processes)
        for task in self.store.claim(open_slots):
            try:
                process = self.launch(task)
            except OSError as error:
                self.store.finish(task.id, -1, f"controller launch failed: {error}")
                continue
            self.store.record_pid(task.id, process.pid)
            self.processes[task.id] = process

    def run(self) -> None:
        self.store.interrupt_orphans(lambda task: task.id in self.processes)
        while True:
            self.tick()
            time.sleep(self.poll_seconds)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _task(row: sqlite3.Row) -> StudyTask:
    return StudyTask(
        id=int(row["id"]),
        config=Path(row["config"]),
        catalog=Path(row["catalog"]),
        database=Path(row["database"]),
        state=TaskState(row["state"]),
        pid=row["pid"],
        created_at=str(row["created_at"]),
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        exit_code=row["exit_code"],
        reason=row["reason"],
    )
