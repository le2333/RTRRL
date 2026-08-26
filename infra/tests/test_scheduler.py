from __future__ import annotations

from pathlib import Path

from trainer_infra.scheduler import TaskState, TaskStore


def test_add_is_idempotent_for_canonical_config_and_database(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "scheduler.sqlite")
    first = store.add(
        tmp_path / "experiment.yml",
        tmp_path / "catalog.json",
        tmp_path / "run.sqlite",
    )
    second = store.add(
        tmp_path / "." / "experiment.yml",
        tmp_path / "catalog.json",
        tmp_path / "run.sqlite",
    )

    assert first.id == second.id
    assert [task.state for task in store.list()] == [TaskState.QUEUED]


def test_claim_and_finish_are_persistent(tmp_path: Path) -> None:
    database = tmp_path / "scheduler.sqlite"
    task = TaskStore(database).add(
        tmp_path / "a.yml",
        tmp_path / "catalog.json",
        tmp_path / "a.sqlite",
    )

    assert TaskStore(database).claim(1)[0].id == task.id
    TaskStore(database).finish(task.id, 0, None)

    assert TaskStore(database).list()[0].state is TaskState.SUCCEEDED


def test_missing_running_controller_is_marked_interrupted(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "scheduler.sqlite")
    running = store.add(
        tmp_path / "a.yml",
        tmp_path / "catalog.json",
        tmp_path / "a.sqlite",
    )
    assert store.claim(1)[0].id == running.id

    interrupted = store.interrupt_orphans(lambda task: False)

    assert [task.id for task in interrupted] == [running.id]
    task = store.list()[0]
    assert task.state is TaskState.FAILED
    assert task.reason == "controller interrupted"
