from __future__ import annotations

from pathlib import Path

from trainer_infra.scheduler import Scheduler, StudyTask, TaskState, TaskStore


class FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.exit_code: int | None = None

    def poll(self) -> int | None:
        return self.exit_code


class FakeProcessFactory:
    def __init__(self) -> None:
        self.started: list[FakeProcess] = []

    def start(self, task: StudyTask) -> FakeProcess:
        process = FakeProcess(1000 + task.id)
        self.started.append(process)
        return process

    @property
    def running(self) -> list[FakeProcess]:
        return [process for process in self.started if process.poll() is None]

    def finish_one(self, exit_code: int) -> None:
        self.running[0].exit_code = exit_code


def store_with_tasks(tmp_path: Path, count: int) -> TaskStore:
    store = TaskStore(tmp_path / "scheduler.sqlite")
    for index in range(count):
        store.add(
            tmp_path / f"{index}.yml",
            tmp_path / "catalog.json",
            tmp_path / f"{index}.sqlite",
        )
    return store


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


def test_tick_never_runs_more_than_four_studies(tmp_path: Path) -> None:
    processes = FakeProcessFactory()
    scheduler = Scheduler(store_with_tasks(tmp_path, 6), processes.start, max_concurrent=4)
    scheduler.tick()
    assert len(processes.running) == 4
    processes.finish_one(0)
    scheduler.tick()
    assert len(processes.running) == 4
    queued = [task for task in scheduler.store.list() if task.state is TaskState.QUEUED]
    assert len(queued) == 1


def test_failed_child_does_not_block_next_task(tmp_path: Path) -> None:
    processes = FakeProcessFactory()
    scheduler = Scheduler(store_with_tasks(tmp_path, 5), processes.start, max_concurrent=4)
    scheduler.tick()
    processes.finish_one(7)
    scheduler.tick()
    states = [task.state for task in scheduler.store.list()]
    assert states.count(TaskState.FAILED) == 1
    assert len(processes.running) == 4



def test_capacity_change_fills_an_extra_slot_without_restarting(tmp_path: Path) -> None:
    store = store_with_tasks(tmp_path, 5)
    store.set_capacity(3)
    processes = FakeProcessFactory()
    scheduler = Scheduler(store, processes.start, max_concurrent=3)
    scheduler.tick()
    assert len(processes.running) == 3

    store.set_capacity(4)
    scheduler.tick()

    assert store.capacity() == 4
    assert len(processes.running) == 4
