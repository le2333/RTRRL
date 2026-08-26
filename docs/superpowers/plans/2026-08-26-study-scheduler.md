# Persistent Study Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one persistent local scheduler that runs at most four `trainerctl` studies concurrently and accepts both DRQN and RTRRL experiment configurations.

**Architecture:** A focused `scheduler.py` module owns SQLite task state and a polling dispatcher that launches `trainerctl run` children. A separate `scheduler_cli.py` exposes add/list/run commands, and one user systemd unit keeps the dispatcher alive. Study-level candidate parallelism remains entirely in each experiment's `hpo.parallel_jobs` setting.

**Tech Stack:** Python 3.11, standard-library `sqlite3`/`subprocess`/`argparse`, PyYAML, pytest, user systemd.

**Spec:** `docs/superpowers/specs/2026-08-26-study-scheduler-design.md`

## Global Constraints

- The dispatcher limit is exactly four concurrently running study controllers.
- The scheduler never changes experiment configuration or manages Aim.
- Each study uses its own Optuna SQLite database.
- Duplicate canonical configuration/database pairs are idempotent.
- Failed studies are recorded and never retried automatically.
- The scheduler uses IAM identity by removing `AWS_PROFILE` and directing AWS config and shared credentials to `/dev/null`.
- Do not run pytest, workers, Docker, Aim, or training on the permanently-on micro host; run only static checks there and use CI for the full test suite.

---

### Task 1: Persistent task store

**Files:**
- Create: `infra/src/trainer_infra/scheduler.py`
- Create: `infra/tests/test_scheduler.py`

**Interfaces:**
- Produces: `TaskState`, `StudyTask`, and `TaskStore`.
- `TaskStore.add(config: Path, catalog: Path, database: Path) -> StudyTask`
- `TaskStore.list() -> tuple[StudyTask, ...]`
- `TaskStore.claim(limit: int) -> tuple[StudyTask, ...]`
- `TaskStore.finish(task_id: int, exit_code: int, reason: str | None) -> None`
- `TaskStore.interrupt_orphans(live: Callable[[StudyTask], bool]) -> tuple[StudyTask, ...]`

- [ ] **Step 1: Write failing persistence and idempotency tests**

```python
def test_add_is_idempotent_for_canonical_config_and_database(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "scheduler.sqlite")
    first = store.add(tmp_path / "experiment.yml", tmp_path / "catalog.json", tmp_path / "run.sqlite")
    second = store.add(tmp_path / "./experiment.yml", tmp_path / "catalog.json", tmp_path / "run.sqlite")
    assert first.id == second.id
    assert [task.state for task in store.list()] == [TaskState.QUEUED]


def test_claim_and_finish_are_persistent(tmp_path: Path) -> None:
    database = tmp_path / "scheduler.sqlite"
    task = TaskStore(database).add(tmp_path / "a.yml", tmp_path / "catalog.json", tmp_path / "a.sqlite")
    assert TaskStore(database).claim(1)[0].id == task.id
    TaskStore(database).finish(task.id, 0, None)
    assert TaskStore(database).list()[0].state is TaskState.SUCCEEDED
```

- [ ] **Step 2: Run the focused tests in CI and verify RED**

Run remotely: `uv run --project infra pytest infra/tests/test_scheduler.py -q`
Expected: collection failure because `trainer_infra.scheduler` does not exist.

- [ ] **Step 3: Implement the SQLite schema and atomic state transitions**

Create `TaskState(str, Enum)`, frozen `StudyTask`, and `TaskStore`. Canonicalize paths with `Path.resolve()`, enforce `UNIQUE(config, database)`, use `BEGIN IMMEDIATE` while claiming oldest queued rows, set a 5-second SQLite busy timeout, and store ISO-8601 UTC timestamps.

- [ ] **Step 4: Add orphan reconciliation tests and implementation**

```python
def test_missing_running_controller_is_marked_interrupted(tmp_path: Path) -> None:
    store = populated_store(tmp_path)
    running = store.claim(1)[0]
    interrupted = store.interrupt_orphans(lambda task: False)
    assert [task.id for task in interrupted] == [running.id]
    task = store.list()[0]
    assert task.state is TaskState.FAILED
    assert task.reason == "controller interrupted"
```

Implement reconciliation so a task remains running only when `live(task)` is true; it must never transition back to queued.

- [ ] **Step 5: Run static lint and commit**

Run on this host: `uv run --project infra ruff check infra/src/trainer_infra/scheduler.py infra/tests/test_scheduler.py`

```bash
git add infra/src/trainer_infra/scheduler.py infra/tests/test_scheduler.py
git commit -m "feat(infra): persist scheduled studies"
```

### Task 2: Four-slot dispatcher

**Files:**
- Modify: `infra/src/trainer_infra/scheduler.py`
- Modify: `infra/tests/test_scheduler.py`

**Interfaces:**
- Consumes: `TaskStore` and `StudyTask` from Task 1.
- Produces: `Scheduler(store: TaskStore, command: Callable[[StudyTask], Sequence[str]], max_concurrent: int = 4, poll_seconds: float = 5.0)`.
- `Scheduler.tick() -> None` reaps terminal children and fills open slots.
- `Scheduler.run() -> None` reconciles once and polls indefinitely.

- [ ] **Step 1: Write a failing four-slot test with controllable fake processes**

```python
def test_tick_never_runs_more_than_four_studies(tmp_path: Path) -> None:
    processes = FakeProcessFactory()
    scheduler = Scheduler(store_with_tasks(tmp_path, 6), processes.start, max_concurrent=4)
    scheduler.tick()
    assert len(processes.running) == 4
    processes.finish_one(0)
    scheduler.tick()
    assert len(processes.running) == 4
    assert len([task for task in scheduler.store.list() if task.state is TaskState.QUEUED]) == 1
```

- [ ] **Step 2: Run the focused test remotely and verify RED**

Run remotely: `uv run --project infra pytest infra/tests/test_scheduler.py::test_tick_never_runs_more_than_four_studies -q`
Expected: failure because `Scheduler` is not defined.

- [ ] **Step 3: Implement child ownership, slot filling, and failure isolation**

Use `subprocess.Popen` without a shell, keep `task_id -> Popen` in memory, stream inherited stdout/stderr to the journal, and call `finish` for every observed exit. A nonzero child exit records `trainerctl exited with code N` and then fills the slot with the next task.

- [ ] **Step 4: Test recovery and failure isolation**

```python
def test_failed_child_does_not_block_next_task(tmp_path: Path) -> None:
    processes = FakeProcessFactory()
    scheduler = Scheduler(store_with_tasks(tmp_path, 5), processes.start, max_concurrent=4)
    scheduler.tick()
    processes.finish_one(7)
    scheduler.tick()
    states = [task.state for task in scheduler.store.list()]
    assert states.count(TaskState.FAILED) == 1
    assert len(processes.running) == 4
```

Also assert startup preserves a recorded controller only when `/proc/<pid>/cmdline` matches the expected config and database arguments; otherwise it marks the task interrupted.

- [ ] **Step 5: Run static lint and commit**

Run: `uv run --project infra ruff check infra/src/trainer_infra/scheduler.py infra/tests/test_scheduler.py`

```bash
git add infra/src/trainer_infra/scheduler.py infra/tests/test_scheduler.py
git commit -m "feat(infra): dispatch four studies concurrently"
```

### Task 3: Scheduler CLI

**Files:**
- Create: `infra/src/trainer_infra/scheduler_cli.py`
- Modify: `infra/pyproject.toml`
- Create: `infra/tests/test_scheduler_cli.py`

**Interfaces:**
- Consumes: `TaskStore` and `Scheduler`.
- Produces executable `study-scheduler = "trainer_infra.scheduler_cli:main"`.
- Commands: `add`, `list`, and `run` with global `--state DATABASE`.

- [ ] **Step 1: Write failing add/list CLI tests**

```python
def test_add_then_list_reports_human_readable_task(tmp_path: Path, capsys: Any) -> None:
    paths = valid_task_files(tmp_path)
    assert main(["--state", str(tmp_path / "queue.sqlite"), "add", *paths]) == 0
    assert main(["--state", str(tmp_path / "queue.sqlite"), "list"]) == 0
    output = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert output[0]["name"] == "R1-1-Minesweeper-DRQN-LSTM"
    assert output[0]["state"] == "queued"
```

- [ ] **Step 2: Run remotely and verify RED**

Run remotely: `uv run --project infra pytest infra/tests/test_scheduler_cli.py -q`
Expected: collection failure because `scheduler_cli` does not exist.

- [ ] **Step 3: Implement CLI validation and sanitized trainer command**

`add` must require existing config and catalog files, parse the YAML experiment name, reject a scheduler database equal to a study database, and print the stored task as JSON. `run --max-concurrent 4 --poll-seconds 5` constructs children equivalent to:

```text
trainerctl run CONFIG --backend batch --catalog CATALOG --database DATABASE --poll-seconds 20
```

Child environments remove `AWS_PROFILE`, set `AWS_CONFIG_FILE=/dev/null` and `AWS_SHARED_CREDENTIALS_FILE=/dev/null`, and retain the service's `PYTHONPATH`.

- [ ] **Step 4: Test malformed input and exact environment/argv**

Assert missing files return exit 2, malformed YAML returns exit 2, and the fake launcher receives no `AWS_PROFILE` plus the exact paths and run-queue defaults.

- [ ] **Step 5: Run static lint and commit**

Run: `uv run --project infra ruff check infra/src/trainer_infra/scheduler_cli.py infra/tests/test_scheduler_cli.py infra/pyproject.toml`

```bash
git add infra/src/trainer_infra/scheduler_cli.py infra/tests/test_scheduler_cli.py infra/pyproject.toml
git commit -m "feat(infra): add study scheduler CLI"
```

### Task 4: Persistent systemd deployment

**Files:**
- Create: `infra/systemd/r1-study-scheduler.service`
- Create: `scripts/install-study-scheduler.sh`
- Create: `scripts/check-study-scheduler-service.sh`

**Interfaces:**
- Consumes: installed `study-scheduler` entry point from Task 3.
- Produces one user service named `r1-study-scheduler.service` using scheduler state `runs/study-scheduler.sqlite` and concurrency four.

- [ ] **Step 1: Write a failing static service check**

The check must assert `Restart=on-failure`, `RestartSec=5`, the exact state database, `--max-concurrent 4`, `AWS_CONFIG_FILE=/dev/null`, and `AWS_SHARED_CREDENTIALS_FILE=/dev/null`.

- [ ] **Step 2: Run the check and verify RED**

Run: `bash scripts/check-study-scheduler-service.sh`
Expected: failure because the unit file does not exist.

- [ ] **Step 3: Add the unit and idempotent installer**

The installer copies the versioned unit to `$XDG_CONFIG_HOME/systemd/user/`, runs `systemctl --user daemon-reload`, and enables/starts the service. It must not delete or overwrite scheduler/Optuna databases.

- [ ] **Step 4: Run static verification and commit**

Run: `bash scripts/check-study-scheduler-service.sh && bash -n scripts/install-study-scheduler.sh`

```bash
git add infra/systemd/r1-study-scheduler.service scripts/install-study-scheduler.sh scripts/check-study-scheduler-service.sh
git commit -m "feat(infra): deploy persistent study scheduler"
```

### Task 5: CI verification and R1.1 migration

**Files:**
- Create: `scripts/enqueue-r1-1-drqn-hpo.sh`
- Create: `scripts/check-enqueue-r1-1-drqn-hpo.sh`
- Modify: `.github/workflows/ci.yml` only if existing CI does not already run all `infra/tests`.

**Interfaces:**
- Consumes: `study-scheduler add` from Task 3 and the service from Task 4.
- Produces an idempotent migration command for the ten unfinished DRQN HPO studies; future RTRRL configs use the same CLI directly.

- [ ] **Step 1: Write a failing static migration check**

Assert the enqueue script contains exactly the ten unfinished configurations, excludes both completed DiscountingChain configurations, assigns one distinct Optuna database per config, and never invokes `systemd-run`.

- [ ] **Step 2: Run the static check and verify RED**

Run: `bash scripts/check-enqueue-r1-1-drqn-hpo.sh`
Expected: failure because the enqueue script does not exist.

- [ ] **Step 3: Implement idempotent enqueueing**

Loop over the ten configuration filenames and invoke `study-scheduler --state "$ROOT/runs/study-scheduler.sqlite" add ...`. Use `runs/r1-1-drqn-hpo-2m-auc-system/CONFIG.sqlite` for study databases. Do not include completed DiscountingChain tasks.

- [ ] **Step 4: Push and run CI before deployment**

Run static checks locally, commit, push the branch, and require green remote tests for `infra/tests/test_scheduler.py` and `infra/tests/test_scheduler_cli.py` before installing the service.

```bash
git add scripts/enqueue-r1-1-drqn-hpo.sh scripts/check-enqueue-r1-1-drqn-hpo.sh .github/workflows/ci.yml
git commit -m "ops: enqueue remaining R1.1 studies"
git push
```

- [ ] **Step 5: Migrate without duplicating the active study**

Record the current Minesweeper-LSTM controller as the externally occupied fourth slot during transition. Install/start the scheduler with three available slots, enqueue the other nine unfinished studies, and do not enqueue Minesweeper-LSTM until its current controller reaches a terminal state. Then set the persistent limit to four and confirm the scheduler has at most four `trainerctl` children.

- [ ] **Step 6: Verify live state**

Run `study-scheduler --state runs/study-scheduler.sqlite list`, `systemctl --user status r1-study-scheduler.service`, and `ps -eo pid,ppid,rss,stat,cmd`. Confirm Aim remains active, no configuration/database pair appears twice, and no more than four study controllers run.
