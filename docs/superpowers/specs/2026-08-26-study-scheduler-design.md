# Persistent Study Scheduler Design

## Purpose

Replace experiment-specific sequence services with one persistent local scheduler that can run DRQN, RTRRL, and future `trainerctl` studies. The scheduler limits control-plane memory use while preserving each study's own Batch-level parallelism.

## Scope

The scheduler manages `trainerctl run` controller processes on the permanently-on host. It does not manage AWS Batch workers directly, alter experiment configurations, or manage Aim. The initial global limit is four concurrently running studies. Each study continues to control its own candidate concurrency through `hpo.parallel_jobs`.

## Interface

Add a scheduler command-line interface under the existing control-plane package:

- `study-scheduler add CONFIG --catalog CATALOG --database DATABASE` adds one study.
- `study-scheduler list` reports queued, running, succeeded, and failed studies.
- `study-scheduler run` starts the long-lived dispatcher.

The scheduler database path and concurrency limit are service configuration. Adding the same canonical configuration path and database path twice is idempotent. A caller must use a new database path to request a genuinely new launch.

## Persistent State

Use one SQLite database owned by the scheduler. Each task stores:

- stable numeric identifier;
- canonical experiment configuration, catalog, and Optuna database paths;
- state: `queued`, `running`, `succeeded`, or `failed`;
- child PID when running;
- creation, start, and finish timestamps;
- exit code and concise failure reason.

SQLite transactions claim queued work and update terminal state. The scheduler database is separate from every study's Optuna database.

## Dispatch and Concurrency

The dispatcher maintains at most four live `trainerctl run` child processes. A slot represents one study controller, not one AWS Batch job. Therefore four studies may each submit the number of candidate jobs declared by their own `hpo.parallel_jobs` setting.

When a child exits, the scheduler records its result and immediately fills the free slot with the oldest queued task. A failed task does not stop unrelated tasks. Failed work is not retried automatically.

The scheduler launches children with the same sanitized AWS environment used by current scripts: `AWS_PROFILE` is absent, shared credential/config files point to `/dev/null`, and `PYTHONPATH` points to this checkout's control-plane source. Each child uses the run queues unless its task explicitly records another permitted tier.

## Restart Recovery

The scheduler runs as one user-level systemd service with automatic restart. On startup it reconciles tasks left in `running`:

- if the recorded child PID is still a live child with the expected command, resume monitoring it;
- otherwise mark the task failed with an interrupted-controller reason;
- never silently start a second controller against the same Optuna database.

Interrupted studies remain auditable and require an explicit operator decision to settle or enqueue a new launch. This avoids duplicate Batch submission after host or service failure.

## Current Migration

The currently running Minesweeper DRQN-LSTM controller remains untouched until it finishes. It counts against the four-study budget during migration. The remaining DRQN HPO configurations are enqueued once, excluding completed DiscountingChain studies. After the existing controller exits, all subsequent work is owned by the persistent scheduler. Future RTRRL configurations use the same `add` interface.

## Operations

The systemd service owns only the dispatcher. Aim remains in its existing service and receives reserved host capacity by limiting controllers to four. Operators inspect state with `study-scheduler list` and journal logs. Queue insertion and state transitions are logged with task identifiers and human-readable experiment names.

## Failure Handling

- Invalid paths or malformed configurations are rejected before insertion.
- A nonzero `trainerctl` exit marks only that task failed.
- Scheduler exceptions terminate the dispatcher so systemd restarts it and recovery runs.
- Automatic retries are intentionally absent because a failed study may already have submitted paid Batch work.
- SQLite uses a busy timeout and short transactions; child output is streamed to the journal rather than stored in the database.

## Verification

Unit tests use short fake child commands and temporary SQLite databases to verify:

- no more than four children run concurrently;
- a completed child causes the next queued task to start;
- one failed child does not block other tasks;
- duplicate additions are idempotent;
- restart recovery does not duplicate a live controller;
- a missing controller is marked interrupted rather than relaunched.

CLI tests cover add/list output and validation. Static service checks verify the persistent unit invokes the scheduler with concurrency four. No real worker, Aim server, Docker process, or AWS call runs on the permanently-on host during tests.
