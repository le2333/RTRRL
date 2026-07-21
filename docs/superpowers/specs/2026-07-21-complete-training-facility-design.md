# Complete Training Facility Design

## Status and Scope

This specification defines the remaining work required to turn the completed
configuration, sampling, observability SDK, memo trace, heavy-test, and AWS
resource foundations into one usable greenfield training facility.

The supported first release:

- runs as a foreground local `trainerctl run` process;
- accepts one experiment YAML and completes every HPO batch in that command;
- supports exactly `memo_stream_ac` with `agent_type=rtu_rtrl` and
  `memo_rtrrl` with `rtrrl_topology=shared`;
- treats environment selection and options as experiment parameters;
- uses the four exact resource profiles `c7am`, `c7al`, `c7ax`, and `g6x`;
- submits formal jobs only to the four run queues;
- records metrics and episode summaries in Aim and complete selected evaluation
  episodes in Rerun;
- preserves every historical shell, HPO, descriptor, image, and workflow entry.

There is no resume command, history import, shared search space, daemon,
automatic Batch resubmission, queue mutation, or legacy-entry cleanup.

## Existing Foundations

The following are already implemented and remain authoritative:

- strict experiment and script descriptor models;
- image digest resolution and catalog codec;
- independent group resolution, Optuna sampling, finite-space tracking, and
  concrete run materialization;
- the repository-level `training-sdk`, Aim spool, Rerun publication, and SDK
  bootstrap;
- generic environment contracts and complete memo evaluation traces;
- the heavy-test image/runner facility;
- four managed compute environments and eight dev/run queues.

The remaining implementation must reuse those contracts rather than introduce a
parallel configuration or observability system.

## Resource Profiles

Formal resources are exact physical profiles:

| Profile | Run queue | Compute environment | Request |
| --- | --- | --- | --- |
| `c7am` | `run-cpu-c7am-queue` | `rtrrl-cpu-c7am-ce` | 1 vCPU, 1600 MiB |
| `c7al` | `run-cpu-c7al-queue` | `rtrrl-cpu-c7al-ce` | 2 vCPU, 3200 MiB |
| `c7ax` | `run-cpu-c7ax-queue` | `rtrrl-cpu-c7ax-ce` | 4 vCPU, 7168 MiB |
| `g6x` | `run-gpu-queue` | `rtrrl-gpu-g6x-ce` | 4 vCPU, 12000 MiB, 1 GPU |

The matching dev queues have priority 10; run queues have priority 100. The
four compute environments are shared capacity pools. Queue priority affects
waiting jobs but never preempts a running job.

Runtime code performs read-only profile preflight and never creates, updates,
scales, disables, or deletes queues or compute environments.

## Execution Data

Every concrete run produces:

- immutable canonical config YAML and hash;
- a complete SDK `RunContext`;
- exact experiment, group, run, trial, seed, script, image digest, and profile
  identity;
- one argv command with a concrete config path;
- one S3 artifact prefix.

Runs are packed by image digest, profile, and `runs_per_job`. A job bundle
contains deterministic ordered child runs and hashes for every input object.

S3 is an exchange and artifact channel:

```text
experiments/<experiment-id>/
  groups/<group>/runs/<run-id>/
    input/config.yaml
    input/run-context.json
    status/attempt-0.json
    aim-buffer/
    rerun/
    checkpoints/
  jobs/<job-id>/bundle.json
```

The first release has exactly one execution attempt, numbered zero.

## Batch Adapter and Worker

The Batch adapter has only two responsibilities:

1. submit a prepared bundle once to a preconfigured digest-bound job
   definition and exact run queue;
2. query submitted job IDs.

It does not register job definitions, retry, resubmit, cancel, roll back, clean
partial submissions, or classify failures. AWS native retry attempts are fixed
to one.

Job definitions are registered by an explicit deployment step after a formal
memo image digest is known. Identity includes image digest, profile, worker
protocol, roles, and logging configuration.

The image-provided worker:

1. downloads and verifies the bundle and child inputs;
2. creates the SDK run-context file and exports
   `TRAINER_RUN_CONTEXT_PATH`;
3. starts each launcher with `shell=False`;
4. uploads its completion marker and registered artifacts;
5. stops the bundle immediately after the first failed child.

The controller stops scheduling future work when any Batch job or child run
fails. It returns a structured failure containing every submitted job ID. It
does not cancel jobs that AWS has already accepted.

## Memo Launchers

The memo image exposes exactly two new facility scripts:

- `memo_stream_ac`: environments `memory_chain`, `kmemory_chain`, and
  `mujoco_masked`; `agent_type` fixed to `rtu_rtrl`;
- `memo_rtrrl`: environment `hopper`; `rtrrl_topology` fixed to `shared`.

Environment options, including `max_episode_steps`, are loaded from the
concrete config.

Both launchers call `bootstrap_from_environment()` outside JIT regions.
Training episode summaries come from completed `RecordEpisodeStatistics`
records and use the real host-visible `state.step`. Complete evaluation traces
are converted to SDK `Episode` values only when they contain a full episode.
Rerun sampling uses evaluation episodes and never moves training transitions
out of JIT.

Legacy RTRRL, PPO, SAC, QRC, TBPTT, independent topology, examples, descriptors,
workflows, and shell entry points remain present and unchanged.

## Memo Image

The memo CPU and GPU images use repository-root build context so they can copy:

- memo source and lock;
- the single `training-sdk`;
- `/opt/trainer/worker.py`;
- `/opt/trainer/scripts/index.yaml` and the two memo descriptors.

The immutable image carries a nonempty
`org.rtrrl.trainer.scripts.v1` label. CPU/GPU dependency versions must be
lock-consistent. The memo catalog is independent from retained legacy catalogs.

## Controller and CLI

`trainerctl validate` performs no writes and verifies:

- experiment and catalog resolution;
- ECR tag-to-digest and digest-bound catalog;
- four profile and job-definition contracts;
- S3, Aim, IAM, queue, and compute-environment reachability.

`trainerctl run` creates one experiment ID and independently advances every
group:

```text
ask -> materialize -> pack -> upload -> submit -> query
-> read completion/Aim -> tell -> next batch
```

The single command is authorization for every trial declared by the experiment
YAML. `study.tell()` executes only in the controller thread. Finite spaces may
finish before the nominal budget.

Aim collection accepts only the exact run ID, finalized marker, expected
objective metric, and a finite objective value. Waiting for Batch or Aim is
bounded polling, not retry or resubmission.

Any failed job, child, missing completion marker, Aim timeout, or invalid
objective terminates the complete experiment command. No later HPO batch is
submitted.

The first release exposes only `validate` and `run`; it has no `status`,
`resume`, or history command.

## Historical Compatibility

The implementation must not delete, rename, or replace:

- `infra/submit.sh`, `submit_many.sh`, `hpo.sh`, `sweep.sh`,
  `build-and-push.sh`, or `backup-aim.sh`;
- legacy descriptors under `rtrrl/infra/scripts`;
- legacy Docker workflows, images, or job definitions;
- existing HPO data and generated plans.

Compatibility tests assert that historical files and their existing help or
dry-run behavior remain available. The new `trainerctl` is an additional
entry, not a wrapper around historical commands.

## Verification and Deployment

Local acceptance requires:

- contract, S3, worker, Batch, launcher, Aim, controller, and CLI tests;
- a fake two-group automatic `2+2+1` HPO run;
- failure-injection tests proving no future batch is submitted;
- full control-plane, SDK, and targeted memo suites;
- Ruff, lock checks, and CPU/GPU image checks.

Real acceptance requires separate authorization for each mutating or paid
phase:

1. restore and verify the isolated Aim scratch service on port 53801;
2. build and push immutable memo CPU/GPU images;
3. register four digest-bound single-attempt job definitions;
4. run a small real `trainerctl run` covering both launchers, CPU and GPU,
   parallel jobs, serial child runs, and automatic multiple HPO batches;
5. verify Batch/CloudWatch, L4/JAX, Aim scratch, Rerun, S3 markers/artifacts,
   and Optuna completion;
6. clean only that smoke experiment after separate authorization.

Historical data, entries, images, job definitions, queues, and compute
environments are never cleanup targets.

## User Manual

After real acceptance, `infra/README.md` becomes the authoritative facility
manual. It must describe only commands that were actually tested, current Batch
topology, profiles, images/catalogs, `trainerctl validate/run`, heavy tests,
Aim/Rerun/S3 lookup, failure semantics, preflight, and the explicit status of
historical entries.
