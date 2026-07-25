# Algorithm-Centric Training Facility Design

**Date:** 2026-07-25
**Status:** Approved design. Implementation plan:
`docs/superpowers/plans/2026-07-25-algorithm-centric-training-facility.md`
**Supersedes:** `2026-07-20-lightweight-training-infrastructure-design.md`,
`2026-07-20-training-control-plane-design.md`,
`2026-07-20-training-observability-sdk-design.md`,
`2026-07-21-complete-training-facility-design.md`

## Purpose

One `trainerctl run` call executes one complete hyperparameter study for one
algorithm on AWS Batch, from the local machine, in the foreground, and exits
when the study is finished or when anything fails.

The facility owns configuration resolution, parameter sampling, job submission,
serial execution of runs inside jobs, score extraction, and reporting. It owns
no algorithm logic and makes no judgement about the meaning of any metric.

## What This Replaces

The current implementation is restructured, not extended. The following
existing behaviours are removed:

- the `experiment` / `defaults` / `groups` layering in the experiment file, and
  the resolution machinery that decides which layer wins;
- multiple algorithms or multiple parameter groups per experiment file;
- platform-side parameter enumeration, explicit scans, duplicate detection with
  resample caps, and any parameter generation that does not come from Optuna;
- the S3 spool in the SDK and every artifact that exists only to survive a
  crash;
- the control plane reading metrics back from Aim, and therefore the Aim
  scratch repository, the Aim process identity gate, and the read-only
  `Run.close()` compatibility shim;
- per-run status and completion marker artifacts;
- Batch-level retry attempts above one;
- byte-exact verification of the remote script catalog against a local copy;
- deep-freezing of in-memory configuration structures;
- resource profile abstraction over instance types.

## Principles

1. One experiment file describes one algorithm, one search space, one budget,
   one study. The `experiment` field is an archive label, not a container of
   workloads.
2. Every parameter value that reaches a training script is produced by Optuna.
   Fixed values are expressed as single-choice distributions, so there is one
   code path and one document.
3. The facility checks structure, never merit. It rejects declarations that
   contradict each other, because those guarantee a useless run and are visible
   before anything is spent: a score window outside the declared step budget, a
   score metric the entry does not report, a parameter the entry does not
   accept. It never judges whether a value is good. A diverging run, an extreme
   learning rate, and a large negative return are all legitimate, and the person
   who wrote the experiment can judge them without help.
4. Any abnormal exit terminates the launch. There are no retries, no
   continuation, no partial success. An abnormal exit means an unhandled
   boundary, which requires a source or configuration change, which requires a
   new launch anyway.
5. Divergence is normal algorithm behaviour, not failure. A bad score is a
   score.
6. Nothing is spent before every locally checkable precondition has passed.
7. Aim and the HPO study hold different things. Aim is the authoritative
   record of what training produced. The study is Optuna's own state, and is
   deletable at any time without loss.

## Experiment Configuration

The experiment file is flat and self-contained. There is no inheritance between
sections, no facility-level defaults file, and no group layer.

```yaml
# One file = one algorithm = one Optuna study = one `trainerctl run`.
experiment: locomotion          # Aim experiment; reused if it exists, created if not
name: walker-lr-sweep           # study name; (experiment, name, launch_id) identifies a launch
description: PPO learning rate and entropy sweep on walker2d

image: trainer-gpu:latest       # tag; resolved to a digest once at launch and frozen
entry: brax_ppo                 # entry declared by the image's script catalog
storage: s3://training-data     # exchange root for configs, scores, and episode files

compute:
  instance_type: g6.xlarge      # must map to an existing queue
  timeout_minutes: 240          # Batch attempt timeout; backstop for a hung worker
  startup_minutes: 15           # allowance before the first heartbeat
  stall_factor: 10              # stall when silence exceeds this many median intervals

hpo:
  sampler: tpe
  rounds: 5
  trials_per_round: 8
  parallel_jobs: 2              # 8 trials split across 2 jobs; each job runs 4 serially

space:                          # overrides the entry's declared space, key by key
  env:           [walker2d]
  backend:       [generalized]
  num_envs:      [2048]
  total_steps:   [50000000]
  seed:          [0]
  learning_rate: {type: float, low: 1.0e-6, high: 1.0e-3, log: true}
  entropy_cost:  {type: float, low: 1.0e-4, high: 1.0e-2, log: true}

score:
  metric: episode_return
  window_steps: [45000000, 50000000]
  reduce: mean
  direction: maximize
  non_finite: worst             # substitute an ordered-worst value for NaN and inf

logging:
  aim: aim://172.31.62.192:53801
  every_steps: 100000
  rerun_every_episodes: 100     # omit to disable Rerun recording
```

Two launches of the same `experiment` and `name` are sequential, independent,
and distinguished by `launch_id`. This is the normal way to rerun a study after
fixing a script or narrowing a range.

## Script Metadata in the Image

Each image declares its entries. One build step emits the declaration twice: as
a file at a fixed path inside the image, which the worker reads to find an
entry's command, and as an ECR image label, which the control plane reads
without running the container. Because both come from one source in one step,
they are not verified against each other.

`contract` is the single version number of the agreement between the control
plane and the SDK package. It covers both this declaration and the run
configuration schema, because both are read by the same package.

```json
{
  "contract": 2,
  "entries": {
    "brax_ppo": {
      "command": ["python", "-m", "brax_ppo.train"],
      "source_hash": "sha256:41b0f1c9...",
      "metrics": ["episode_return", "episode_length", "policy_loss"],
      "space": {
        "env": ["walker2d", "halfcheetah", "ant"],
        "backend": ["generalized", "mjx"],
        "num_envs": {"type": "int", "low": 256, "high": 8192, "step": 256},
        "total_steps": {"type": "int", "low": 1000000, "high": 200000000},
        "seed": {"type": "int", "low": 0, "high": 65535},
        "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-2, "log": true},
        "entropy_cost": {"type": "float", "low": 1e-5, "high": 1e-1, "log": true}
      }
    }
  }
}
```

`source_hash` is computed at image build time over a declared set of source
paths belonging to the entry. It identifies the algorithm across image
rebuilds: an unrelated dependency bump changes the digest but not the
`source_hash`.

`source_hash` never substitutes for the digest. The digest identifies the whole
environment, including the CUDA layer, and is the only thing that reproduces a
result exactly. Both are recorded.

All space entries must be scalars: number, string, or boolean. Structured
parameters are declared as scalars by the script, for example `hidden_width`
and `hidden_depth` rather than a list.

`total_steps` is a reserved parameter name. Every entry declares it in its
space, where it may be fixed or searched like any other parameter. What a step
means is the script's business: environment steps, gradient steps, or frames are
all acceptable, and the facility never interprets the unit. The only contract is
that the step value reported alongside each metric uses the same unit as
`total_steps`, which is what makes comparing the score window against the budget
a numeric comparison rather than a semantic one.

`metrics` lists the metric names the entry reports, so that preflight can reject
a typo in `score.metric` instead of discovering it after a round has been paid
for.

The cost of `metrics` is that a stale list rejects a valid configuration. The
error names both sides, so the fix is obvious, but it does require an image
rebuild. This is the one place in the design where an out-of-date declaration
can block correct usage.

## Search Space Resolution

The resolved space is the entry's declared space with the experiment file's
`space` block applied key by key. Every key appears exactly once, either as a
distribution or as a single-choice list.

A single-choice list is a categorical distribution with one option. Sampling it
is deterministic. As a result:

- there is no separate "fixed parameters" document and no merge conflict rule;
- `params` handed to a script is exactly `trial.params`;
- the study records the complete configuration of every trial, not only the
  varied part;
- an experiment key that the entry does not declare is a typo or an unsupported
  parameter, and is rejected before anything is submitted.

The resolved space is printed to the terminal and archived. Nothing about the
effective space is implicit.

## Identity and Layout

`launch_id` is a UTC timestamp to the second, for example `20260725-051400`. It
is a timestamp rather than a content hash precisely because rerunning an
identical configuration must produce a distinct launch.

`run_id` is `<name>-<launch_id>-t<trial>`. Aim run names carry the launch id, so
score readback and comparison can never pick up a previous launch's data.

S3 exchange, under `storage`:

```
<storage>/<experiment>/<name>/<launch_id>/
  experiment.yaml               # copy as submitted
  space.json                    # resolved space
  launch.json                   # description, digest, source_hash, contract, sampler, budget
  rounds/round-000/job-0.json   # list of config URIs this job runs serially
  trials/t7/config.json         # written by ctl
  trials/t7/score.json          # written by the worker
  trials/t7/episodes/*.rrd      # written by the SDK
  report.json                   # written by ctl at the end
```

Local archive, under `~/.trainer/launches/<experiment>/<name>/<launch_id>/` and
overridable with `--archive-dir`: the same metadata, plus `study.db` and the log
tails of failed jobs.

## Module: Preflight

Runs before anything is created or submitted. Local computation plus read-only
AWS calls. Region and credentials come from the standard AWS environment, and
the queue table is scoped to that region.

Checks:

- the experiment file parses and every required field is present;
- the image tag resolves to a digest;
- the image label carries a catalog whose `contract` this control plane
  understands, and which declares the requested `entry`;
- the resolved space contains only scalar values and declares `total_steps`;
- every key in the experiment's `space` block is declared by the entry;
- `instance_type` maps to an existing queue and to a digest-bound job
  definition;
- `hpo.sampler` is `tpe`, `random`, or `grid`, and `grid` is used only when every
  entry in the resolved space is a discrete list;
- `hpo.parallel_jobs` is at least one and at most `trials_per_round`;
- `score.metric` is listed in the entry's `metrics`;
- `score.window_steps` is ordered and lies within the step budget: its upper
  bound does not exceed the smallest value the resolved space can produce for
  `total_steps`;
- `score.reduce`, `score.direction`, and `score.non_finite` are known values;
- the `storage` bucket exists and is accessible;
- a TCP connection to the Aim endpoint succeeds.

The window check uses the smallest producible step budget rather than the
largest, because a window that some trials cannot reach guarantees that those
trials produce nothing.

The Aim check exists for one mistake: starting a launch without the Aim server
running, which would otherwise cost a full round before anything reports it. It
proves only that the port accepts a connection. Identifying the process is not
attempted, and on this host no other service competes for the port anyway;
connecting to something that is not Aim fails at the first write, which is the
correct outcome.

Output: the resolved space printed to the terminal, the queue's derived
concurrency limit printed alongside `parallel_jobs`, and a validated launch plan
in memory. Requesting more parallel jobs than the queue can run concurrently is
not an error; the excess waits in the queue.

Does not: create AWS resources, write to S3, submit jobs, create an Aim run.

## Module: HPO Loop

Optuna, one study per launch, storage is a SQLite file in the local archive
directory. The study is single-process: only the control plane calls `ask` and
`tell`, and workers never know Optuna exists. Optuna's recommendation of a
shared relational store applies to distributed multi-worker sampling and does
not apply here.

Study `user_attrs` record `experiment`, `name`, `launch_id`, `entry`,
`source_hash`, image digest, and the resolved space, so the file is
self-describing.

There is no deduplication. With any continuous parameter in the space, exact
repeats do not occur; in a fully discrete space, repeats are Optuna's decision
and the correct response is a grid sampler or a smaller trial count, not a
platform-side cache.

There is no resume. A rerun is a new launch with a new study.

Does not: generate, filter, clamp, or reorder parameters; enumerate the space;
read Aim.

## Module: Run Configuration Generation

Input: a trial's number and `trial.params`, the resolved launch plan, the
launch id. Deterministic and local.

```json
{
  "contract": 2,
  "run_id": "walker-lr-sweep-20260725-051400-t7",
  "experiment": "locomotion",
  "name": "walker-lr-sweep",
  "launch_id": "20260725-051400",
  "trial": 7,
  "entry": "brax_ppo",
  "params": {
    "env": "walker2d",
    "backend": "generalized",
    "num_envs": 2048,
    "total_steps": 50000000,
    "seed": 0,
    "learning_rate": 3.17e-4,
    "entropy_cost": 0.0021
  },
  "logging": {
    "aim": "aim://172.31.62.192:53801",
    "every_steps": 100000,
    "rerun_s3": "s3://training-data/locomotion/walker-lr-sweep/20260725-051400/trials/t7/episodes/",
    "rerun_every_episodes": 100
  },
  "score": {
    "metric": "episode_return",
    "window_steps": [45000000, 50000000],
    "reduce": "mean",
    "direction": "maximize",
    "non_finite": "worst",
    "s3": "s3://training-data/locomotion/walker-lr-sweep/20260725-051400/trials/t7/score.json"
  }
}
```

`params` is passed to the script verbatim.

`contract` is checked in both directions, which matters because reruns pin old
digests. The control plane refuses an image whose catalog declares a contract it
does not understand, and the worker refuses a configuration whose contract it
does not understand. Either way the message names both versions instead of
surfacing a missing field somewhere deeper.

Does not: validate parameter semantics, supply defaults, inject a seed, choose
a queue or instance, contact S3.

## Module: Job Packing and Upload

Serial execution inside a job is required: many runs finish in minutes, and
paying instance startup and image pull per run is wasteful at that duration.

A round's trials are split across `parallel_jobs` jobs in order, with the
remainder distributed to the earliest jobs, so eight trials across three jobs
become three, three, and two. Each job gets a manifest listing the S3 URIs of
the run configurations it executes in order.
Because the manifest is referenced by a single URI, there is no limit on runs
per job; the previous limit of four came from inlining configurations into the
Batch submission payload.

All configuration objects and manifests are uploaded before any job is
submitted, so a worker cannot observe a missing object. S3 is strongly
consistent, so no read-back verification is performed.

Does not: submit jobs, decide instance types, retry uploads beyond the AWS SDK
default.

## Module: Submit and Wait

One `SubmitJob` per job, passing only the manifest URI in the environment. The
job definition is bound to the resolved digest and declares one attempt. Array
jobs are not used, because each job has a distinct manifest and a round contains
at most a few dozen jobs.

The queue comes from a fixed table validated in preflight. Each existing compute
environment is pinned to exactly one instance type and is fronted by two queues:

| `instance_type` | Compute environment | `run` queue | Max vCPUs |
| --- | --- | --- | --- |
| `c7a.medium` | `rtrrl-cpu-c7am-ce` | `run-cpu-c7am-queue` | 16 |
| `c7a.large` | `rtrrl-cpu-c7al-ce` | `run-cpu-c7al-queue` | 32 |
| `c7a.xlarge` | `rtrrl-cpu-c7ax-ce` | `run-cpu-c7ax-queue` | 16 |
| `g6.xlarge` | `rtrrl-gpu-g6x-ce` | `run-gpu-queue` | 32 |

Every delivered command uses the `run` queues, including a rerun of a failed
study. The `dev` queues exist only for infrastructure development and belong to
no delivered workflow. The experiment file therefore has no lane field and the
control plane never selects `dev`.

Job definitions request the whole instance, so one job occupies one instance and
the concurrency limit follows from the compute environment's maximum vCPUs. This
also gives GPU runs exclusive access to the accelerator without a separate
mechanism.

`DescribeJobs` is polled at a fixed interval until every job in the round
reaches a terminal state. Runs last minutes to hours, so the interval is not
performance relevant.

Sibling jobs are never cancelled when one fails; the round is allowed to reach
a terminal state and the launch then stops.

On `SIGINT` or an unhandled control plane exception, every in-flight job is
terminated. This is the only cleanup handler in the design, and its purpose is
to stop spending, not to preserve data.

Output: per-job terminal state, plus trial number, job id, and log stream name
recorded in the round record.

Does not: create or modify queues, compute environments, or job definitions;
interpret failure reasons; read scores.

## Module: Worker

The worker ships in the SDK package. The image installs one package, the job
command is `python -m trainer_sdk.worker`, and worker and SDK are always the
same version, so the configuration contract has a single implementation.

For each configuration in its manifest, in order: download it, look up the
entry's command in the catalog file inside the image, start it as a separate
process with the configuration file path in the environment, watch the
heartbeat, and wait. S3 access uses the Batch job role; the worker holds no
credentials of its own.

Separate processes are what make serial execution safe: each run gets clean JAX
and accelerator state and cannot inherit global state from its predecessor.

Between runs, the worker deletes the finished run's scratch directory. Episode
recordings can be large and the container disk is limited.

Heartbeat: the SDK appends to a local metrics file on every report, so that
file's modification time is the heartbeat and no separate artifact is needed.
Before the first report, `startup_minutes` applies, because there is nothing to
measure yet and JIT compilation legitimately takes minutes. After a few
reports, the worker takes the median observed interval and declares a stall
when silence exceeds `stall_factor` times that median. On stall it sends
`SIGTERM`, allows a short grace period, sends `SIGKILL`, and exits non-zero.

`stall_factor` must stay loose because a mid-run recompilation legitimately
suppresses reporting for a while. A small factor misreads recompilation as a
hang.

If a child exits non-zero, the worker exits non-zero immediately and does not
start the remaining runs in its manifest.

Exit code semantics: zero means every run in the manifest completed and each
child exited zero. Non-zero means something went wrong, without distinguishing
which; the consequence is the same and the log says what happened.

Does not: parse or validate parameters; contact Aim; write episode files;
write status files; retry; decide what runs next.

## Module: SDK Sinks

The algorithm reports metrics periodically and, when configured, hands over
complete episodes. `every_steps` and `rerun_every_episodes` are passed to the
algorithm, which decides when to call. The SDK forwards every call and never
throttles, samples, or coalesces, because deciding that a report is unnecessary
would be a judgement about content. A consequence is that the heartbeat cadence
is whatever the algorithm actually does, not what the configuration asked for.

The SDK fans one report out to three consumers that do not know about each
other:

- Aim: a live RPC connection to the declared endpoint. The run is created at
  startup under the declared `experiment`, named `run_id`, with `launch_id`,
  `trial`, `entry`, `source_hash`, image digest, and every parameter recorded as
  run fields, so any launch can be isolated or compared in the Aim interface.
- Rerun: each complete episode written locally as `.rrd`, uploaded to the run's
  episode prefix as soon as that episode ends, then deleted locally, so disk use
  stays bounded by one episode. Disabled when `rerun_every_episodes` is absent.
- a local metrics file: one line per report, consumed by the worker for the
  score and used as the heartbeat.

The three are decoupled in data: the score no longer passes through Aim, Rerun
does not affect the score, and deleting Aim runs does not affect HPO.

They are not isolated in failure. Any sink error is an abnormal exit. The SDK
does not swallow exceptions, degrade, buffer to disk, or retry. If Aim is
unreachable the run crashes and the launch stops.

Does not: judge metric semantics, check value ranges, decide what to record,
compute the score.

## Module: Score Computation

The worker computes the score after the child exits, from the local metrics
file, using the run configuration's score block: select the reported points for
`metric` whose step lies inside `window_steps` inclusive, apply `reduce`, and
upload one small object. `reduce` is one of `mean`, `median`, `min`, `max`, or
`last`.

The step in `window_steps` is compared against the step the algorithm reported
alongside the metric. Preflight has already established that the window lies
within `total_steps`, so an empty window here means the run stopped early or
reported steps in a different unit than its budget.

The score is HPO's concern, so the algorithm process does not need to know the
score definition exists.

If the window contains no point for `metric`, the worker exits non-zero and
names the metric and the window. This locates the error at its source instead
of leaving the control plane to infer it from a missing object.

If the reduced value is NaN or infinite, `non_finite` applies. `worst`
substitutes an ordered-worst value for the declared direction. This is
declared, not invented by the platform, and it is better than recording a
failed trial: a failed trial carries no value and TPE learns nothing from it,
whereas an ordered-worst value teaches the sampler to avoid the region.

TPE splits trials by objective quantile and therefore uses only ordering, so an
extreme sentinel does not distort it. `GPSampler` fits a surrogate to the
values, so under that sampler `non_finite` must be a realistic bad value rather
than an extreme one.

## Module: Score Readback and Reporting

For each trial in the finished round, the control plane reads the score object
and calls `tell`. There is no aggregation, no query language, and no Aim
access, because the reduction already happened in the worker.

A succeeded job whose score object is absent should be unreachable, because a
worker that cannot compute or upload a score exits non-zero. No check is added
for it; reading a missing object fails, and the launch stops with that error.

After each round the control plane prints one line per trial with its
parameters and score. At the end it writes and prints a report containing every
trial's parameters, score, job id, and log stream name, the best trial, and
elapsed times.

When a job fails, the tail of its log stream is fetched and printed locally and
the full stream is written to the local archive, so the traceback appears in
the terminal that started the run.

The Batch log group is a dedicated group with a retention period rather than
the default `/aws/batch/job`.

## Command Surface

`trainerctl validate <file>` runs preflight and prints the resolved space, then
exits. It creates nothing and spends nothing.

`trainerctl run <file>` runs preflight and then the whole study.

There is no command to resume, retry, or continue a launch.

## Control Loop

```
preflight
  -> resolve digest, catalog, space; validate; print space
create launch
  -> launch id, archive directory, study, launch.json
for each round:
  -> ask trials_per_round trials
  -> generate configurations, upload configurations and manifests
  -> submit parallel_jobs jobs, poll until all terminal
  -> if any job failed: print log tail, write failed report, exit non-zero
  -> read scores, tell, print round summary
write final report, exit zero
```

The call blocks in the foreground for the whole study. It does not fork, does
not daemonise, and does not run in Batch.

Trials within a round are sampled before any of that round's results exist, so
they learn only from previous rounds. `trials_per_round` trades wall clock
against sample efficiency, and this is inherent to batching rather than a
defect.

## Failure Policy

| Condition | Detected by | Consequence |
| --- | --- | --- |
| Configuration, catalog, space, queue, or score declaration invalid | Preflight | Exit non-zero, nothing spent |
| Image tag unresolvable or catalog missing | Preflight | Exit non-zero, nothing spent |
| Upload or `SubmitJob` failure | Control plane | Exit non-zero |
| Child process exits non-zero | Worker | Remaining runs in the job skipped, worker exits non-zero |
| No heartbeat within `startup_minutes` or `stall_factor` medians | Worker | Child killed, worker exits non-zero |
| Score window contains no data point | Worker | Exit non-zero naming metric and window |
| Aim unreachable or write error | SDK, inside the child | Uncaught, child exits non-zero |
| Worker or container hangs entirely | Batch `timeout_minutes` | Job fails |
| Any job in a terminal state is not `SUCCEEDED` | Control plane, after the round | Log tail printed, failed report written, exit non-zero |
| Job succeeded but score object absent | Control plane | Exit non-zero |
| Score is NaN or infinite | Worker | `non_finite` substitution, run counts as complete |
| `SIGINT` or unhandled control plane exception | Control plane | In-flight jobs terminated, exit non-zero |

Recovering from any of these is a new launch. The archive of the failed launch
identifies where it stopped.

## Declared Limitations

These are consequences of the chosen structure, not defects to be fixed later
without a decision:

- The space is a declaration, so parameters cannot be conditional on each
  other. Optuna supports conditional spaces only when the space is expressed as
  code, which would make the effective space invisible.
- Joint feasibility constraints cannot be expressed. The intended response is
  reparametrisation, or deriving one parameter from another inside the script.
  Optuna 4.9 does support `constraints_func` on TPE, NSGA-II, and GP samplers,
  but it learns from violations after the fact, so using it would reintroduce
  proposal rejection and a retry bound. Deferred until a real constraint exists
  that cannot be reparametrised away.
- Parameters must be scalars.
- Trials inside a round do not learn from each other.
- A launch has one `compute` block, so one study cannot place some trials on CPU
  and others on GPU. Comparing an algorithm across instance types is two
  launches sharing an `experiment` label.
- There is no resume, no cross-launch score reuse, and no deduplication.
- Repeated launches of the same experiment and name accumulate archive
  directories. They are not cleaned automatically.
- Stall detection cannot distinguish a long recompilation from a hang except by
  magnitude.
- A stale `metrics` declaration in an image rejects a valid score metric until
  the image is rebuilt.
- `non_finite: worst` is safe under TPE because TPE is order-based, and unsafe
  under `GPSampler`.

## Testing Strategy

The failure that motivated this redesign was an Aim 3.28.0 read-only
`Run.close()` incompatibility that reached a paid AWS run because the test
double for an Aim run had a no-op `close()`. The rule that follows:

- No hand-written double for a third-party object whose lifecycle behaviour we
  depend on. Aim, Optuna, and Rerun are exercised against the real packages:
  Aim against a real server process and a temporary repository, Optuna against
  a real SQLite study performing real `ask` and `tell`, Rerun by writing and
  reading back a real recording.
- AWS Batch is the only hand-written double, and its job state machine is driven
  by recorded real `DescribeJobs` payloads rather than invented shapes, so schema
  drift is visible. S3 is exercised against a local S3-compatible server rather
  than a stub, so the real client, real paths, and real error types are used.
- A local end-to-end gate runs the entire control loop with job submission
  replaced by local process execution of the real worker and a real trainer,
  against a real Aim server and the local S3 server. This gate covers the
  configuration contract, the heartbeat, the score path, and the report, and
  would have caught the failure above.
- One paid AWS acceptance run follows a passing local gate, and requires
  explicit authorisation at that time.

## Out of Scope

- No change under `memo/`. Algorithm-side integration is separate work.
- All historical Batch entry points, including `infra/submit.sh`, are
  preserved.
- No new Batch queues, compute environments, or instance types. The eight
  existing queues, two lanes over four single-instance-type compute
  environments, are reused unchanged; only the log group changes.
- No dashboard, no web interface, no scheduler, no daemon.
