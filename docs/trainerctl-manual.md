# trainerctl Manual

`trainerctl` runs one hyperparameter study for one algorithm on AWS Batch, from your
terminal, in the foreground, in a single command. It samples parameters with Optuna,
writes a configuration per trial, submits Batch jobs, waits, collects scores, and
starts the next round with what it learned. When it exits, the study is over.

This manual covers three audiences, in order: people running experiments, people
adding an algorithm to the facility, and people operating the AWS side. Read the
first part; read the others when you need them.

---

## Contents

- [Concepts](#concepts)
- [Part 1 — Running an experiment](#part-1--running-an-experiment)
  - [Prerequisites](#prerequisites)
  - [Quick start](#quick-start)
  - [The experiment file](#the-experiment-file)
  - [The search space](#the-search-space)
  - [Scoring](#scoring)
  - [Command reference](#command-reference)
  - [Output and where things land](#output-and-where-things-land)
  - [When something fails](#when-something-fails)
- [Part 2 — Adding an algorithm](#part-2--adding-an-algorithm)
- [Part 3 — Operations](#part-3--operations)
- [Sharp edges](#sharp-edges)

---

## Concepts

**One experiment file describes one algorithm and one study.** There is no inheritance,
no defaults file, no groups. Everything a launch needs is in the one YAML you pass on
the command line, and the file is archived verbatim alongside the results.

**One trial is one training run.** The environment and budget are fixed for the study.
Optuna proposes a complete algorithm parameter dictionary; that dictionary is handed to
your script unchanged alongside the environment and budget; the script trains once and
reports metrics. A Batch job may carry several trials, but a trial is never split across
jobs. Within the algorithm parameters there is no separate notion of "fixed" versus
"searched" — a parameter pinned to one value is just a distribution with one option.

**The image is the source of truth for what an algorithm accepts.** Each image carries
a catalog declaring its entry points, their parameter spaces, and the metrics they report.
The experiment file may narrow those parameters but may not invent new ones. The immutable
image digest identifies the code that runs.

**Rounds are how the optimiser learns.** A study of `rounds × trials_per_round` trials
runs in `rounds` waves. Each wave's results are read back before the next wave is
sampled, which is what lets TPE improve. Within a wave, trials are distributed over
`parallel_jobs` Batch jobs and the trials sharing a job run one after another.

**Failure stops everything.** If any job or any training process exits abnormally,
`trainerctl` terminates the surviving jobs and exits non-zero. There is no retry, no
resume, and no command to continue a launch. Fix the cause and run the study again.
A diverging run is not a failure: it produces a bad score, which the optimiser uses.

---

## Part 1 — Running an experiment

### Prerequisites

Run from the control plane directory, which is where the examples and the project
environment live:

```bash
cd rtrrl/infra/control-plane
uv run trainerctl --help
```

You need, for a Batch launch:

- AWS credentials for account `007122174918`. The region is fixed at `eu-north-1` in
  code and is not read from your environment.
- A reachable Aim server. Runs connect to it directly over the network while training,
  so the address in your experiment file must be one a Batch worker can resolve — the
  control plane's private IP, not `127.0.0.1`. Preflight rejects loopback addresses.
- An image already pushed to ECR with a registered job definition. See
  [Part 3](#part-3--operations).

### Quick start

Check the experiment against the image without spending anything:

```bash
uv run trainerctl validate examples/experiment-acceptance.yaml --backend batch
```

This resolves the image, reads its catalog, merges your space over it, and verifies the
queue, job definition, S3 bucket and Aim endpoint. It never submits a job. On success it
prints the resolved space:

```
resolved search space:
  learning_rate: {"type":"float","low":0.0001,"high":0.001,"log":true}
  seed: 0
```

Then run the study:

```bash
uv run trainerctl run examples/experiment-acceptance.yaml --backend batch
```

Progress goes to stderr, the final report to stdout, so `> report.json` gives you a clean
machine-readable result.

### The experiment file

Every field below is required unless marked otherwise, and unknown keys are rejected.
A complete file:

```yaml
experiment: infra-acceptance          # Aim experiment name; an archival label
name: brax-ppo-smoke                  # study name
description: Infrastructure-owned CPU acceptance sweep

image: 007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@sha256:d84ccca3d066ed070bd39840aebb0b04dc23d97dcaac544d2fcbca28d73dd9d9
entry: brax_ppo_acceptance            # an entry declared by the image's catalog
storage: s3://rtrrl-artifacts-007122174918/trainer

environment:
  id: brax::hopper
  backend: spring
  num_envs: 1
  observed: [0, 1, 2, 3, 4]          # optional; omit for full observability

budget:
  total_steps: 128
  epoch_steps: 128
  eval_steps: 100

compute:
  instance_type: c7a.medium
  timeout_minutes: 60

hpo:
  sampler: tpe
  rounds: 2
  trials_per_round: 2
  parallel_jobs: 1

space:
  seed: [0]
  learning_rate: {type: float, low: 1.0e-4, high: 1.0e-3, log: true}

score:
  metric: episode_return
  window_steps: [0, 128]
  reduce: mean
  direction: maximize
  non_finite: worst

logging:
  aim: aim://172.31.62.192:53801
  every_steps: 1
  rerun_every_episodes: 1
```

**Identity.** `experiment` is the Aim experiment the runs are filed under; `name` is the
study. Repeated launches of the same file are told apart by a launch id, a UTC timestamp
generated at start, so nothing is overwritten.

**`image`** must be an ECR reference in this account. A digest (`@sha256:...`) is strongly
preferred and is what the shipped examples use: a tag can be repointed between the moment
preflight resolves it and the moment a job pulls it.

**`environment`.**

| Field | Meaning |
| --- | --- |
| `id` | Environment identifier passed to the entry, such as `brax::hopper` |
| `backend` | Environment physics backend, such as `spring` |
| `num_envs` | Number of parallel environment streams; must be positive |
| `observed` | Optional observation dimension indices visible to the agent |

`observed` selects dimensions rather than zeroing the rest, so the observation space and
the network's input layer genuinely shrink. Omit it for full observability. The list may
not be empty, repeat an index, or contain a negative index.

**`budget`.**

| Field | Meaning |
| --- | --- |
| `total_steps` | Total training budget; must be positive |
| `epoch_steps` | Steps per epoch; must be positive and divide `total_steps` |
| `eval_steps` | Evaluation steps; may be zero |

`epoch_steps` must also contain a whole number of environment streams: it must be divisible
by `environment.num_envs`.

**`compute`.**

| Field | Meaning |
| --- | --- |
| `instance_type` | Selects the queue. One of `c7a.medium`, `c7a.large`, `c7a.xlarge`, `g6.xlarge` |
| `timeout_minutes` | Batch kills the job after this. It covers every trial packed into that job |

`timeout_minutes` must be at least 1, and it is the only thing bounding a wedged run:
the worker starts a run and waits for it, without judging how often it reports.
`instance_type` is only checked against the queue table during a Batch preflight, so a
typo survives `validate --catalog`.

**`hpo`.**

| Field | Values |
| --- | --- |
| `sampler` | `tpe`, `random`, or `grid` |
| `rounds` | Number of sequential waves |
| `trials_per_round` | Trials sampled per wave |
| `parallel_jobs` | Batch jobs per wave; must not exceed `trials_per_round` |

`parallel_jobs: 1` packs the whole wave into one job that runs its trials serially — the
right choice for short runs, since it pays for one instance startup instead of several.
`grid` requires every parameter to be a fixed list; preflight names the offending ranges
if it is not.

**`logging`.**

| Field | Required | Meaning |
| --- | --- | --- |
| `aim` | yes | `aim://host:port` of the Aim server |
| `every_steps` | yes | Passed to your script; the Aim sink also skips reports closer together than this |
| `rerun_every_episodes` | no | Omit to disable Rerun recording entirely |

### The search space

The image's catalog declares the full set of parameters an entry accepts. Your `space`
overrides entries in that set key by key. Naming a key the entry does not declare is an
error that lists what it does declare; omitting a key means the catalog's own declaration
is used.

Three ways to write a parameter:

```yaml
seed: [1]                                              # pinned: one option
backbone: [lru, ctrnn]                                 # categorical: several options
entropy_rate: {type: float, low: 1.0e-8, high: 1.0, log: true}
hidden_dim:    {type: int,   low: 1,      high: 512, step: 1}
```

Pinning is not a special case: a one-element list is a categorical distribution with one
option, so the trial's recorded parameters always contain the complete configuration
rather than only the parts that varied.

The environment and budget are not algorithm parameters. Catalog entries must not declare
them, and an experiment may not put their reserved names under `space`: `environment`,
`env_mode`, `env_backend`, `observed`, `num_envs`, `total_steps`, `epoch_steps`, or
`eval_steps`. Their values belong in the top-level `environment` and `budget` sections.
The `step` value your script passes to `report()` must use the same unit as
`budget.total_steps`, so comparing a score window against the budget is pure arithmetic.

### Scoring

The score is what the optimiser sees, and it is computed by the worker from the metrics
your script reported, not by your script.

| Field | Values |
| --- | --- |
| `metric` | Must be one of the metrics the entry declares |
| `window_steps` | `[low, high]`, inclusive, in the same unit as `budget.total_steps` |
| `reduce` | `mean`, `median`, `min`, `max`, `last` |
| `direction` | `maximize` or `minimize` |
| `non_finite` | `worst`, or a number |

The window's upper bound may not exceed `budget.total_steps`; otherwise the run could never
fill it, and preflight says so with both numbers.

`non_finite: worst` substitutes an ordered-worst value for NaN or infinity, which keeps a
diverged trial in the study as a strong negative signal rather than killing the launch.
Giving a number instead pins that substitution yourself. A run that reports nothing inside
the window is a failure, not a bad score.

### Command reference

Two subcommands. There is deliberately no `status`, `resume`, or `history`.

**`trainerctl validate EXPERIMENT`** — exactly one of:

| Flag | Effect |
| --- | --- |
| `--catalog PATH` | Offline check against a catalog JSON. Touches no AWS |
| `--backend batch` | Full check: resolves the image from ECR and verifies queue, job definition, bucket and Aim reachability |
| `--queues run\|dev` | Which queue tier to verify against. Default `run` |

**`trainerctl run EXPERIMENT --backend local|batch`**

| Flag | Default | Effect |
| --- | --- | --- |
| `--backend` | none; required | `batch` runs on AWS; `local` runs the worker as a subprocess here |
| `--catalog PATH` | — | Required for `local`, ignored for `batch` |
| `--archive-dir PATH` | `archive` | Local archive root |
| `--jobs-dir PATH` | `jobs` | Worker workspace; `local` only |
| `--queues run\|dev` | `run` | `batch` only |

`dev` queues are for infrastructure development. Delivered runs use `run` queues, and
choosing `dev` prints a warning.

Exit codes:

| Code | Meaning |
| --- | --- |
| 0 | Validation passed, or every trial completed |
| 1 | Preflight rejected the experiment, or the launch failed |
| 2 | Usage error |
| 130 | Interrupted with Ctrl-C during `run --backend batch` |

### Output and where things land

While running, one line per finished trial goes to stderr:

```
trial 0: {'learning_rate': 0.00013, 'seed': 0, ...} -> 21.0
best trial 2 scored 25.0
```

On success, stdout carries the report:

```json
{
  "best": {"job_id": "...", "log_stream": "...", "params": {...}, "trial": 2, "value": 25.0},
  "elapsed_seconds": 426.3,
  "failure": null,
  "launch_id": "20260726-003613",
  "status": "succeeded",
  "trials": [{"job_id": "...", "params": {...}, "trial": 0, "value": 21.0}]
}
```

The same report is written to the archive and to S3 **even when the launch fails**, with
`status: "failed"` and `failure` naming the exception — so a paid run always leaves a
post-mortem.

Locally, under `--archive-dir`:

```
archive/{experiment}/{name}/{launch_id}/
    experiment.yaml   # byte-for-byte copy of what you passed
    space.json        # the space after merging with the catalog
    launch.json       # image digest, environment, budget, queue, job definition, hpo settings
    study.db          # the Optuna study
    report.json
```

In S3, under `{storage}/{experiment}/{name}/{launch_id}/`: the same documents except
`study.db`, which stays on your machine, plus the per-job and per-trial artefacts:

```
rounds/round-000/job-0.json        # manifest: which trial configs this job must run
trials/t0/config.json              # the run configuration handed to the script
trials/t0/score.json               # the score the worker computed
trials/t0/episodes/episode-000001.rrd
```

In Aim, each trial appears as a run named `{name}-{launch_id}-t{trial}` under the
experiment, carrying the launch id, trial number, entry, image digest and the full
algorithm parameter dictionary.

### When something fails

**Preflight rejected it.** Nothing was submitted and nothing was spent. The message names
the field and, where a fix exists, the fix. Common ones:

| Message | Cause |
| --- | --- |
| `image does not declare entry '...'` | `entry` does not match the image's catalog |
| `experiment declares parameters the entry does not accept: ...` | A `space` key the catalog does not have |
| `space names {keys}, which belong to the environment and budget sections and are not searched` | Reserved environment or budget keys appear under `space` |
| `epoch_steps {N} is not {M} streams' worth` | `budget.epoch_steps` is not divisible by `environment.num_envs` |
| `score window upper bound {N} exceeds the budget's total_steps ({M})` | The window cannot be filled |
| `entry ... does not report metric '...'` | `score.metric` is not among the entry's declared metrics |
| `job definition '...' is not registered` | The image was never deployed; see Part 3 |
| `logging aim '...' is a loopback address` | Batch workers would resolve it to themselves |
| `instance_type '...' has no queue` | Not one of the four supported types |

**A job failed.** The failing job's name, the reason, and the last 200 lines of its
CloudWatch log are printed to stderr, surviving jobs in that round are terminated, and the
command exits 1. The report in S3 records how far the study got.

**A run hung.** Nothing intervenes until `timeout_minutes` elapses, at which point Batch
kills the job and the launch fails. Raise the timeout for a run that legitimately needs
longer; lower it to cap what a wedged run can cost.

---

## Part 2 — Adding an algorithm

An algorithm joins the facility by depending on the SDK, declaring a catalog entry, and
being baked into an image. The reference implementation is
`rtrrl/infra/mock-trainer/`, which is small enough to read in one sitting.

### What your script must do

The SDK exports nothing from its top level; import from the submodule.

```python
from training_sdk.reporter import Reporter

with Reporter.from_env() as reporter:
    params = reporter.config.params        # this trial's complete configuration
    scratch = reporter.scratch             # a writable directory, cleaned up for you
    ...
    reporter.report(step, {"episode_return": value})
```

`Reporter.from_env()` reads `TRAINER_RUN_CONFIG` and `TRAINER_SCRATCH`, both injected by
the worker; your script never sets them. It wires up the sinks automatically:

- **metrics file** — always. Every `report()` is appended to `metrics.jsonl` in the
  scratch directory. This is what the score is computed from.
- **Aim** — always, over the network to the configured server.
- **Rerun** — only when `rerun_every_episodes` is set.

To record a trajectory, hand over a complete episode:

```python
from training_sdk.episode import Episode

reporter.log_episode(Episode(
    number=n,                       # 1..999999
    phase="eval",
    start_env_steps=start, end_env_steps=end,
    observations=obs,               # exactly one more than actions
    actions=acts, rewards=rews,
    terminals=terms, truncations=truncs,
))
```

The episode must be complete — the last transition terminal or truncated — and the arrays
length-consistent. `Episode` checks all of this and raises rather than recording something
misleading.

Two constraints matter:

**Never log inside a JIT kernel.** Cross to the host first. The reference implementation
blocks on the device, converts with `np.asarray(jax.device_get(value))`, and only then
appends to a Python list.

**Sinks are not isolated from each other's failures.** If Aim is unreachable, the run
crashes and the launch stops. The SDK does not buffer, retry, or degrade. This is
deliberate: silently losing half a study's telemetry is worse than stopping.

### Declaring the entry

A catalog entry states how to start the script, what it reports, and what it accepts:

```python
EntryDescriptor(
    command=["python", "-m", "your_algorithm"],
    metrics=["episode_return", "episode_length"],
    space={
        "seed": {"type": "int", "low": 1, "high": 1_000},
        "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-2},
        "batch_size": {"type": "int", "low": 16, "high": 256, "step": 16},
    },
)
```

Copy `rtrrl/infra/mock-trainer/scripts/build_catalog.py`. It writes `catalog.json` and,
with `--print-label`, prints the gzipped base64 form used as an image label.

### Baking the image

Follow `rtrrl/infra/mock-trainer/docker/Dockerfile.cpu`. The parts that are not optional:

```dockerfile
ARG TRAINER_CATALOG_V2
RUN test -n "${TRAINER_CATALOG_V2}"
LABEL org.rtrrl.trainer.catalog.v2="${TRAINER_CATALOG_V2}"
COPY your/catalog.json /opt/trainer/catalog.json
CMD ["python", "-m", "training_sdk.worker"]
```

The catalog appears twice on purpose: the label lets the control plane read it without
running the container, and the file lets the worker look up the command. The CPU and GPU
Dockerfiles should differ only in acceleration.

---

## Part 3 — Operations

### Queues

Eight queues, one pair per instance type, created already and described in
`trainer_infra/queues.py`. Region `eu-north-1`, account `007122174918`.

| `instance_type` | Profile | Run queue | Dev queue | vCPU / job | Memory | GPU |
| --- | --- | --- | --- | --- | --- | --- |
| `c7a.medium` | `c7am` | `run-cpu-c7am-queue` | `dev-cpu-c7am-queue` | 1 | 1600 MiB | — |
| `c7a.large` | `c7al` | `run-cpu-c7al-queue` | `dev-cpu-c7al-queue` | 2 | 3200 MiB | — |
| `c7a.xlarge` | `c7ax` | `run-cpu-c7ax-queue` | `dev-cpu-c7ax-queue` | 4 | 7168 MiB | — |
| `g6.xlarge` | `g6x` | `run-gpu-queue` | `dev-gpu-queue` | 4 | 12000 MiB | 1 |

### Building and pushing images

Images are built by GitHub Actions, never on the development machine. A push touching the
SDK, the algorithm or the control plane builds and verifies the image without publishing
it; publishing to ECR is a separate manual dispatch:

```bash
gh workflow run build-infra-acceptance-image.yml \
  -f push=true -f confirm_account=007122174918
```

The verification step is worth knowing about, because it has caught real breakage: it
checks the label decodes to exactly the committed `catalog.json`, that the command is
`python -m training_sdk.worker`, that running without a manifest fails with a clear
message, and that the GPU image really has a CUDA plugin while the CPU image really
reports a CPU backend.

### Registering job definitions

A job definition binds a queue profile to one immutable image digest, so its name contains
the digest. Registration is a separate, explicit step; the script is dry-run by default:

```bash
uv run scripts/deploy_facility.py \
  --cpu-digest 007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@sha256:<cpu> \
  --gpu-digest 007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@sha256:<gpu>

# add --register --confirm-account 007122174918 to actually create them
```

It refuses tags, refuses identical CPU and GPU digests, and verifies the credentials
belong to the expected account before changing anything. It also ensures the
`/trainer/jobs` log group exists with 30-day retention. Jobs are registered with
`attempts: 1` — a retry would only pay to reproduce the same failure.

### Tests

Tests never run on the development machine, which is a micro instance. `tests.yml` runs
ruff and pytest for `training-sdk`, `rtrrl/infra/control-plane` and
`rtrrl/infra/mock-trainer` on every push.

---

## Sharp edges

Behaviours that are intentional but surprising, gathered in one place:

- `run --backend batch` silently ignores `--catalog` and `--jobs-dir`;
  `run --backend local` silently ignores `--queues`.
- `validate --catalog` and `run --backend local` do not validate `compute.instance_type`,
  because the queue table is only consulted during a Batch preflight.
- `logging.every_steps` is not checked for positivity, and an empty `space` is valid: it
  uses the catalog's complete algorithm parameter space unchanged.
- Overriding a catalog parameter only checks the key's name, not that your values fall
  within the range the catalog declared.
- `run --backend local` does not model Ctrl-C: the report is still written and children
  are killed, but the exit code is whatever Python produces.
- Older design documents mention `trainer_sdk.worker`. The module is `training_sdk.worker`.
