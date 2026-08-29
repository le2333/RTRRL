# trainerctl Manual

`trainerctl` runs one hyperparameter study for one algorithm, from your terminal, in the
foreground, in a single command. It samples parameters with Optuna, writes a run
configuration per trial, hands a round to an executor — AWS Batch, or a local worker
subprocess — waits, scores what came back, and starts the next round with what it
learned. When it exits, the study is over.

This manual covers three audiences, in order: people running experiments, people adding
an algorithm to the facility, and people operating the AWS side. Read the first part;
read the others when you need them.

---

## Contents

- [Concepts](#concepts)
- [Part 1 — Running an experiment](#part-1--running-an-experiment)
  - [Prerequisites](#prerequisites)
  - [The experiment file](#the-experiment-file)
  - [Metric names and what they were reduced over](#metric-names-and-what-they-were-reduced-over)
  - [The search space](#the-search-space)
  - [One value at several parameters](#one-value-at-several-parameters)
  - [Scoring](#scoring)
  - [Reporting a result](#reporting-a-result)
  - [Command reference](#command-reference)
  - [Output and where things land](#output-and-where-things-land)
  - [When something fails](#when-something-fails)
- [Part 2 — Adding an algorithm](#part-2--adding-an-algorithm)
- [Part 3 — Operations](#part-3--operations)
- [Sharp edges](#sharp-edges)

---

## Concepts

**One experiment file describes one algorithm and one study.** There is no inheritance,
no defaults file, no groups. Everything a launch needs is in the one YAML you pass on the
command line.

**One trial is one configuration, and one run per seed it is measured on.** The task, the
budget and the list of seeds are fixed for the study. Optuna proposes a complete algorithm
parameter dictionary; that dictionary reaches the entry unchanged alongside the run's
other sections; the entry trains once and reports metrics. A Batch job may carry several
runs, but a run is never split across jobs.
Within the algorithm parameters there is no separate notion of "fixed" versus "searched" —
a parameter pinned to one value is a distribution with one option.

**The image is the source of truth for what an algorithm accepts.** Each image carries a
catalog declaring its entries, their parameter spaces, and the metrics they report. The
experiment file may narrow those parameters but may not invent new ones. The immutable
image digest identifies the code that runs.

**The two sides share a JSON format, not a package.** Infra writes run configurations and
manifests; the Worker and the entry inside the image read them. Neither imports the
other. `docs/contract.md` is that format, and `contract: 11` in a document is what lets a
mismatch be refused rather than misread.

**Rounds are how the optimiser learns.** A study of `rounds × trials_per_round` trials
runs in `rounds` waves. Each wave's results are read back before the next wave is sampled,
which is what lets TPE improve. Within a wave, trials are distributed over
`hpo.parallel_jobs` Batch jobs and the trials sharing a job run one after another.

**Failure stops everything.** If any job or any training process exits abnormally,
`trainerctl` terminates the surviving jobs of that round and exits non-zero. There is no
retry and no resume. A diverging run is not a failure: it produces a bad score, which the
optimiser uses. A controller killed after its workers finished is the one recoverable
case, and `trainerctl settle` is for it.

---

## Part 1 — Running an experiment

### Prerequisites

Run from `infra/`, which is where the control plane's project environment lives:

```bash
cd infra
uv run trainerctl run --help
```

You need, for a Batch launch:

- AWS credentials for account `007122174918`. The region is fixed at `eu-north-1` in code
  and is not read from your environment.
- A reachable Aim server. Runs connect to it directly over the network while training, so
  the address in your experiment file must be one a Batch worker can resolve — the control
  plane's private IP, not `127.0.0.1`.
- An image pushed to ECR, with a job definition registered for that digest. See
  [Part 3](#part-3--operations).
- The image's `catalog.json`, which you pass with `--catalog`.

### The experiment file

Every field below is required unless marked otherwise. A complete file, modelled on
`experiments/streamac template.yaml`, which is the one to copy:

```yaml
experiment: streamac                  # Aim experiment the runs are filed under
name: streamac-hopper                 # study name
description: StreamAC on a masked Hopper       # optional; archival only

image: 007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@sha256:...
entry: stream_ac                      # an entry declared by the image's catalog
storage: s3://rtrrl-artifacts-007122174918/trainer

environment:
  id: brax::hopper
  backend: spring                     # null where the namespace has one implementation
  seeds: [0]                          # every configuration runs once per seed; not searched
  episode_length: 1000
  observed: [0, 2, 4]                 # optional; omit for full observability

training:
  num_envs: 16
  total_steps: 2000000
  chunk_steps: 10000

evaluation:
  every_steps: 10000
  episodes: 5                         # exactly this many complete episodes per checkpoint
  chunk_steps: 16000                  # memory bound on one evaluation call
  seed: 1000                          # the evaluation's own key stream

compute:
  instance_type: c7a.large
  timeout_minutes: 240

hpo:
  rounds: 2
  trials_per_round: 4
  startup_trials: 4
  parallel_jobs: 2
  seed: 7

space:                                # overrides the catalog's tree, same shape
  backbone:
    kind: [rtu]
    rtu:
      hidden_dim: [32]

score:
  metric: eval/episode/return_per_step
  window_steps: [0, 2000000]
  reduce: auc
  episodes_per_checkpoint: 5          # optional; refuses a checkpoint that reported fewer
  non_finite: worst
  direction: maximize

logging:
  aim:
    url: aim://172.31.62.192:53801
    training:
      window: { every_steps: 100000 }
  rerun:
    log_every_steps: 100000
```

**Identity.** `experiment` is the Aim experiment the runs are filed under; `name` is the
study. Repeated launches of the same file are told apart by a launch id, so nothing is
overwritten. A generated one is the UTC second the launch started followed by eight random
characters — `20260818-151037-3f9c1ab2` — because the second alone is not enough: two
controllers started together read the same one, and before this they took the same control
prefix and wrote over each other's manifests. `run` announces the id it took on stderr —
`launch 20260818-151037-3f9c1ab2` — before it submits anything, because a generated one
cannot be worked out from the start time and `settle` needs it from a launch that died.
`--launch-id` pins it when you need the artifacts of a launch to land where an earlier
one's did.

**`image`** must be pinned to a digest (`@sha256:...`). A tag is refused outright: the
catalog binds a search space to the image that declared it, and a floating tag would let
that space change under a study that has already recorded trials against the old one.

**`environment`.**

| Field | Meaning |
| --- | --- |
| `id` | Environment identifier passed to the entry, such as `brax::hopper` |
| `backend` | Physics backend, such as `spring`; `null` where the namespace has only one |
| `seeds` | The seeds every configuration is run on; distinct, non-negative, at least one |
| `episode_length` | The environment's own episode cap; must be positive |
| `observed` | Optional observation dimension indices visible to the agent |

`observed` selects dimensions rather than zeroing the rest, so the observation space and
the network's input layer genuinely shrink. Omit it for full observability. The list may
not be empty, repeat an index, or contain a negative index. `backend: null` and
`observed: null` both say "not applicable", which is not the same as omitting the field.

`seeds` is a list, and it is **not searched**. A seed is not a hyperparameter: two runs
that differ only in it are the same configuration measured twice, so letting the sampler
draw one would spend the study's budget modelling noise and then report the luckiest draw
as the best setting. Every configuration is instead run once per listed seed, and the
optimiser is told the mean; the per-seed scores are printed beside it under `seed_values`
and are what a result table reports.

A tuning launch lists one seed, which leaves that mean the one run's score unchanged. The
formal launch that follows lists ten fresh seeds on a discrete task or five on Brax — see
[Reporting a result](#reporting-a-result).

**`training`.** Three schedules that were one field before contract 9, and none of them
derives from another.

| Field | Meaning |
| --- | --- |
| `num_envs` | Number of parallel training streams; must be positive |
| `total_steps` | Total training budget; must be positive |
| `chunk_steps` | How much one training call may hold — the run's memory cost |

`chunk_steps` is deliberately unrelated to `episode_length`: how long an episode runs is
decided while the run is going and grows as the policy improves, so sizing a buffer from
it would tie a memory budget to a number that has nothing to do with memory.

**`evaluation`.**

| Field | Meaning |
| --- | --- |
| `every_steps` | Environment steps between measurements of the policy |
| `episodes` | How many complete episodes each measurement is scored on, exactly; zero skips evaluation entirely |
| `chunk_steps` | How much of one measurement a single call may hold |
| `seed` | The evaluation's own key stream; must not be negative |

`episodes` is a count, not a budget. How long those episodes run is what the policy
decides, so asking for steps instead lets the number of episodes vary with the task and
with training — which is what a protocol may not let vary. Worse, it varies in a
direction: a step budget truncates whichever episode was still running, which is the
longest one, so the episodes that survive are systematically the short ones. On a task
where the return is the length, that is a thumb on the scale of the very quantity being
measured.

The runtime keeps advancing the rollout until the count is reached, and refuses the run
if the episodes are not ending. An episode a stream ran past the count is not scored
either: one extra changes the number as much as one missing.

With more than one stream the count is split by naming the slots before the rollout
runs — stream `i`'s `j`-th episode fills slot `j * num_envs + i`, and the scored episodes
are the ones whose slot is below `episodes`. Nothing has to divide anything: the lower
streams simply contribute one more, and which episodes count never depends on which
finished first.

`chunk_steps` is a memory bound of the same kind as `training.chunk_steps` and says
nothing about how much is run.

`seed` is separate from the environment's seeds so that evaluation cannot move the
training key stream: a run must learn the same thing whether it was measured every ten
thousand steps or every hundred thousand. Two methods that declare the same evaluation
seed are measured on paired evaluation episodes.

The entry refuses a document whose schedules do not divide: `chunk_steps`,
`evaluation.chunk_steps` and `evaluation.every_steps` must each contain a whole number of
environment steps (divisible by `num_envs`), and `total_steps` must be a whole number of
evaluation intervals. These are checked when the run starts, not by the control plane.

**`compute`.** Required for `--backend batch`; unread for `local`.

| Field | Meaning |
| --- | --- |
| `instance_type` | Selects the queue. One of `c7a.medium`, `c7a.large`, `c7a.xlarge`, `g6.xlarge` |
| `timeout_minutes` | Batch kills the job after this. It covers every trial packed into that job |

`timeout_minutes` is the only thing bounding a wedged run: the worker starts a run and
waits for it, without judging how often it reports.

**`hpo`.**

| Field | Meaning |
| --- | --- |
| `rounds` | Number of sequential waves |
| `trials_per_round` | Trials sampled per wave |
| `startup_trials` | Trials TPE samples at random before it starts modelling |
| `seed` | Sampler seed |
| `parallel_jobs` | Batch jobs per wave; must be between one and `trials_per_round`. Read only by `--backend batch`, which fails without it |

`parallel_jobs: 1` packs the whole wave into one job that runs its trials serially — the
right choice for short runs, since it pays for one instance startup instead of several.
The sampler is TPE; there is no `sampler` field to choose another.

**`logging`.** A block that is present is a destination that is on. There is no separate
boolean beside a value for it to disagree with.

| Field | Required | Meaning |
| --- | --- | --- |
| `aim.url` | yes | `aim://host:port` of the Aim server |
| `aim.training` | no | Which scopes of training reach Aim; omit for evaluation only |
| `rerun.log_every_steps` | no | Stride in environment steps between kept trajectories |

The metrics artifact is the run's complete record: every episode, always, not
configurable. `logging` decides what a *dashboard* receives, which is a different
question, and the next section is about how to ask for it.

### Metric names and what they were reduced over

A metric name is `{phase}/{scope}/{quantity}`. The middle segment is the scope the number
was reduced over, and it is the only place that says how the number was arrived at:

```
train/step/td_error       a reading at one moment
train/episode/return      one episode's statistic
train/window/return       every episode in a stretch, averaged
eval/episode/return       what a score reads
```

Evaluation always reaches Aim. Training reaches it only through the scopes you name under
`aim.training`, and each scope's interval is expressed in that scope's own unit:

| scope | interval | answers |
| --- | --- | --- |
| `step` | `every_steps` | what a reading looks like at a typical moment |
| `episode` | `every_episodes` | what a typical episode's statistic is |
| `window` | `every_steps` + optional `length_steps` | what every episode in a stretch averaged |

```yaml
logging:
  aim:
    url: aim://172.31.62.192:53801
    training:
      window: { every_steps: 100000 }   # length_steps defaults to every_steps
      step:   { every_steps: 10000 }    # any subset; all three are allowed
```

Why the unit matters: sampling an *episode* on a *step* schedule is biased, and it was the
default before contract 10. The episode a step mark lands in is the episode that spans it,
and a long episode spans more of the axis, so long episodes were over-represented. On a
masked Hopper that put the sampled mean episode length near twice the true one early in
training and near the true one at the end, so `return` and `trace_norm` came back inflated
at the start and not at the end — a compressed curve, in which a run that is learning
looks like a run that is not.

Each scope avoids that by selecting among objects of one size. `step` selects a step. Every
step is one step, and a mark names one stream's one transition. `episode` selects every Nth
episode, which is uniform in episode space. `window` selects a stretch of the axis and
counts an episode in the window it *ends* in, which partitions the episodes between
windows. `window.length_steps` defaults to `every_steps`, which tiles the axis and uses
every episode; a shorter length samples stretches instead — still unbiased, since a stretch
is a fixed size — and keeps the accumulator alive for less of the run.

Inside a window a series is pooled over transitions, not over episodes: episodes differ in
length, so a mean of per-episode means is not the mean and a mean of per-episode variances
is not the variance. `return` and `length` are per-episode quantities to begin with, so the
window averages the episodes.

Omitting `aim.training` records evaluation only. Writing the block and naming no scope in
it is refused, rather than silently meaning nothing.

### The search space

The image's catalog declares the full set of parameters an entry accepts, as a tree. Your
`space` is a tree of the same shape that narrows it node by node; what comes out is a
third tree of that shape. It stays a tree because the tree *is* the conditional structure:
a parameter under a branch exists only for the trials that chose that branch, and a flat
table cannot say so.

Nothing but the leaf separates a group from a value. A component chosen among branches is
a parameter named `kind` living beside those branches, which is also why the same name at
two sites is two parameters and not a clash — `actor.optimizer` and `critic.optimizer` are
one declaration used twice, each in its own scope.

```yaml
space:
  backbone:
    kind: [rtu]                     # pinned: a one-option categorical
    rtu:
      hidden_dim: [16, 32, 64]      # categorical
      differentiation:
        kind: [tbptt]
  actor:
    optimizer:
      base:
        sgd:
          lr: {type: float, low: 1.0e-5, high: 1.0e-3, log: true}
```

Pinning is not a special case: a one-element list is a categorical distribution with one
option, so a trial's recorded parameters always contain the complete configuration rather
than only the parts that varied. Omitting a node means the catalog's own declaration is
used, so an empty `space` is valid and means "the catalog's space, unchanged".

Three rules the control plane enforces before any trial starts:

- a pin must name a parameter the image declares. One that does not is refused rather than
  ignored: it is a knob that turns nothing, and the run would start with the value its
  author believed they had set;
- an override must lie inside the `valid` domain the catalog declares for that parameter;
- a structural parameter — a `kind` whose value selects which sub-graph exists — must be
  pinned to a single option for the whole study. A structural sweep changes what other
  parameters even exist, and that is not a search the optimiser can model.

The task, the run shape and the evaluation shape are not algorithm parameters. They belong
in the top-level `environment`, `training` and `evaluation` sections and reach the entry
through the run configuration rather than through a sampled trial.

### One value at several parameters

Some comparisons hold a setting equal across the agent's blocks rather than searching it
in each. Giving `actor.optimizer.adam.b2`, `critic.optimizer.adam.b2` and
`torso.optimizer.adam.b2` the same list does not do that: they are three leaves, so the
sampler draws three numbers and the study searches three dimensions. The comparison being
run is then not the one the file describes.

A `bindings` section declares a variable once and names the paths it is written into:

```yaml
bindings:
  shared_beta2:
    domain: [0.9, 0.99, 0.995, 0.999, 0.9999]   # the same notation a space leaf uses
    paths:
      - actor.optimizer.adam.b2
      - critic.optimizer.adam.b2
      - torso.optimizer.adam.b2

space:
  actor:
    optimizer:
      kind: [adam]
      adam:
        lr: {type: float, low: 1.0e-5, high: 1.0e-2, log: true}   # still its own
  critic:
    optimizer:
      kind: [adam]
      adam:
        lr: {type: float, low: 1.0e-5, high: 1.0e-2, log: true}
  torso:
    optimizer:
      kind: [adam]
      adam:
        lr: {type: float, low: 1.0e-5, high: 1.0e-2, log: true}
```

The study searches one parameter, `shared_beta2`, and each run document carries an
ordinary number at each of the three paths. The three learning rates beside it are
untouched and are still drawn apart.

**A binding is configuration and nothing else.** The three blocks build three optimizers
with three states exactly as they did before; no moment, trace or bound statistic is
shared, and nothing in this section can make one be. Four rules given one setting are four
rules, which is what lets `output_iu` fan one value out to every intentional rule an agent
runs — the two heads and the torso's two branches — while each keeps its own `eta` and its
own state:

```yaml
bindings:
  shared_clip:
    domain: [10.0, 20.0, 50.0]
    paths:
      - actor.optimizer.iu.clip
      - critic.optimizer.iu.clip
      - torso.optimizer.output_iu.actor.clip
      - torso.optimizer.output_iu.critic.clip
```

A variable's name has no dots and a destination is a dotted path, which is how the two
namespaces stay apart — and why a variable cannot name another variable, so there is no
cycle to spell. A `domain` is written exactly the way a `space` leaf is, and what makes one
frozen is the domain rather than the notation: `[0.999]`, `{type: choice, values: [0.999]}`
and `{type: float, low: 0.999, high: 0.999}` are one value each, to a formal launch as much
as to the sampler. The rest is checked before a container starts:

| Refusal | Cause |
| --- | --- |
| `the image declares no parameter at [...]` | A destination the catalog does not have |
| `the shared domain is outside the valid domain at [...]` | Some destination will not take the value |
| `[...] are under branches this experiment does not select` | A destination under a `kind` this file pinned elsewhere |
| `[...] select which parameters exist` | A `kind` was bound; pin a structural choice under `space` |
| `more than one value is written into [...]` | Two variables name one path |
| `[...] are bound to a shared variable and also pinned under space` | One leaf with two authors |
| `shared variables [...] are already names in the image's parameter tree` | The name would collide with a parameter |
| `shared variables [...] contain a dot` | A name shaped like a destination |
| `binding '...' reaches 1 path(s)` | One destination is that parameter's own range |

Two complete files are checked in beside the others:
`experiments/rtrrl issue81 shared adam.yaml` is the first example as a launch, and
`experiments/rtrrl issue81 shared iu output.yaml` is the second, with the two arms it is a
short edit away from named in its header.

The binding is archived on the study alongside the seeds and the selection, because what
the one dimension stood for is not recoverable from what Optuna stores. Resuming a study —
`trainerctl settle` — reads the variable back and writes it out to its destinations again,
so the runs it scores are the runs that were submitted.

**A study cannot be restamped.** That record is what the trials already in it were drawn
under, so resuming with an edited file is refused rather than allowed to overwrite it:

```
study 'rtrrl-issue81-shared-adam' already records ['bindings'] differently; it describes
the trials it has already drawn, so resume the launch it belongs to or name a study of
your own
```

The same holds for the seeds a launch was measured on, its evaluation seed, its sampler
seed and its selection — everything the control plane archives.

Two things make that check total rather than nearly so. Every block is recorded even when
it is empty — `bindings: []` where nothing is shared, `selection: null` where nothing was
frozen — because a key the study never held is a key nothing can be compared against; and
the comparison starts from what the study holds rather than from what the file still says,
so a block *deleted* from a file is caught as well as one edited. Deleting `selection` from
a formal launch and resuming its study is refused for exactly that reason, rather than
quietly carrying on as a tuning launch in a study whose trials were drawn as a formal
one's.

A key the launch records and the study does not is written rather than refused: that is
what a study opened by an older build looks like, and it cannot be told apart from one that
recorded the key as absent. Omitting `bindings` otherwise leaves a file exactly as it
was.

### Scoring

The score is what the optimiser sees. It is computed by the control plane from the
`metrics.jsonl` the run uploaded — not by the training code, and not by the worker.

| Field | Values |
| --- | --- |
| `metric` | Must be one of the metrics the entry declares in the catalog |
| `window_steps` | `[low, high]`, inclusive, in environment steps |
| `reduce` | `mean`, `median`, `min`, `max`, `last`, `auc`, `last_checkpoints` |
| `checkpoints` | Required by `last_checkpoints`, accepted by nothing else: how many of the last measured steps to average |
| `episodes_per_checkpoint` | Optional, and only for `auc` and `last_checkpoints`: the episode count each checkpoint must have reported |
| `direction` | `maximize` or `minimize` |
| `non_finite` | `worst`, or a number |

Score an evaluation metric. `eval/episode/return` and `eval/episode/return_per_step` are
the two usual choices, and both are unbiased by construction: an evaluation is a
measurement the run asked for at a fixed step, not a sample of something that happened.

The first five reductions read the metric's rows. `auc` and `last_checkpoints` read the
**evaluation curve**: a checkpoint is one measured step and arrives as one row per
episode, so the rows of a step become that checkpoint's mean and the reduction runs over
those means. Reducing the rows directly would weigh each checkpoint by how many episodes
it happened to report, which is what the fixed episode count exists to stop varying.

`auc` is the area under that curve per environment step — a step-weighted mean, so it
reads on the scale of the returns it integrates and is comparable across budgets. Its
endpoints are the first and last checkpoint the window admitted, never the window's own
bounds, which nothing was measured at. Between them the trapezoid rule spans whatever
spacing the checkpoints have, so an interval missing its measurement is crossed by the
line between its neighbours rather than dropped. One checkpoint has no area and is an
error.

`episodes_per_checkpoint` turns the runtime's exactness into a claim the control plane
checks: a checkpoint that did not report that many values for the metric is refused
rather than averaged, because a checkpoint scored on nine episodes is not the quantity
the other runs report.

`non_finite: worst` substitutes an ordered-worst value for NaN or infinity, which keeps a
diverged trial in the study as a strong negative signal rather than killing the launch.
Giving a number instead pins that substitution yourself. A run that reports nothing inside
the window is a failure, not a bad score.

### Reporting a result

A study chooses a configuration. It does not measure one: the configuration it reports as
best was chosen partly by the luck of the seed it was tuned on, so its tuning score is
biased upward and is not a result. Measuring it is a second launch.

That launch is an ordinary experiment file with three differences, and declaring them is
what makes it one:

```yaml
environment:
  seeds: [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]  # fresh; ten discrete, five Brax

selection:
  study: rtrrl-repeatprevious          # where the configuration came from
  trial: 3                             # which trial was frozen
  tuning_seeds: [0]                    # what it was chosen on, and may not reuse

space:
  # every leaf a single value: the frozen best parameter dictionary
```

`trainerctl` refuses the launch if any of those claims fails, before any container starts:

| Refusal | Cause |
| --- | --- |
| `formal seeds [...] were already used to tune this configuration` | A listed seed appears in `selection.tuning_seeds` |
| `a formal launch runs the configuration it froze, but [...] still offer more than one value` | Something under `space`, or a `bindings` domain, is still being searched |
| `a formal launch cannot be scored on 'train/...'` | Training return is a diagnostic, never a formal score |
| `the selection block does not say [...]` | `selection` is present but incomplete |

Every run carries `identity.role` (`tuning` or `formal`) and `identity.seed` into its
`result.json` and onto its Aim run, so a result found on its own can still say what it is
allowed to be used for. The launch's seeds, evaluation seed and selection are archived on
the study as user attributes and printed in the report.

There is no `trainerctl formal` subcommand. Freezing the best dictionary is an edit to a
file you keep, which is what every other decision in this facility is.

### Command reference

Two subcommands. There is deliberately no `status`, `resume`, or `history`.

**`trainerctl run EXPERIMENT`**

| Flag | Default | Effect |
| --- | --- | --- |
| `--backend local\|batch` | none; required | `batch` submits to AWS; `local` runs the worker as a subprocess here |
| `--catalog PATH` | none; required | The image's `catalog.json` |
| `--database PATH` | none; required | The Optuna study database |
| `--launch-id ID` | generated | Names this launch's artifacts; a UTC timestamp and random suffix by default |
| `--exchange PATH` | — | Where round configs and manifests are written; required for `local` |
| `--workspace PATH` | — | Worker scratch root; required for `local` |
| `--queues run\|dev` | `run` | `batch` only |
| `--poll-seconds N` | `20` | How often a Batch round is polled |
| `--worker-command ...` | `python -m worker` | `local` only; must be the last option |

**`trainerctl settle EXPERIMENT --launch-id ID`** takes the same flags, with
`--launch-id` required. It submits nothing: for each trial the study still has open it
reads that trial's already-uploaded results, scores them, and tells Optuna. This is the
one recoverable failure — a controller killed between a worker's last upload and
`study.tell` leaves a trial RUNNING whose training is finished and paid for. A trial whose
work genuinely has not finished is reported as still running and left alone.

`dev` queues are for infrastructure development; delivered runs use `run` queues.

### Output and where things land

`run` prints the study to stdout as JSON: the launch id, every trial's number, state,
value and parameters, any `bindings` the file declared, and the best trial. `settle` prints
what it settled and what is
still running. The launch id is on stderr as well, printed before the first round rather
than after the last, so a run that never reaches its report has still said what it was.
Progress and worker output go to stderr, so `> report.json` gives a clean machine-readable
result.

Under `--backend local`, the exchange holds one directory per round:

```
{exchange}/round-000/trial-000000.json     # the run configuration handed to the entry
{exchange}/round-000/manifest.json         # which runs this worker must execute, in order
{exchange}/round-000/worker.log            # the worker's combined stdout and stderr
```

Under `--backend batch`, the same rounds are written to
`{storage}/{experiment}/{launch_id}/control/`, and beside them `control/launch.json` says
which process took that prefix. It is written with a conditional create, so a launch that
generated its own id and found the prefix already taken stops there rather than submitting
into someone else's.

Each run's own artifacts are written by the entry into a scratch directory and uploaded,
relative paths preserved, to `artifacts.root`, which is
`{storage}/{experiment}/{launch_id}/{run_id}`:

```
metrics.jsonl                  # the complete record: every episode, both phases
rerun/train-sample-000000010000.rrd
result.json                    # written beside the run's artifacts once the upload is done
```

The worker uploads only after the entry exits zero, and it keeps the local scratch
directory of a failed run for diagnosis rather than cleaning it up.

In Aim, each run appears named `{name}-{launch_id}-t{trial}-s{seed}` under the
experiment, carrying the launch id, trial number, entry, image digest and the full
algorithm parameter dictionary.

### When something fails

**The experiment file was rejected.** Nothing was submitted and nothing was spent. The
message names the field:

| Message | Cause |
| --- | --- |
| `the experiment file does not say [...]` | Required keys are missing; the list is exhaustive |
| `image '...' is not pinned to a digest; use name@sha256:...` | `image` names a tag |
| `the image catalog declares no entry '...'` | `entry` is not in the catalog |
| `entry '...' declares no score metric '...'` | `score.metric` is not one the entry declares |
| `the image declares no parameter named ...` | A `space` key the catalog does not have |
| `experiment range is outside the valid domain for ...` | An override leaves the declared domain |
| `structure parameters must be fixed for one experiment: ...` | A structural parameter is being searched |
| `the image declares no parameter at [...]` | A `bindings` destination the catalog does not have; see [One value at several parameters](#one-value-at-several-parameters) |
| `parallel_jobs must be between one and the number of trials` | `hpo.parallel_jobs` exceeds `trials_per_round` |

**The run was rejected inside the image.** The run configuration is validated again by the
entry, which is where the schedule arithmetic lives — `chunk_steps must contain whole
environment steps`, `total_steps must consist of whole evaluation intervals`, and the
`logging` shape, including `aim training names no scope`. These arrive as a failed job, not
as a preflight message, so a shape mistake costs one container start.

**The launch id was lost with the terminal.** `settle` needs it and a generated one is
not the start time. Every launch of an experiment is a directory under
`{storage}/{experiment}/`, and a Batch launch's `control/launch.json` says which host and
process took it and when — enough to tell your dead controller's launch from the others
without having kept its output.

**The control prefix was already taken.** `... already belongs to ...` names the launch
that holds it, and nothing was submitted. A generated launch id refuses to write into an
existing prefix; passing `--launch-id` is how you say you meant that one, which is what a
resumed launch does.

**A job failed.** The failing job's name, the reason, and the last 200 events of its
CloudWatch log are printed to stderr, surviving jobs in that round are terminated, and the
command exits non-zero.

**A run hung.** Nothing intervenes until `timeout_minutes` elapses, at which point Batch
kills the job and the launch fails. Raise the timeout for a run that legitimately needs
longer; lower it to cap what a wedged run can cost.

---

## Part 2 — Adding an algorithm

An algorithm joins the facility by living in `memo/memorax/algorithms/`, being exposed
through a module in `memo/entries/`, and being baked into the image.
`memo/entries/stream_ac.py` is a complete example, and it is 66 lines.

### What an entry module must export

The catalog scanner imports every module in `entries/` whose name does not start with an
underscore, and requires three names from each:

| Name | Meaning |
| --- | --- |
| `PARAMETERS` | The algorithm's complete parameter space, which the catalog describes |
| `METRICS` | Every metric name the run reports, which `score.metric` is checked against |
| `main` | The process entry point the worker starts |

A module missing any of them fails the build with a message naming what it lacks, and an
image with no entries at all is refused, since it could run nothing.

An image declares parameters; it does not declare which of them an experiment holds equal.
A binding is written by the experiment against the paths this tree produces — the same
dotted names a run document carries — so nothing here has to anticipate one, and adding a
block or renaming a branch changes what a binding may say without changing anything about
how it is declared. See
[One value at several parameters](#one-value-at-several-parameters).

```python
from memorax.algorithms.stream_ac import METRICS as METRICS
from memorax.algorithms.stream_ac import PARAMETERS as PARAMETERS
from memorax.algorithms.stream_ac import StreamAC

from ._observability import build_reporter, load_run
from ._schedule import trajectory_at_steps, trajectory_record


def main(argv: list[str] | None = None) -> int:
    del argv
    config, scratch = load_run()
    with build_reporter(config, scratch) as reporter:
        run(reporter, config)
    return 0
```

`load_run()` reads `TRAINER_RUN_CONFIG` and `TRAINER_SCRATCH`, both injected by the
worker; the entry never sets them. It validates the whole run document against
`entries/_contract.py`, so a document from an older contract fails here, naming the field
it is missing, rather than half-running.

An entry's job is composition: project the run document onto the algorithm's build
request, onto `RuntimeConfig`, and onto the reporter. It must not become the place where
the algorithm's graph is defined.

### Declaring metrics

Do not spell metric names by hand. `metric_names` builds them from the phase and the
per-transition series the algorithm reports, so the declared set and the reported set
cannot drift:

```python
from memorax.observability.metrics import metric_names

TRAINING_METRICS: tuple[str, ...] = taken(REPORTS, parts=PARTS)
METRICS: tuple[str, ...] = metric_names("train", TRAINING_METRICS) + metric_names(
    "eval"
)
```

`METRICS` declares what the *record* contains, which is one reduction per episode in both
phases — that is what a score reads. The dashboard scopes are a deployment choice, not an
algorithm one, and do not appear here. `check_names` refuses a name that is not
`{phase}/{scope}/{quantity}` with a known scope, so a misspelling and an invented scope
are both caught.

### Reporting

The runtime reports complete episodes; the algorithm does not call a logger. Every
transition-level quantity the algorithm wants recorded is declared in its
`ObservationSchema.series`, and the tracker assembles it into the `Episode` the reporter
reduces. Two constraints matter:

**Never log inside a JIT kernel.** Cross to the host first; the tracker does this once per
chunk rather than once per step.

**Sinks are not isolated from each other's failures.** If Aim is unreachable, the run
crashes and the launch stops. There is no buffering, retry, or degradation. This is
deliberate: silently losing half a study's telemetry is worse than stopping.

### Baking the image

The catalog is generated at image build time by scanning `entries/`:

```bash
python -m deployment.catalog --print-label
```

That writes `catalog.json` and prints the gzipped base64 form carried as an image label.
The catalog appears twice on purpose: the label lets the control plane read it without
running the container, and the file lets the worker look up an entry's command.
`.github/workflows/build-memo-image.yml` builds and verifies the image on every push to
`main` that touches `memo/`; pushing to ECR is a separate manual dispatch (see Part 3).

---

## Part 3 — Operations

### Queues and job definitions

Eight queues, one pair per instance type. Region `eu-north-1`, account `007122174918`.

| `instance_type` | Profile | Run queue | Dev queue |
| --- | --- | --- | --- |
| `c7a.medium` | `c7am` | `run-cpu-c7am-queue` | `dev-cpu-c7am-queue` |
| `c7a.large` | `c7al` | `run-cpu-c7al-queue` | `dev-cpu-c7al-queue` |
| `c7a.xlarge` | `c7ax` | `run-cpu-c7ax-queue` | `dev-cpu-c7ax-queue` |
| `g6.xlarge` | `g6x` | `run-gpu-queue` | `dev-gpu-queue` |

A job definition binds a queue profile to one immutable image digest, so its name contains
the digest: `trainer-{profile}-{digest without the sha256: prefix}`. `trainerctl` derives
that name and submits against it; it does not create it. Registering a job definition for a
newly pushed digest is a separate, deliberate step performed outside this repository.

### Building and pushing images

Images are built by GitHub Actions, never on the development machine. A push touching
`memo/` builds and verifies the image without publishing it; publishing to ECR is a manual
dispatch that requires naming the account:

```bash
gh workflow run build-memo-image.yml \
  -f push=true -f confirm_account=007122174918
```

The verification step is worth knowing about, because it has caught real breakage: it
checks that the image label decodes to exactly the committed catalog, that a worker
started without a manifest fails with a clear message, that the CPU image really reports a
CPU backend, and that the environments an experiment names actually build inside the
image.

### Tests

The development machine is a micro instance and cannot run the suites without running out
of memory, so CI is where they execute. `memo-ci.yml` runs the static checks and the memo
suites — fast, service, and the pinned external parity comparisons. `tests.yml` runs
`infra`. A `workflow_dispatch` tests the remote state, so dispatching on an unpushed
commit tests the previous one.

---

## Sharp edges

Behaviours that are intentional but surprising, gathered in one place:

- `run --backend batch` ignores `--exchange`, `--workspace` and `--worker-command`;
  `run --backend local` ignores `--queues` and never reads `compute`, so an
  `instance_type` typo survives a local run.
- The control plane checks that an override lies in the catalog's declared domain, but not
  that the run's schedules divide. That arithmetic is checked inside the image, so it costs
  a container start to find out.
- An empty `space` is valid: it uses the catalog's complete algorithm parameter space
  unchanged.
- A bound parameter disappears from a study's parameters: the trial records the variable
  that was drawn, not the paths it was written to. The run documents carry the paths, and
  the study's `bindings` attribute says which they were.
- `metrics.jsonl` is not configurable and never sampled. Turning Aim's training scopes
  down makes the dashboard cheaper; it does not make the run's record thinner, and the
  score reads the record.
- A window that the end of a run cuts short is still reported, at the close it was
  scheduled for rather than at the last episode that reached it.
- `settle` scores what is already in storage. It cannot tell a trial that never ran from
  one whose upload failed; both come back as still running.
