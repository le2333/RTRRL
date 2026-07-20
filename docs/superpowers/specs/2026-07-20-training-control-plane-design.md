# Training Control Plane Design

## Purpose

Define the local configuration, HPO, scheduling, worker, and result-collection
behavior. The control plane is an independent Python environment under
`rtrrl/infra`; it must not import JAX, Brax, training entry points, or the
project’s logger implementation.

## Experiment Configuration

```yaml
experiment:
  name: hopper-comparison
  description: Compare independent RTRRL configurations
  metadata:
    owner: research
    tags: [hopper, partial-observation]

defaults:
  image: 123456789.dkr.ecr.eu-north-1.amazonaws.com/rl:2026-07-20
  environment:
    env_name: hopper
    backend: spring
    observation_mode: P
    max_episode_steps: 1000
  training_budget:
    env_steps: 2000000
  logging:
    aim_every_env_steps: 10000
    rerun_every_episodes: 100
  resources:
    profile: gpu
  hpo:
    total_trials: 20
    configs_per_batch: 4
    parameter_policy: scan_unfixed
  execution:
    runs_per_job: 2
    max_infra_retries: 2
    max_algorithm_retries: 0
    retry_backoff_seconds: 30
    aim_result_timeout_seconds: 600
  parameters:
    seed:
      values: [7]

groups:
  shared-rtrrl:
    script: rtrrl
    metadata:
      algorithm: rtrrl
      variant: shared
    parameters:
      topology:
        values: [shared]
      hidden_size:
        values: [64, 128, 256]
      learning_rate:
        min: 1.0e-5
        max: 1.0e-2
        scale: log

  dual-rtrrl:
    script: rtrrl
    metadata:
      algorithm: rtrrl
      variant: dual
    overrides:
      training_budget:
        env_steps: 3000000
      logging:
        rerun_every_episodes: 50
    parameters:
      topology:
        values: [dual]
      hidden_size:
        values: [64, 128]
      learning_rate:
        min: 1.0e-5
        max: 1.0e-3
        scale: log
```

Every named `groups` entry is independent. Traversal order may determine
deterministic identity allocation, but neither script identity nor parameter
similarity may merge groups.

## Script-Bound Description

Each image contains an index and one description per script:

```text
/opt/trainer/scripts/index.yaml
/opt/trainer/scripts/rtrrl.yaml
```

Example field:

```yaml
fields:
  learning_rate:
    path: optimizer_params_td.learning_rate
    type: float
    default: 0.001
    searchable: true
    constraints:
      gt: 0
    default_search:
      min: 1.0e-5
      max: 1.0e-2
      scale: log
```

The description also declares a stable script name, argv launch command,
training-budget fields, HPO objective metric/direction/reduction, and required
SDK protocol version.

Descriptions are embedded in image metadata during build. The controller may
accept a friendly image tag, but resolves it once to a digest, retrieves the
description bound to that digest, and stores only the digest in resolved
experiment and run snapshots. Updating a tag cannot change a running
experiment.

The facility validates field existence, declared basic types, domain shape, and
explicit constraints. It assumes users do not deliberately provide values that
the script cannot consume beyond what the description declares.

## Parameter Policies

### `scan_unfixed`

This is the default. Every field with `searchable: true` enters HPO using its
experiment domain or `default_search`, unless the resolved experiment gives it
one value. A searchable field lacking both a finite default search domain and
an experiment domain is invalid.

### `explicit_scan`

Omitted fields resolve to `default` and remain fixed. Only experiment domains
with multiple discrete values or a continuous range enter HPO.

An experiment range replaces the complete default search domain. It may extend
beyond the default domain but must satisfy genuine declared constraints.
Continuous Optuna domains must always be finite. Fixed fields are materialized
directly and are never passed through `trial.suggest_*`.

## Identity

- Internal experiment ID: generated once at command start.
- Study identity: internal experiment ID plus group name.
- Run sequence: independent and one-based per group.
- Run name: `<group>-<four-digit-number>`.
- Trial number: Optuna’s study-local number.
- Execution attempt: independent retry counter for one logical run.

Resolved groups and concrete runs are immutable snapshots containing the image
digest, full environment, budget, logging policy, resource profile, metadata,
fixed values, sampled values, and launch command.

## Independent Group State Machines

Each group moves through:

```text
READY
→ ASKING
→ RUNS_READY
→ SUBMITTED
→ RUNNING
→ COLLECTING_AIM
→ TELLING
→ READY or COMPLETE
```

The controller event loop advances every ready group independently. It asks at
most `configs_per_batch` trials while respecting remaining `total_trials`.
Multiple outstanding suggestions are represented to the sampler so that one
batch does not repeatedly select the same candidate.

Finite discrete spaces stop with `SPACE_EXHAUSTED` once all unique combinations
have been allocated, even when the nominal budget is larger.

## Scheduling and Worker

Ready runs are partitioned by:

- image digest;
- fixed resource profile;
- `runs_per_job`.

The scheduler does not partition by group or script. A job bundle may therefore
contain different scripts and groups. Each bundle preserves deterministic run
order.

The facility-provided worker:

1. Downloads the bundle and concrete configs from S3.
2. Verifies their SHA-256 hashes.
3. For each run, injects the facility run context.
4. Starts the declared argv with `shell=False` in a fresh child process.
5. Waits for the child, records its outcome, and continues to the next item.
6. Uploads completion status and registered artifacts.

The worker contains no sampling, Optuna, Aim query, or Batch control logic.

## Result Collection and Retry

Training scripts write Aim through the SDK. After a child exits, the controller
waits for a finalized Aim run with the expected internal run ID and complete,
finite objective. It then calls `study.tell()` sequentially for the owning
group.

Default retry policy:

- Infrastructure failures: two additional attempts.
- Algorithm failures: zero additional attempts.
- Aim result wait: bounded by 600 seconds by default.

Overrides must be non-negative. Every retry preserves trial/run identity and
increments only the execution attempt. Exhausted algorithm failures consume the
trial. Persistent controller, permission, or infrastructure failure terminates
the command with a structured experiment/group/run report.

The first release deliberately has no `resume`, `next`, cross-experiment
history-import, or shared-space command.

## S3 Exchange

```text
experiments/<experiment-id>/
  groups/<group>/
    runs/<run-id>/
      input/config.yaml
      input/run.json
      status/attempt-N.json
      checkpoints/
      rerun/
      aim-buffer/
  jobs/<job-id>/bundle.json
```

Completion markers contain identity, attempt, exit code, start/end times, Aim
run identifier when available, and artifact keys. They do not duplicate
metrics. S3 is an exchange and artifact channel, not HPO history.

## Testing

- Configuration/default/override contract tests.
- Parameter-policy and fixed-field exclusion tests.
- Independent study and run-sequence tests for same-script groups.
- Finite-space exhaustion and mixed continuous/discrete sampling tests.
- Worker hash, ordering, subprocess isolation, failure-continuation tests.
- Fake Aim/Batch/S3 controller test proving one call completes `2+2+1` for
  multiple independent groups.
- Bounded infrastructure, algorithm, and Aim-timeout tests.
