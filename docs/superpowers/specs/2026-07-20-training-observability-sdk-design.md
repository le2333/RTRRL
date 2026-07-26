# Training Observability SDK Design

## Purpose

Provide a facility-owned Python SDK that training scripts call without knowing
Aim repository internals, Rerun file organization, S3 keys, or Batch metadata.
The SDK is installed in every training image and is stable across experiments.

## Boundary

The SDK handles:

- run context and structured identity;
- Aim hparams, metrics, episode summaries, and finalization;
- complete-episode Rerun recording;
- checkpoint and artifact registration;
- local buffering when Aim is unavailable.

It does not handle:

- algorithm updates, evaluation policy, or trajectory generation;
- HPO sampling or objective selection;
- AWS Batch submission or polling;
- direct S3 upload from algorithm code.

The worker injects context and uploads registered artifacts. The algorithm
produces metrics and complete episodes and calls the SDK.

## User API

The intended interface is:

```python
run = trainer_sdk.current_run()
run.log_metrics(env_steps, metrics)
run.log_episode_summary(
    env_steps=env_steps,
    episode_return=episode_return,
    episode_length=episode_length,
)
run.log_episode(complete_episode)
run.register_checkpoint(checkpoint_path)
run.finish(final_metrics)
```

`current_run()` reads a worker-provided context file or environment pointer. It
must fail clearly outside a configured facility run unless the script
explicitly requests a documented local-development mode.

## Run Context

The worker-provided context contains:

- exact user experiment name and internal experiment ID;
- group and script names;
- user metadata such as algorithm and variant labels;
- run name, run ID, run number, trial number, and seed;
- image digest and worker/SDK protocol versions;
- complete environment and training-budget snapshots;
- fixed, sampled, and final parameter maps;
- resource profile and artifact directory.

The run name is always `<group>-<four-digit-number>`.

## Aim Contract

Every run uses the exact user `experiment.name` as Aim’s experiment. All
different groups in that experiment therefore appear in one Aim experiment.

The SDK records filterable fields under structured hparams:

```text
hparams.identity.group
hparams.identity.script
hparams.identity.run_number
hparams.identity.trial_number
hparams.identity.seed
hparams.metadata.algorithm
hparams.metadata.variant
hparams.environment.*
hparams.training_budget.*
hparams.parameters.fixed.*
hparams.parameters.sampled.*
hparams.parameters.final.*
hparams.infrastructure.image_digest
hparams.infrastructure.resource_profile
```

The visible run name is for reading only; user workflows must not parse it to
recover fields.

Parameters are recorded once at start. Training episode summaries are mandatory
and cannot be disabled by experiment configuration. At minimum, every summary
contains:

- `train/episode_return`;
- `train/episode_length`;
- `train/env_steps`.

Other training and evaluation metrics are throttled according to
`aim_every_env_steps`. The SDK owns throttling; algorithms always report their
native environment step.

`finish()` writes the descriptor-declared objective and a finalized marker.
The control plane must not complete an Optuna trial until both are queryable and
finite.

## Rerun Contract

The algorithm submits an already complete episode. An episode contains:

- episode number;
- phase such as train or eval;
- start and end environment steps;
- observations;
- actions;
- rewards;
- terminal/truncation indicators;
- optional environment state suitable for visualization.

The SDK selects episodes according to `rerun_every_episodes`; it never writes a
partial episode. Each selected episode is a separate artifact:

```text
<experiment-name>/<run-name>/episode-<six-digit-number>.rrd
```

Rerun metadata includes experiment, group, script, run/trial/episode numbers,
phase, and start/end environment steps. Array conversion occurs at the SDK
boundary so algorithm kernels do not depend on Rerun types.

## Buffering

SDK events are appended to a local spool before an Aim send is attempted. A
temporary Aim outage must not terminate training.

At process exit:

- successfully sent events remain auditable by event ID;
- unsent events remain in the spool;
- the worker uploads the spool under the run’s `aim-buffer/` prefix;
- the local controller replays buffered events into Aim before objective
  collection;
- replay is idempotent by event ID.

S3 buffering is temporary transport. Aim remains the user-facing record and HPO
metric source.

## Existing Script Integration

The existing logger injection pattern remains valid. `DummyLogger`,
`AimLogger`, and `MultiLogger` gain episode-summary and complete-episode
methods while preserving existing `log`, `log_params`, `finalize`,
`log_video`, and summary behavior.

Training loops must not log inside JIT kernels. Scripts retain or generate
complete evaluation transitions outside the update kernel:

- RTRRL and LRU evaluation retain observation/action/reward/done sequences.
- PPO and SAC perform a post-training or evaluation-point rollout using the
  trained policy.

Algorithm update semantics are out of scope and must remain unchanged.

## Documentation

The SDK delivery includes:

- a five-minute quick start;
- API reference with argument and lifecycle semantics;
- a complete minimal script;
- examples for episode summaries, complete Rerun episodes, and checkpoints;
- local-development behavior;
- Aim queries using structured hparams;
- Rerun lookup by experiment, run, and episode;
- error and buffering behavior.

Documentation must be sufficient for a script author who has not read the
facility implementation.

## Testing

- Exact Aim experiment/run naming and nested hparams tests.
- Mandatory episode-summary and environment-step tests.
- Metric-throttling tests.
- Complete-only Rerun selection and artifact-path tests.
- Local spool, interrupted send, upload, replay, and replay-idempotence tests.
- Existing logger backward-compatibility tests.
- Script contract tests proving every registered script emits required records
  without placing host callbacks inside JIT code.
