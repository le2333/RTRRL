# Task 11 Report: Legacy Entry Point and Configuration Migration

## Status

The historical `rtrrl/rtrrl.py` is now a 72-line compatibility CLI. It keeps
`RTRRLParams`, `train_rtrrl`, `--config_path`, legacy YAML fields, and CLI
overrides, while Memorax owns normalization, component/program selection,
training, evaluation, historical metric translation, and logger lifecycle.

## Preserved Work Audit and TDD

Two interrupted agents left valid uncommitted implementation and tests but no
report, commit, terminal record, or recoverable behavior-level RED output. That
work was audited rather than discarded. The retained compatibility test file
was green locally before further implementation (7 tests, 44,748 KiB peak
RSS), so no earlier RED claim is made.

The audit found one remaining behavior gap: discovery used `rtrrl*.yml` in
`memo/config` and silently omitted
`memo/config/independent_rtrrl_hopper_maskP_lru.yml`.

- RED: the repository audit regression expected 697 runtime RTRRL YAMLs but
  received 696.
- GREEN: discovery now accepts both `.yml` and `.yaml`, and any filename
  containing `rtrrl`, in both runtime config directories. The full entrypoint
  test file then passed 7/7.

No production change made during this continuation preceded its failing test.

## Delegation and Old-Core Removal

- `rtrrl/rtrrl.py` contains CLI/path compatibility and delegation only.
- Parse/build imports stop at the lightweight Memorax compatibility and
  entrypoint modules. Package exports were made lazy so this path does not
  import JAX, Brax, Optax, Distrax, environments, or the AAAI oracle.
- `normalize_legacy_invocation()` maps old budget names and delegates to
  `normalize_legacy_config()`.
- `describe_legacy_build()` delegates effective recipe construction to
  `to_component_config()` without creating an environment.
- Training delegates dynamically to `experiments/rtrrl_hopper/run.py`;
  `train_legacy()` selects the Memorax shared or independent builder and calls
  the common `train_loop()`.
- Evaluation remains in that common program lifecycle. `run_legacy_experiment`
  delegates logger creation/finalization to `with_logger` while preserving
  project `"RTRRL"`, legacy run names, logger backend, and repository.
- Historical strict logging now calls the shared
  `historical_rtrrl_metrics()` translator.

Against base `87fbcf5`, the external script changed from 983 lines to 72
lines (965 deletions, 54 compatibility additions). Its former TD, recurrent,
trace, optimizer, train-step, evaluation, and logging mathematics are absent.
The AST contract rejects JAX/Optax/Distrax/oracle imports and the former core
definitions.

## Subprocess Parse/Build Evidence

Representative legacy and Memorax YAMLs are parsed in fresh subprocesses:

- `rtrrl/config/rtrrl_hop_533.yml`: 10,000,000 steps, 10,000 epochs,
  `memo_experimental`, Aim, old run name, TD LR `3e-5`, RNN LR `2e-6`, and
  RNN clip `1.0`.
- `memo/config/rtrrl_hopper_533.yml`: 1,000,000 steps, 20 epochs,
  `memo_experimental`, no logger, Memorax run name, and the same optimizer
  values.

Both report `environment_started: false` and `jax_imported: false`. These
actions resolve only the static AgentProgram recipe; no Brax environment,
JAX trace, or compilation occurs.

## Exact Historical Mock Epoch

`--compat-action mock-epoch` emits the frozen 16-key historical dictionary
exactly, including `steps=30`, `mean_reward=4.0`,
`mean_delta=0.8333333134651184`,
`total_td_loss=19.399999618530273`,
`v_targ=20.66666603088379`, both optimizer learning rates, and all three
historical norm paths. The subprocess compares the complete decoded dictionary,
not a subset or tolerance. A separate test replaces the shared translator and
proves the mock epoch goes through that production logging boundary.

## Repository YAML Discovery and Classification

The plan expected 686 runtime RTRRL YAMLs. Current discovery found 697:

- `rtrrl/config`: 684
- `memo/config`: 13
- delta from plan: +11

Classification:

- accepted: 697
- unsupported explicit branch: 0
- unknown fields: 0
- deprecated no-op: 0

All supported runtime files parsed without edits; no YAML file changed.
Synthetic fixtures additionally prove explicit CTRNN and no-RNN branches are
classified as unsupported, unknown fields separately, and deprecated
`save_model` as a no-op warning. HPO study specifications and the GitHub
workflow whose filenames contain `rtrrl` are not runtime CLI configuration
files and are intentionally outside these two config-directory counts.

## Resource Incident Handling

Before testing, the process table showed no stale pytest, JAX, RTRRL training,
or environment process. No complete RL environment ran locally.

The earlier server-exhaustion run left no durable test output, so no result is
attributed to it. Local work was limited to config/static/focused checks:

- entrypoint-only: 7 passed, 44,440 KiB peak RSS;
- config plus entrypoint: 79 passed, 465,544 KiB peak RSS;
- ruff, compileall, `git diff --check`, and IDE diagnostics passed.

JAX/mock/full relevant parity ran on authorized AWS Batch queue
`rtrrl-cpu2-queue`, job definition `rtrrl-cpu-job:14`, with an explicit
8,192 MiB/4-vCPU override on a `c7a.2xlarge` compute environment.

## Batch and Static Evidence

- `843beb65-2b6b-4f68-b829-63adca40946f`: infrastructure-only failure before
  tests because the image lacked `/usr/bin/time` (exit 127). The retry installed
  that utility.
- `acd414ac-8b29-40c5-9309-b3e3a2f5e557`: exercised the complete selected
  set and measured 4,112,708 KiB peak RSS. It found only the two already
  documented Task-10 StreamAC x64-disabled int64/int32 baseline mismatches.
- `7e4fdedd-e0d9-43b7-8889-5e644f6830c1`: after excluding exactly those two
  known StreamAC cases and restoring the passing RTRRL parameterization,
  produced 207 passed and five authorized directional finite-difference skips;
  peak RSS was 3,944,868 KiB. Ruff passed. Its tests were green, but the job
  exited on pyright environment discovery because the temporary venv was not
  at the repository-configured `.venv` path.
- `95a29ae6-119c-48c2-a00d-8ff5ecf27010`: final Batch static retry binds that
  temporary environment at `.venv`, but failed before checks while installing
  the editable package because `git` was absent from the image.
- `e6e70a93-0d1b-40b2-8c32-66733a56c16b`: installed the missing prerequisite
  and succeeded. Pyright reported 0 errors/0 warnings, ruff and compileall
  passed, and all seven entrypoint contracts passed.

## Concerns

The two full selected-suite failures are unchanged pre-existing StreamAC
counter dtype differences under JAX x64-disabled operation, previously
recorded in Task 10; Task 11 does not modify StreamAC. HPO study YAMLs are
configuration generators rather than direct legacy CLI inputs and therefore
are reported as explicitly out of runtime-audit scope.
