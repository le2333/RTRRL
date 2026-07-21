# Complete Facility Task 3 Report

## Status

Implemented Task 3 on baseline `f96df6a` and completed the Critical/Important
review follow-up: strict concrete protocol materialization, explicit success
and failure lifecycles, exact per-completion environment steps, the two
supported launchers, configurable Brax horizons, host-only
training/evaluation observability, and trace exposure.

## Delivered

- `FacilityInput.load()` requires `protocol_version: "1"` plus exactly the
  concrete control-plane objects (`environment`, `logging`, `parameters`,
  `training_budget`), rejects unknown/missing fields, and recursively validates
  finite JSON values.
- Control-plane materialization emits that protocol version and applies every
  `FieldDescriptor.path` as a safe nested path. Unsafe, duplicate, and
  scalar/object prefix conflicts fail closed. A descriptor-to-resolve-to-
  materialize-to-`FacilityInput` test covers nested environment options and
  nested parameters without depending on Task 4 descriptors.
- `memo_stream_ac` statically supports only `memory_chain`, `kmemory_chain`,
  and `mujoco_masked` with `agent_type=rtu_rtrl`.
- `memo_rtrrl` statically supports only Hopper with
  `rtrrl_topology=shared`.
- Both launchers expose only required `--config`, bootstrap the SDK once on the
  host, and reuse the existing memo config, environment, agent, and training
  APIs. Main-path tests execute each selected builder/trainer/bootstrap seam.
- `max_episode_steps` now reaches Brax `EpisodeWrapper` and
  `BraxGymnaxWrapper.default_params`; it is no longer hard-coded to 1000.
- Training budgets must divide exactly across `num_envs * num_epochs`, giving
  one fixed JIT scan length with no overshoot or uneven-epoch recompilation.
  A fake-state host bridge proves the final state reaches the exact budget.
- Every completed `RecordEpisodeStatistics` record produces its own mandatory
  episode summary at
  `epoch_start_state.step + (transition_index + 1) * num_envs`, including
  several environments and repeated episodes in one epoch.
- Complete fixed-shape evaluation traces are converted on the host to SDK
  `Episode` values with N+1/N lengths, separate terminal/truncation flags, and
  optional environment states. Evaluation does not consume training steps, so
  both episode step bounds equal the current training `state.step`.
  Nonpositive transition counts are explicitly rejected by direct conversion
  and skipped by cadence emission.
- `LegacyProgram.evaluate()` now returns its trace instead of discarding it.
- SDK and NumPy conversion remain outside JIT. Successful execution alone calls
  `finish`/`finalize`. `TrainingRun.fail()`/`abort()` writes only non-secret
  `sdk/failed` and exception-type metadata, never objective/finalized events,
  and best-effort closes Aim, Rerun, and spool resources. Bootstrap closes
  partially created resources. Failure is idempotent and cannot replace the
  original training exception.
- Masked MuJoCo `env_name`, `mode`, and `backend` values are validated while
  loading the concrete config.
- Added focused concrete YAML fixtures and launcher/observability tests.
- Updated the memo lock with the already-declared standalone training SDK
  Pydantic and PyYAML dependencies; `uv lock --check` passes.

## TDD Evidence

The review follow-up RED runs showed missing SDK failure APIs, leaked successful
finalization on exceptions, missing protocol version, flat descriptor paths,
epoch-end episode timestamps, evaluation ranges that consumed fictitious
training steps, unvalidated MuJoCo values, launchers without failure ownership,
and an Aim resource created before start configuration that could not be
closed. Each was observed failing before its implementation change.

## Verification

- Facility launcher/observability/logger targets: `52 passed`.
- Standalone training SDK suite: `124 passed`.
- Full fake-only control-plane suite: `376 passed`.
- Evaluation trace fixture tests without optional real Brax:
  `16 passed, 2 deselected`.
- JIT contract single-file suite: `10 passed`.
- Ruff on new/core changed files: passed.
- `uv lock --check`: passed.
- `git diff --check`: passed.
- IDE lint: no findings.

The complete evaluation-trace file was also attempted unfiltered. Its 16
fixture tests passed; the two real-Brax cases failed at import because the
optional `brax` package is not installed in this local memo environment. This
is an environment limitation, not an assertion failure.

## Intentionally Not Run

Per the no-OOM/no-real-infrastructure constraint, the following require parent
authorization and the dev Batch runner or an appropriately provisioned image:

- full `memo/tests/online_ac` suite;
- standard/meta/evaluation parity and golden/legacy characterization suites;
- full JAX/parity suite and complete facility launcher training runs;
- actual compiled JAX `state.step` budget proof for both launchers through the
  dev Batch heavy-test runner;
- the two real-Brax trace cases in an environment with the Brax extra;
- CPU/GPU Docker builds, image smoke tests, and GPU runtime checks;
- any real AWS, Batch, ECR, S3, Aim service, or paid job operation.

No catalog, Dockerfile, workflow, controller, or AWS resource code was changed.
