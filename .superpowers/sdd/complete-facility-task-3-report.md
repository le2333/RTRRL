# Complete Facility Task 3 Report

## Status

Implemented Task 3 on baseline `f96df6a`: strict memo facility inputs, the two
supported launchers, exact environment-interaction budgeting, configurable
Brax horizons, host-only training/evaluation observability, trace exposure, and
exception-transparent finalization.

## Delivered

- `FacilityInput.load()` accepts exactly the concrete control-plane top-level
  objects (`environment`, `logging`, `parameters`, `training_budget`), rejects
  unknown/missing fields, and recursively validates finite JSON values.
- `memo_stream_ac` statically supports only `memory_chain`, `kmemory_chain`,
  and `mujoco_masked` with `agent_type=rtu_rtrl`.
- `memo_rtrrl` statically supports only Hopper with
  `rtrrl_topology=shared`.
- Both launchers expose only required `--config`, bootstrap the SDK once on the
  host, and reuse the existing memo config, environment, agent, and training
  APIs.
- `max_episode_steps` now reaches Brax `EpisodeWrapper` and
  `BraxGymnaxWrapper.default_params`; it is no longer hard-coded to 1000.
- Training budgets are partitioned in vector-environment interaction units.
  Unrepresentable budgets are rejected, and logging uses `int(state.step)`.
- Every completed `RecordEpisodeStatistics` record produces its own mandatory
  episode summary.
- Complete fixed-shape evaluation traces are converted on the host to SDK
  `Episode` values with N+1/N lengths, separate terminal/truncation flags, and
  optional environment states. Nonpositive transition counts are explicitly
  rejected by direct conversion and skipped by cadence emission.
- `LegacyProgram.evaluate()` now returns its trace instead of discarding it.
- SDK and NumPy conversion remain outside JIT. Logger finalization runs on both
  success and failure; a finalization error cannot replace the original
  training exception.
- Added focused concrete YAML fixtures and launcher/observability tests.
- Updated the memo lock with the already-declared standalone training SDK
  Pydantic and PyYAML dependencies; `uv lock --check` passes.

## TDD Evidence

The first targeted run failed during collection because `base.facility` and the
host observability functions did not exist. After correcting test import-path
setup, RED again showed the intended missing module/functions. Implementation
then proceeded to GREEN. A subsequent RED exposed NumPy boolean scalars leaking
through the SDK episode boundary; conversion now emits built-in booleans.

## Verification

- Facility launcher/observability targets: `20 passed`.
- Memo logger compatibility: `21 passed`.
- Standalone training SDK suite: `120 passed`.
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
- the two real-Brax trace cases in an environment with the Brax extra;
- CPU/GPU Docker builds, image smoke tests, and GPU runtime checks;
- any real AWS, Batch, ECR, S3, Aim service, or paid job operation.

No catalog, Dockerfile, workflow, controller, or AWS resource code was changed.
