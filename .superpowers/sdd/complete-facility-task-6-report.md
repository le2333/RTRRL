# Complete Facility Task 6 Report

## Status

Completed Task 6 from baseline `fd195c4` with a fake-only local integration
harness. No AWS operation, Docker command, or real JAX training was run.

The harness uses the real image catalog codec and ECR reader, resolver,
materializer, controller, `RunBundle`/`JobBundle`, worker entry,
`FacilityInput`/memo builder seams, training SDK bootstrap and lifecycle,
durable Aim spool, Rerun writer, `AimReader`, and Optuna studies. Fakes replace
only ECR, S3, Batch, and Aim service boundaries.

## Delivered

- Added `test_end_to_end.py`, whose fake Batch executes the real worker entry
  and real child process command for two lightweight memo launcher fixtures.
  The fixtures use the real `FacilityInput` and builder seams, SDK bootstrap,
  metric and episode-summary logging, an eval `Episode`, checkpoint
  registration, Rerun emission, Aim finalization, and spool upload without
  invoking training or JAX.
- Proved two independent groups each allocate `2+2+1`, while each controller
  round interleaves the groups into mixed two-child bundles. The first two
  rounds submit two jobs before polling, and the final round submits one.
  Completion-marker order proves children execute serially within each worker.
- Proved exact YAML protocol and seed materialization plus experiment name,
  internal experiment/group/run identities, digest-bound image, `c7am`
  profile, and attempt zero across contexts, bundles, markers, and artifact
  keys.
- Proved objective values reach real Optuna `tell`, every accepted trial is
  terminal, finite spaces stop before the nominal budget, and no later round is
  submitted after a failure.
- Proved Batch `FAILED` and poll timeout, real child nonzero, run-input and
  `JobBundle` upload failures, partial submit exception, artifact upload
  failure, missing/tampered completion marker, Aim failed/timeout/nonfinite
  result, and one-or-both final persistence failures. Accepted Batch IDs are
  retained; every asked pending trial becomes `FAIL`; no resubmit, cancel,
  retry, or future submit occurs.
- Proved each run's completion marker independently owns exactly one Aim spool,
  checkpoint, and Rerun artifact under that run's artifact prefix. Artifact
  URIs are one-to-one across runs, exist in fake S3, and match stored SHA-256.
- Added an explicit 1,428-entry
  `tests/data/historical-fd195c4.json` manifest with path, baseline git blob
  SHA, and category for every selected historical entry.

## Implementation Defects Exposed and Fixed

1. **The controller serialized whole groups.** The old outer loop completed
   every batch for one group before creating the next study, so ready work from
   independent groups could never coexist, jobs could not mix groups, and
   polling observed one job at a time. `_GroupLoop` now retains independent
   study/sampling state while one global round asks every ready group,
   round-robin orders their runs, partitions only by image digest, resource
   profile, and `runs_per_job`, submits the complete round once, and tells on
   the controller thread. Round failure terminates pending trials in every
   affected group.
2. **`TrainingRun.register_checkpoint()` discarded its input.** The worker only
   uploads the reserved `checkpoints/` tree, so the SDK no-op made checkpoint
   registration unverifiable and silently lost the artifact. It now validates
   a regular non-symlink source and copies a bounded stream into the run-owned
   `checkpoints/` directory through a same-directory temporary file, file
   `fsync`, atomic no-overwrite hard-link publication, and directory `fsync`.
   Duplicate and same-path registration reject without deleting the existing
   artifact; failures remove only this call's temporary file.
3. **Default Rerun output bypassed the worker exchange namespace.** Bootstrap
   rooted Rerun directly at `artifact_directory`, while the worker intentionally
   uploads only `aim-buffer/`, `checkpoints/`, and `rerun/`. The default root is
   now `artifact_directory / "rerun"`, so real Rerun files are included in the
   completion marker and S3 artifact set.

## Historical Compatibility Inventory

The reviewable manifest is
`rtrrl/infra/control-plane/tests/data/historical-fd195c4.json`. Its
baseline-derived union contains 1,428 unique paths:

- 44 commands selected by baseline mode `100755`, shebang, Python
  `__main__` guard, or a `pyproject.toml` `project.scripts` registration and
  target. This includes non-shell entries and `infra/submit.sh`.
- 3 workflows under `.github/workflows/`.
- 7 memo/RTRRL catalog index and descriptor YAML files.
- 1,375 complete HPO source/data entries under `infra/hpo/` and `rtrrl/hpo/`,
  without suffix filtering and explicitly including `infra/hpo/uv.lock`.

The test independently recomputes the selection from
`git ls-tree --full-tree fd195c4`, requires exact manifest equality, and recomputes each
current file's git blob SHA from its bytes. Every entry exists and is unchanged.
Workflows, descriptors, and HPO specs parse as YAML. Every selected shell
command is syntax-checked with `bash -n`; safe Python CLI help surfaces and the
historical HPO dry run are exercised. A fake `aws` executable proves those
checks never crossed the AWS boundary. No historical file was modified.

### Historical concern

`rtrrl/infra/submit.sh` does **not** exist in baseline `fd195c4`; the baseline
canonical shared entry is `infra/submit.sh`, consistent with `rtrrl/AGENTS.md`.
The test explicitly locks this evidence rather than creating or claiming a
nonexistent historical path. An intermediate attempt to add `--help` behavior
to `infra/submit.sh` was rejected during diff review and fully restored before
verification.

## TDD Evidence

- The first success integration run failed because controller polling was
  `[1, 1, 1, 1, 1, 1]`, proving groups were serialized instead of producing
  mixed concurrent jobs.
- After the scheduling fix, the same test failed first on absent checkpoint
  artifacts and then on absent `rerun/` artifacts before the respective SDK
  fixes were made.
- Failure-boundary tests initially retained only one submitted job and created
  only one group study; after the controller fix they retain both first-round
  jobs and fail both groups' pending trials.
- Historical compatibility initially exposed that invoking the shell submit
  entry as `--help` sourced environment/AWS logic. Since changing that
  historical behavior was outside Task 6, the implementation change was
  removed and the test was corrected to use non-mutating syntax/help/dry-run
  surfaces.
- Review RED tests reproduced existing checkpoint deletion on duplicate and
  same-path registration. Both failed because the old broad exception handler
  unlinked the final target rather than a call-owned temporary file.
- The new submission-boundary parameterization first completed all four modes
  because the harness lacked those injected boundaries; after injection it
  proves Batch timeout, input upload, job bundle upload, and partial submit
  semantics independently.
- The first DI-seam test failed with an unsupported `aim_reader_factory`
  constructor argument. The controller now formally supports exactly one
  fixed reader or store-bound reader factory, so Task 6 tests no longer modify
  controller private attributes.

## Verification

- Task 6 targeted:
  `cd rtrrl/infra/control-plane && uv run pytest tests/test_end_to_end.py tests/test_historical_entries.py -q`
  — 21 passed in 68.86 seconds, one upstream Aim/SQLAlchemy warning.
- Full control plane:
  `cd rtrrl/infra/control-plane && uv run pytest -q`
  — 467 passed in 100.62 seconds.
- Full training SDK:
  `cd training-sdk && uv run pytest -q`
  — 132 passed in 8.43 seconds.
- Checkpoint review target:
  `cd training-sdk && uv run pytest tests/test_aim_adapter.py -k checkpoint -q`
  — 3 passed, 37 deselected.
- Memo lightweight facility targets:
  `cd memo && uv run pytest tests/test_facility_catalog.py tests/test_facility_launchers.py tests/test_experiment_observability.py tests/test_logging_compat.py -q`
  — 60 tests passed (exit 0), with one upstream JAX deprecation warning; no
  training function was executed.
- Ruff:
  `cd rtrrl/infra/control-plane && uv run ruff check src tests` and
  `cd training-sdk && uv run ruff check src tests`
  — both passed.
- Lock checks:
  `uv lock --check` in control plane, training SDK, memo, and `infra/hpo`
  — passed, resolving 62, 59, 306, and 60 packages respectively.
- `git diff --check` — passed.
- IDE diagnostics for every changed Python implementation and test file — no
  findings.

## Review Remediation Concerns

- The manifest deliberately defines "command" by auditable baseline evidence:
  executable mode, shebang, Python main guard, or `project.scripts`
  registration/target. The checked-in per-path blob list is the source of truth
  and the test detects both omissions and additions relative to that rule.
- Upstream Aim/SQLAlchemy and JAX deprecation warnings remain; neither is caused
  by Task 6.

## Not Run

- Real AWS, ECR, S3, Batch, or Aim service operations.
- Docker builds or image execution.
- Real JAX/Brax training, full memo online-AC/OOM-prone suites, or GPU tests.
