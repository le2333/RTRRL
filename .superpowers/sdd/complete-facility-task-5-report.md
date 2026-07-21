# Complete Facility Task 5 Report

## Status

Implemented Task 5 from baseline `f0bffc0`: exact Aim result collection,
read-only validation, foreground experiment orchestration, formal CLI wiring,
control configuration, examples, and failure-boundary coverage.

## Delivered

- `AimReader` derives the Aim hash only from the exact `run_id`, replays the
  injected run spool before reading, verifies the stored exact identity,
  rejects `sdk/failed`, and accepts only the exact finalized objective with a
  finite non-boolean numeric value. Its timeout uses injected monotonic clock
  and sleep functions.
- `ExperimentController.validate()` parses and resolves the experiment,
  resolves each ECR reference to one digest-bound catalog, validates schemas,
  parameters, budgets, profiles, job grouping, and read-only preflight
  contracts. It does not create studies, stores, Batch adapters, experiment
  IDs, Aim runs, or AWS resources.
- `ExperimentController.run()` creates one fresh experiment ID per invocation,
  creates isolated studies per group, and advances automatic HPO rounds through
  ask, sample, materialize, bundle, upload, submit, query, marker verification,
  exact Aim collection, and controller-thread-only `study.tell()`.
- Batch rounds submit each prepared concurrent job once and never retry,
  resubmit, cancel, or start a later HPO round after failure. Batch failure,
  nonzero child exit, missing/tampered marker, Aim failure/timeout/nonfinite
  objective, and final persistence failure all propagate.
- Failure reports preserve every accepted Batch job ID. Final state and report
  are each attempted once under the fresh experiment prefix.
- `trainerctl` exposes only foreground `validate` and `run`. Success writes
  stable JSON to stdout; errors write stable JSON to stderr and return nonzero.
  AWS, Optuna, and Aim runtime construction is lazy and explicit.
- The control example includes region, account, bucket, exact prefix, Aim repo,
  poll/timeout settings, network contract, IAM roles, and all four digest-bound
  job definitions. The experiment example covers both memo launchers.
- Historical configuration, HPO data, commands, descriptors, and workflows
  were not modified or removed.

## TDD Evidence

RED was observed before implementation for missing `aim_reader`, `controller`,
and `cli` modules. GREEN coverage includes exact Aim identity and finalization,
injected spool replay and monotonic timeout, read-only validation call logs,
two isolated groups completing automatic `2+2+1`, fresh IDs, controller thread
ownership of every tell, all required fail-fast boundaries, concurrent
submitted-ID retention, one-shot final persistence, and exact CLI surface and
streams.

## Verification

- Task 5 targeted suite: 22 passed.
- Full control-plane suite: 434 passed.
- Full training SDK suite: 129 passed.
- Ruff over control-plane `src` and `tests`: passed.
- Control-plane lock check: passed.
- Control and experiment examples: parsed successfully.
- CLI module/help, `validate --help`, and `run --help`: passed.
- `git diff --check`: passed.
- IDE lint for the control-plane tree: no findings.

No AWS mutation, Docker, JAX, daemon, background service, or paid workload was
run.
