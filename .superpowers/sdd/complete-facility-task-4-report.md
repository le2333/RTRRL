# Complete Facility Task 4 Report

## Status

Implemented Task 4 on baseline `00c8238` without AWS, ECR, Batch, image push,
or containerized JAX training. The formal memo facility now has a protocol-v1
catalog for exactly `memo_stream_ac` and `memo_rtrrl`, repository-root
CPU/GPU image definitions, deterministic catalog labels, fixed worker/script
paths, lock-consistent CPU/CUDA dependencies, and an additive GitHub build path.
The review follow-up also moved the retained legacy CPU/GPU builds to the same
repository-root context so their `../training-sdk` dependency works from a clean
checkout.

## Delivered

- Added `memo/infra/scripts/index.yaml` and exactly two descriptors.
- `memo_stream_ac` launches
  `python /app/experiments/memo_stream_ac/run.py --config {config_path}`,
  supports only `memory_chain`, `kmemory_chain`, and `mujoco_masked`, and fixes
  `agent_type` to `rtu_rtrl`.
- `memo_rtrrl` launches
  `python /app/experiments/memo_rtrrl/run.py --config {config_path}`, supports
  only `hopper`, and fixes `rtrrl_topology` to `shared`.
- Both descriptors use the real Aim objective `eval/rewards`. Every descriptor
  path uses the Task 3 `algorithm.*`, `network.*`, or `runtime.*` namespace and
  materializes through `FacilityInput` into the real memo config dataclasses.
- Added `Dockerfile.facility` and `Dockerfile.facility.gpu`.
- Minimally updated the historical memo Dockerfiles in place (same paths,
  entrypoints, commands, and tags) to copy `memo/*` from repository-root
  context, copy the shared `training-sdk` path dependency before frozen sync,
  and use the lock-defined CUDA extra instead of an independent JAX install.
- Both formal images use repository-root context and frozen, non-editable,
  no-development, multi-stage installs. Runtime stages contain the installed
  virtual environment, memo source, shared `training_sdk`, worker runtime
  dependency (`boto3`), descriptors, and no control-plane package.
- Formal image paths are `/opt/trainer/worker.py` and
  `/opt/trainer/scripts/*`. The required nonempty
  `org.rtrrl.trainer.scripts.v1` build argument is guarded before labeling.
- CPU resolves lock-pinned JAX/JAXLIB 0.10.0. GPU resolves the matching
  `jax[cuda12]==0.10.0` plugin/PJRT and CUDA 12 userspace wheels, statically
  targeting the g6.xlarge L4 host-driver model; dynamic acceptance is pending.
- Added a repository-root, allowlist-first `.dockerignore` admitting only memo,
  training-sdk, and the worker while excluding VCS/worktree metadata, virtual
  environments, caches, history, Aim/Optuna state, artifacts, logs, W&B state,
  and common secret/credential patterns.
- Extended `build-memo-image.yml` additively. Both retained legacy tags and both
  formal tags now build from repository-root context; Dockerfile paths, tags,
  entrypoints, and commands remain stable. The separate formal job builds
  `memorax-rtrl-facility-cpu` and
  `memorax-rtrl-facility-gpu` with the canonical catalog build argument.

## TDD Evidence

The first catalog run produced seven expected failures because
`memo/infra/scripts/index.yaml` and both descriptors were absent. After the
catalog implementation, catalog/field/objective tests passed; the real
materialization tests then exposed missing cross-project test environment
paths and were corrected before passing in the memo environment.

The first formal-image contract run produced seven expected failures because
the two formal Dockerfiles, root `.dockerignore`, runtime extras, and additive
workflow job were absent. Those tests passed after the minimal implementation.

The review follow-up added executable encoder, parsed-workflow, actual-context,
COPY-source, path-dependency, legacy clean-build, exact-field, and import
isolation contracts. The first strengthened image run produced 16 expected
failures for the legacy memo-context build, broad context rules, missing secret
patterns, absent shared SDK copies, and unparsed workflow assumptions. The
root-context implementation made those contracts pass.

## Verification

- Full fake-only control-plane suite, including catalog encoder, descriptor,
  workflow, Docker context, and COPY-source contracts: `412 passed`.
- Memo catalog-to-materialize-to-`FacilityInput`-to-real-builder plus facility
  launcher/observability targets: `38 passed` with one upstream JAX
  deprecation warning.
- Standalone training SDK suite: `129 passed`.
- Full control-plane and SDK Ruff plus changed memo Ruff targets: passed.
- Memo `uv lock --check`: passed (`306` packages resolved).
- Facility runtime import smoke: `boto3`, `training_sdk`, and
  `training_sdk.execution.JobBundle` imported successfully.
- Catalog CLI produced the same nonempty 1248-character payload twice; the ECR
  decoder read protocol `1` and exactly the two memo scripts.
- Workflow YAML parsed with exactly the retained `build` job and additive
  `build-facility` job.
- `git diff --check`: passed.
- IDE lint on edited Python tests: no findings.
- Historical Dockerfile paths, workflow entry, tags, entrypoints, commands,
  RTRRL descriptors, and RTRRL workflow remain present; only the memo COPY/sync
  paths and workflow context changed to support clean repository-root builds.

## Not Run

Docker CLI is installed, but the current user cannot access
`/var/run/docker.sock`; `docker version` failed with permission denied before
any build started. Therefore CPU/GPU image builds, image inspection, fixed-path
in-container imports, L4/CUDA detection, and GPU runtime acceptance were not run
locally. GPU dynamic acceptance remains for the later authorized Batch phase.
No attempt was made to elevate privileges. No AWS/ECR/Batch operation or full
JAX training was run.
