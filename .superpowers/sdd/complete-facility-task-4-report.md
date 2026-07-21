# Complete Facility Task 4 Report

## Status

Implemented Task 4 on baseline `00c8238` without AWS, ECR, Batch, image push,
or containerized JAX training. The formal memo facility now has a protocol-v1
catalog for exactly `memo_stream_ac` and `memo_rtrrl`, repository-root
CPU/GPU image definitions, deterministic catalog labels, fixed worker/script
paths, lock-consistent CPU/CUDA dependencies, and an additive GitHub build path.

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
- Added `Dockerfile.facility` and `Dockerfile.facility.gpu`; the historical memo
  Dockerfiles remain unchanged.
- Both formal images use repository-root context and frozen, non-editable,
  no-development, multi-stage installs. Runtime stages contain the installed
  virtual environment, memo source, shared `training_sdk`, worker runtime
  dependency (`boto3`), descriptors, and no control-plane package.
- Formal image paths are `/opt/trainer/worker.py` and
  `/opt/trainer/scripts/*`. The required nonempty
  `org.rtrrl.trainer.scripts.v1` build argument is guarded before labeling.
- CPU resolves lock-pinned JAX/JAXLIB 0.10.0. GPU resolves the matching
  `jax[cuda12]==0.10.0` plugin/PJRT and CUDA 12 userspace wheels suitable for
  the g6.xlarge L4 host-driver model.
- Added a repository-root `.dockerignore` excluding VCS/worktree metadata,
  virtual environments, caches, agent history, Aim/Optuna state, artifacts,
  logs, and W&B state while retaining memo, training-sdk, worker, and
  descriptors.
- Extended `build-memo-image.yml` additively. The original memo-context job,
  Dockerfiles, tags, and entry behavior remain. A separate root-context job
  builds `memorax-rtrl-facility-cpu` and
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

## Verification

- Catalog, ECR codec, image static contracts, and concrete control-plane
  contracts: `39 passed`.
- Memo catalog-to-materialize-to-`FacilityInput`-to-real-builder plus facility
  launcher/observability targets: `38 passed` with one upstream JAX
  deprecation warning.
- Standalone training SDK suite: `129 passed`.
- New control-plane and memo Ruff targets: passed.
- Memo `uv lock --check`: passed (`306` packages resolved).
- Facility runtime import smoke: `boto3`, `training_sdk`, and
  `training_sdk.execution.JobBundle` imported successfully.
- Catalog CLI produced the same nonempty 1248-character payload twice; the ECR
  decoder read protocol `1` and exactly the two memo scripts.
- Workflow YAML parsed with exactly the retained `build` job and additive
  `build-facility` job.
- `git diff --check`: passed.
- IDE lint on edited Python tests: no findings.
- Historical memo Dockerfiles, RTRRL image workflow, descriptors, tags, and
  entries were not modified or removed.

## Not Run

Docker CLI is installed, but the current user cannot access
`/var/run/docker.sock`; `docker version` failed with permission denied before
any build started. Therefore CPU/GPU image builds, image inspection, fixed-path
in-container imports, and GPU runtime detection were not run locally. No
attempt was made to elevate privileges. No AWS/ECR/Batch operation or full JAX
training was run.
