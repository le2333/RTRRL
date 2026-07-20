# Batch Heavy-Test Runner Design

## Purpose

Move memory-intensive JAX/JIT test files out of the local controller host and
run each file in an isolated AWS Batch job. This runner is a development test
facility; it does not change experiment resource profiles or execute training
runs.

## Test Profiles

The runner exposes exactly three explicit profiles in `eu-north-1`:

- `c7am`: existing `rtrrl-cpu-c7am-queue`, `c7a.medium`, 1 vCPU,
  1600 MiB, no GPU;
- `c7ax`: new `rtrrl-cpu-c7ax-queue`, `c7a.xlarge`, 4 vCPU,
  7168 MiB, no GPU;
- `g6x`: existing `rtrrl-gpu-g6x-queue`, `g6.xlarge`, 4 vCPU,
  12000 MiB, one NVIDIA L4 with 24 GiB device memory.

The new `rtrrl-cpu-c7ax-ce` is managed On-Demand EC2 with min/desired vCPUs
zero, max vCPUs 16, the existing Batch instance profile, subnets, and security
group. The runner validates all profile fields before submission and never
silently falls back to another queue or instance type.

## Source and Image Identity

Tests must execute the current worktree, including uncommitted Task 3 changes.
A test-image builder creates a minimal temporary context containing:

- `memo/` source without virtual environments or caches;
- the standalone `training-sdk/`;
- a test Dockerfile layered on a pinned existing memo CPU or GPU image.

The overlay installs only the current standalone SDK and test tooling into the
base virtual environment, then replaces memo source under `/app`. CPU and GPU
test images use unique `trainer-test-<timestamp>-cpu|gpu` tags, are resolved to
digests after push, and are never published under formal project tags.

Job definitions are test-labelled, profile-specific, and bound to the exact
test image digest. The GPU job prints `jax.devices()` before running tests so
the CloudWatch evidence proves that the device is an L4-backed GPU.

## Execution

The runner accepts one profile and one or more exact pytest file paths. It:

1. rejects paths outside `memo/tests/`;
2. submits one Batch job per test file;
3. sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `MALLOC_ARENA_MAX=2`;
4. runs `/usr/bin/time -v python -m pytest <one-file> -q`;
5. records job ID, queue, image digest, exit state, and maximum RSS;
6. returns failure if any job does not reach `SUCCEEDED`.

Combining multiple heavy files in one pytest process is intentionally
unsupported because JAX executables and compilation caches can accumulate until
the process exits. Parallel jobs are allowed only when the selected queue has
capacity.

## Boundaries and Cleanup

- No Aim run is created.
- No formal experiment prefix is written.
- CloudWatch logs and Batch job records are test-labelled.
- Temporary image tags and test job-definition revisions are retained only
  until the associated task evidence has been reviewed.
- The dedicated c7ax queue and compute environment remain available as a
  reusable test profile; they are not removed with per-run cleanup.
- Existing c7am and g6x queue configuration is validated but never mutated.

## Acceptance

- Read-only preflight proves all three exact instance profiles.
- A lightweight command succeeds on c7am.
- one evaluation-trace test file succeeds on c7ax.
- a GPU probe on g6x reports a JAX GPU device and NVIDIA L4.
- separate-file execution prevents local-controller OOM and reports per-job
  peak RSS.
