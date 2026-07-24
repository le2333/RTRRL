# Infrastructure-Only Training Acceptance — Phase A

## Reproduction identity

- Evidence generated at (UTC): `2026-07-24T07:20:00Z`
- Branch: `feature/trainer-infra`
- HEAD: `c839f80ca548e1e90d7bf6b0aecaf4f1dbb7ccfc`
- Live merge base with local `main`:
  `1551fda2ecb92dc6351113fb3ee77e55bfe56cd0`
- Read-only preflight result: `pass`
- AWS account: `007122174918` (`match`)
- Region: `eu-north-1` (`match`)
- Caller ARN:
  `arn:aws:sts::007122174918:assumed-role/controller/i-024ef6f6b61fffdcc`

This phase performed no Docker or GitHub operation and no AWS mutation.
The only AWS-facing command was the facility read-only preflight.

## Explicit mutation counters

- ECR image pushes: `0`
- Batch job-definition registrations: `0`
- Batch job submissions: `0`
- S3 object writes: `0`
- S3 object deletes: `0`

CloudTrail evidence was unavailable. The mutation boundary is instead supported
by an audit of the preflight's reachable client calls. Its complete allowlist
was `sts:GetCallerIdentity`, `batch:DescribeComputeEnvironments`,
`batch:DescribeJobQueues`, `s3:HeadBucket`, `s3:GetBucketLocation`,
`s3:ListObjectsV2`, `ecr:DescribeRepositories`, `ecr:BatchGetImage`, and
`iam:SimulatePrincipalPolicy`. No reachable create, update, register, put,
delete, submit, upload, start, terminate, modify, tag, or untag client call was
present. The canonical report records top-level, ECR, and S3
`writes_performed:false`.

## Local gate

The fresh gate began at `2026-07-24T06:32:47Z` and exited `0` after
`2,750.752` seconds:

```bash
scripts/verify-infra-only-acceptance.sh
```

- Initial and final protected-tree checks matched merge base
  `1551fda2ecb92dc6351113fb3ee77e55bfe56cd0` by path, blob, and mode.
- Training SDK lock check passed; tests: `132 passed`; peak RSS:
  `188,332 KiB`; Ruff passed.
- Mock trainer lock check passed; tests: `94 passed` with `51 warnings`; peak
  RSS: `1,209,060 KiB`; Ruff passed.
- Control-plane lock check passed; tests: `548 passed` with `16 warnings`;
  peak RSS: `760,940 KiB`; Ruff passed.
- Both forbidden-reference scans returned the expected no-match status `1`.
- `git diff --check` passed.

The gate ran with `UV_OFFLINE=1`, explicit `uv --offline` operations, CPU-only
JAX settings for the trainer suites, fixed timeouts, and per-command maximum
RSS reporting.

## Read-only facility preflight

Run from `rtrrl/infra/control-plane` after the local gate:

```bash
uv run --offline python scripts/facility_preflight.py \
  --control config/facility.yaml \
  | tee /tmp/infra-only-training-acceptance-phase-a.json
```

- Canonical report:
  `/tmp/infra-only-training-acceptance-phase-a.json`
- Canonical report SHA-256:
  `56bfb35f78f725e3b87527b7a6d99b4c33cdab03f4c278676328d47bf10b9ab3`
- Schema version: `1`
- Overall status: `pass`
- S3 bucket: `rtrrl-artifacts-007122174918`
- S3 prefix: `experiments/`
- S3 status: `visible`; region: `eu-north-1`; sample count: `0`

The generated JSON is runtime evidence under `/tmp` and is not committed.

## Profiles, queues, and compute environments

All four profiles, all eight queues, and all four compute environments passed
the committed concrete-contract validation:

- `c7am`: CE `rtrrl-cpu-c7am-ce`; dev queue
  `dev-cpu-c7am-queue` priority `10`; run queue
  `run-cpu-c7am-queue` priority `100`; `1` vCPU, `1600` MiB, `0` GPUs.
- `c7al`: CE `rtrrl-cpu-c7al-ce`; dev queue
  `dev-cpu-c7al-queue` priority `10`; run queue
  `run-cpu-c7al-queue` priority `100`; `2` vCPUs, `3200` MiB, `0` GPUs.
- `c7ax`: CE `rtrrl-cpu-c7ax-ce`; dev queue
  `dev-cpu-c7ax-queue` priority `10`; run queue
  `run-cpu-c7ax-queue` priority `100`; `4` vCPUs, `7168` MiB, `0` GPUs.
- `g6x`: CE `rtrrl-gpu-g6x-ce`; dev queue `dev-gpu-queue` priority
  `10`; run queue `run-gpu-queue` priority `100`; `4` vCPUs,
  `12000` MiB, `1` GPU.

## Image visibility and IAM limitation

The ECR repository was visible at
`007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl`. Both test labels were
missing:

- `infra-acceptance-brax-ppo-cpu-20260723`: `missing`
- `infra-acceptance-brax-ppo-gpu-20260723`: `missing`

Consequently ECR status was `pending`, not `visible`. This is the expected
pre-push Phase A state and is not evidence that either image exists.

`iam:SimulatePrincipalPolicy` returned `AccessDenied` for
`arn:aws:iam::007122174918:role/controller`. The canonical result is therefore
`unknown/unavailable`, with `blocking:false`; this report does not claim that
the requested IAM actions were proven allowed. Account and caller identity
matched, but IAM policy readiness remains unverified by simulation. No
permission or AWS resource was changed in response.

## Aim scratch migration and identity

Before preflight, the legacy scratch supervisor had PID `56049`, start time
`1406401` ticks, exact repository
`/home/ubuntu/trainer/task7-aim-scratch`, and no lifecycle lock or recorded
start-time identity. Its listener child was PID `56055`, start time `1406512`
ticks, on `127.0.0.1:53801`.

A one-use local Python migration verified the legacy metadata and PID file,
exact command lines, parent/child relationship, working directories, repository
and filesystem-root inodes, repository argument, port, and listener socket
ownership. It opened a pidfd, confirmed the generation was stable and the
pidfd was not ready, sent `SIGTERM` through the pidfd to exact PID `56049`, and
waited on that pidfd. The listener child did not exit with its parent, so the
migration stopped without signaling any unverified process. PID `56055` was
then independently revalidated against its fixed uvicorn command, start time,
working directory, root inode, and sole ownership of listener inode `363563`;
it was terminated with the same pidfd generation checks. Both pidfds reported
exit, the exact repository-bearing Aim process scan was empty, and port `53801`
was closed before the unchanged legacy metadata and PID evidence were removed.
No main Aim repository or data was touched.

The committed launcher then started and subsequently resumed the new scratch
server. Preflight recorded:

- PID: `341383`
- Start time: `5782057` ticks
- Endpoint: `aim://127.0.0.1:53801`
- Repository root device/inode: `66305` / `527503`
- Status: `ready`
- Command, working directory, port, repository, isolation, and health checks:
  all `true`

The new private `.trainer-aim-scratch.lock` existed and a separate
non-blocking lock attempt confirmed it was held. The metadata contains
`start_time_ticks`, `repo_root_dev`, `repo_root_ino`, `repo_fd`, and
`trusted_repo`. These runtime-only lifecycle files remain outside Git.

## Phase boundary

Phase A did not build an image, push a branch or image, register a job
definition, submit a Batch job, or write/delete an S3 object. The two missing
test labels and unavailable IAM simulation are recorded without promotion to
ready. Any image publication, registration, execution, or cleanup remains a
separately authorized later phase.
