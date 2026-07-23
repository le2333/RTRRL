# Complete Facility Task 7 — Phase A Acceptance

## Reproduction Identity

- Generated at (UTC): `2026-07-23T18:30:09Z`
- Baseline: `c763673`
- Review hardening commits:
  `64668aecc5aa80e54493c98fc48fb80b2b778b04`,
  `30427f2a1ddea4c1833660734e70036e4b211009`
- Result: `PASS`
- AWS writes performed: none
- Image builds performed: none
- ECR pushes performed: none
- Job definitions registered: none
- Batch jobs submitted: none
- S3 objects written or deleted: none

## Commands

Run from `rtrrl/infra/control-plane`:

```bash
uv run python scripts/start_facility_aim.py --control config/facility.yaml
uv run python scripts/facility_preflight.py \
  --control config/facility.yaml \
  --output /tmp/complete-facility-task7-phase-a-review-preflight.json
uv run pytest \
  tests/test_facility_control.py \
  tests/test_aim_scratch.py \
  tests/test_facility_preflight_review.py \
  tests/test_facility_deploy_review.py \
  tests/test_facility_deployment.py \
  tests/test_aws_batch.py -q
uv run pytest -q
uv run ruff check src tests scripts
git diff --check
```

No command contains an account credential, authorization token, registry
password, or other secret.

## Caller and Region

- Caller ARN:
  `arn:aws:sts::007122174918:assumed-role/controller/i-024ef6f6b61fffdcc`
- Account: `007122174918`
- Region: `eu-north-1`
- S3: `s3://rtrrl-artifacts-007122174918/experiments/`
- ECR repository:
  `007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl`

`iam:SimulatePrincipalPolicy` is unavailable to the caller. This is recorded as
`unknown/unavailable`, is not a preflight blocker, and does not justify adding
IAM permissions. Later explicit push, registration, and run operations will
surface their own authorization errors after their separate approvals.

## Authoritative Profile Validation

The preflight uses Task 2 `AwsBatchPreflight.validate_profiles()` rather than a
second profile table. It validates managed/enabled/valid compute environments,
EC2 type, minimum and maximum vCPUs, exact instance type, AL2023 image variant,
empty AMI override, subnets, security group, instance profile, run queue
priority 100, and exact queue-to-CE binding.

| Profile | Run queue | Compute environment | Instance | Job resources |
| --- | --- | --- | --- | --- |
| `c7am` | `run-cpu-c7am-queue` | `rtrrl-cpu-c7am-ce` | `c7a.medium` | 1 vCPU, 1600 MiB |
| `c7al` | `run-cpu-c7al-queue` | `rtrrl-cpu-c7al-ce` | `c7a.large` | 2 vCPU, 3200 MiB |
| `c7ax` | `run-cpu-c7ax-queue` | `rtrrl-cpu-c7ax-ce` | `c7a.xlarge` | 4 vCPU, 7168 MiB |
| `g6x` | `run-gpu-queue` | `rtrrl-gpu-g6x-ce` | `g6.xlarge` | 4 vCPU, 12000 MiB, 1 GPU |

All four profile checks passed. The S3 bucket/location/prefix check passed.
Both formal ECR tags are currently absent, which is the expected pre-push state.

## Raw Preflight Evidence

- Path: `/tmp/complete-facility-task7-phase-a-review-preflight.json`
- SHA-256:
  `961b41b81f360d1e198d381fdeed6e5d44bea7b424de2b5f30f549c575e28210`
- Schema version: `1`
- Stable JSON status: `pass`

The raw report contains no credential or token.

## Deployment Safety Properties

- `--push` and `--register` each require the exact argument
  `--confirm-account 007122174918`.
- Before either mutation, STS caller account and session region must match the
  committed facility control. A mismatch occurs before ECR or Batch clients are
  constructed.
- `--build` is local and independent. `--push` can publish already-built formal
  tags without forcing a rebuild.
- Job and execution role ARNs come only from `config/facility.yaml`; the CLI
  accepts no arbitrary role ARN.
- Registration remains digest-bound and uses native retry attempts equal to
  one for all four profiles.
- ECR login uses a temporary `DOCKER_CONFIG`, which is removed on success or
  failure.
- Push digests are parsed from Docker push output. Catalog verification then
  addresses the immutable digest through the existing ECR reader; it does not
  resolve the tag again.
- Build verification checks the catalog label by real decode, the fixed worker
  path, `training_sdk`, both memo launcher imports, and CPU/GPU JAX package
  variants.
- The deploy script has no Batch submission or cleanup path.

## Paid Job Derivation

The committed smoke example has two groups:

- `stream`: profile `c7am`
- `rtrrl`: profile `g6x`

Each group requests five trials, two configurations per batch, and two child
runs per job. Therefore each group needs `ceil(5 / 2) = 3` Batch jobs with child
counts `2 + 2 + 1`.

Expected paid acceptance total:

- `c7am`: 3 jobs
- `c7al`: 0 jobs
- `c7ax`: 0 jobs
- `g6x`: 3 jobs
- Total: 6 jobs

This is an estimate only. No job was submitted in Phase A.

## Verification

- Targeted review suite: `64 passed`
- Full control-plane suite: `495 passed`
- Ruff over `src`, `tests`, and `scripts`: passed
- IDE lint on changed implementation files: no findings
- `git diff --check`: passed

## Dynamic Build Inputs

The formal Dockerfiles still use `python:3.12-slim` and
`ghcr.io/astral-sh/uv:latest`. No trusted digest for either base was available
locally, and Phase A did not perform a build or registry resolution. They remain
explicit dynamic-build concerns for the separately authorized image phase; this
report does not invent or claim pinned digests.

## Runtime Appendix

At report generation time:

- Aim scratch repository:
  `/home/ubuntu/trainer/task7-aim-scratch`
- Endpoint: `aim://127.0.0.1:53801`
- Recorded supervisor PID: `56049`
- PID file:
  `/home/ubuntu/trainer/task7-aim-scratch/aim-server-53801.pid`
- Metadata file:
  `/home/ubuntu/trainer/task7-aim-scratch/aim-server-53801.json`

The preflight matched the recorded PID, exact command line, working directory,
port, repository argument, isolated path, and live health endpoint. The PID is
runtime-only evidence and is not a durable deployment identity.
