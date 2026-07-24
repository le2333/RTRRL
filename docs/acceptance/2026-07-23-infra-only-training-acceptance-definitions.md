# Infrastructure-Only Training Acceptance — Job Definitions

## Scope and evidence identity

- Branch: `feature/trainer-infra`
- Repository HEAD before this evidence-only commit:
  `ccfba9b6cfa3c362bb77b81ca799b570db8a6518`
- AWS account: `007122174918`
- Region: `eu-north-1`
- Task 10 prerequisite:
  `docs/acceptance/2026-07-23-infra-only-training-acceptance-images.md`
- Registration output:
  `/tmp/infra-only-training-acceptance-definitions.json`
- Registration output SHA-256:
  `b5fa75e65482fcf48ac70573c390381a0d5dcfc0f274997439b9a188e9141761`
- Exact-ARN readback:
  `/tmp/infra-only-training-acceptance-definitions-readback.json`
- Exact-ARN readback SHA-256:
  `ea4842d69333e060bb26bc60b0ce5ea57499901968e04d4487b00ca7ee8bc5d2`
- Phase A preflight:
  `/tmp/infra-only-training-acceptance-phase-a.json`
- Phase A preflight SHA-256:
  `56bfb35f78f725e3b87527b7a6d99b4c33cdab03f4c278676328d47bf10b9ab3`

The three `/tmp` files are runtime evidence and are not committed. No AWS or
GitHub mutation was performed while preparing this document.

## Pre-registration evidence and chronology

The Phase A preflight report returned overall `pass`, account and region
`match`, and all four profiles `ready`: `c7am` at `1` vCPU and `1600` MiB,
`c7al` at `2` vCPUs and `3200` MiB, `c7ax` at `4` vCPUs and `7168` MiB, and
`g6x` at `4` vCPUs, `12000` MiB, and `1` GPU. That report predates the Task 10
push: it records ECR status `pending` and both fixed tags as `missing`, not
visible.

Task 10 subsequently recorded a separately authorized, read-only
`ecr:BatchGetImage` verification with `failures: []`. It made these exact
immutable images visible:

- CPU:
  `007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@sha256:ec1ae1e426313dbbda1567dc48eb942f7b76c1b0d91b3738584ef5a9fda91a08`
- GPU:
  `007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@sha256:938fb1bfce6131ecd3f40a56475946e8865eed6f902b74f15c3e808f9da97e6a`

The Phase A IAM simulation did not succeed.
`iam:SimulatePrincipalPolicy` returned `AccessDenied` for the controller role,
so the canonical IAM result remains `unknown` / `unavailable`, with
`blocking:false`. This evidence does not claim that the requested IAM actions
were proven allowed and no permission was changed.

## Initial CLI attempt and fail-closed boundary

The first authorized registration invocation supplied bare
`sha256:<64 lowercase hex>` values to `--cpu-digest` and `--gpu-digest`. After
the read-only caller-account check, local validation rejected the CPU argument
with:

```text
ValueError: CPU digest must be 007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@sha256:<64 lowercase hex>
```

The process exited before ECR catalog verification and before
`RegisterJobDefinition`; this attempt registered `0` definitions. The failure
did not broaden authorization or trigger a Batch submission.

## Authorized retry

The retry remained within the separate authorization for exactly four
registrations and supplied the full repository-qualified immutable references:

```bash
cd rtrrl/infra/control-plane
uv run python scripts/deploy_facility.py \
  --control config/facility.yaml \
  --register --confirm-account 007122174918 \
  --cpu-digest "007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@sha256:ec1ae1e426313dbbda1567dc48eb942f7b76c1b0d91b3738584ef5a9fda91a08" \
  --gpu-digest "007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@sha256:938fb1bfce6131ecd3f40a56475946e8865eed6f902b74f15c3e808f9da97e6a" \
  | tee /tmp/infra-only-training-acceptance-definitions.json
```

The output records `mode:"execute"`, `requested.register:true`,
`retry_attempts:1`, exactly four nonempty job-definition ARNs, and
`submission_supported:false`.

## Exact ACTIVE revision-1 readback

Read-only `batch:DescribeJobDefinitions` requests for each exact ARN returned
exactly the four requested definitions. Every definition has revision `1`,
status `ACTIVE`, type `container`, platform capability `EC2`, retry attempts
`1`, command `["python", "/opt/trainer/worker.py"]`, and environment
`TRAINER_WORKER_PROTOCOL_VERSION=1`. Every definition also has:

- job role:
  `arn:aws:iam::007122174918:role/rtrrl-batch-job-role`
- execution role:
  `arn:aws:iam::007122174918:role/rtrrl-batch-execution-role`

The profile-specific exact readback is:

- `c7am`:
  `arn:aws:batch:eu-north-1:007122174918:job-definition/trainer-c7am-ec1ae1e426313dbbda1567dc48eb942f7b76c1b0d91b3738584ef5a9fda91a08:1`;
  CPU image; `1` vCPU; `1600` MiB; no GPU resource requirement.
- `c7al`:
  `arn:aws:batch:eu-north-1:007122174918:job-definition/trainer-c7al-ec1ae1e426313dbbda1567dc48eb942f7b76c1b0d91b3738584ef5a9fda91a08:1`;
  CPU image; `2` vCPUs; `3200` MiB; no GPU resource requirement.
- `c7ax`:
  `arn:aws:batch:eu-north-1:007122174918:job-definition/trainer-c7ax-ec1ae1e426313dbbda1567dc48eb942f7b76c1b0d91b3738584ef5a9fda91a08:1`;
  CPU image; `4` vCPUs; `7168` MiB; no GPU resource requirement.
- `g6x`:
  `arn:aws:batch:eu-north-1:007122174918:job-definition/trainer-g6x-938fb1bfce6131ecd3f40a56475946e8865eed6f902b74f15c3e808f9da97e6a:1`;
  GPU image; `4` vCPUs; `12000` MiB; `1` GPU.

The CPU definition names and container images contain the exact Task 10 CPU
digest; the GPU definition name and container image contain the exact Task 10
GPU digest. The registration output's four ARNs and both image references match
the exact-ARN readback without substitution.

## Authorization boundary and mutation counters

The authorization covered only the four successful job-definition
registrations. It did not authorize submission, cancellation, cleanup, S3
writes, or IAM changes. The observed Task 11 counters are:

- Successful Batch job-definition registrations: `4`
- Registrations from the rejected bare-digest attempt: `0`
- Batch job submissions: `0`
- Batch job cancellations: `0`
- Acceptance cleanup operations: `0`
- S3 object writes: `0`
- S3 object deletes: `0`

The deployment output explicitly records `submission_supported:false`. This
phase therefore proves only the four digest-bound definition contracts and
does not claim a runnable Batch job, successful training, cleanup, or broader
IAM readiness.
