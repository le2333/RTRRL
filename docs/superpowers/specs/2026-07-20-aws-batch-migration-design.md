# AWS Batch Migration Design

## Purpose

Define the immutable AWS execution profiles, image/job-definition lifecycle,
real smoke-test evidence, cleanup, and safe cutover from superseded queues.

## Existing Target Profiles

The new facility reuses these already-created resources only after strict
read-only validation:

### CPU

- Compute environment: `rtrrl-cpu-c7am-ce`
- Queue: `rtrrl-cpu-c7am-queue`
- Instance type: `c7a.medium`
- Job request: 1 vCPU, 1600 MiB, 0 GPU
- ECS AMI family: AL2023

### GPU

- Compute environment: `rtrrl-gpu-g6x-ce`
- Queue: `rtrrl-gpu-g6x-queue`
- Instance type: `g6.xlarge`
- Job request: 4 vCPU, 12000 MiB, 1 GPU
- Accelerator: one full NVIDIA L4
- ECS AMI family: AL2023 NVIDIA

The old `rtrrl-cpu-queue` and `rtrrl-gpu-queue` are not configuration sources
for the new facility.

## Preflight Validation

Before every experiment submission, the control plane verifies:

- region and account;
- compute environment existence, `VALID` status, enabled state, exact instance
  type, expected provisioning type, subnets, security group, and instance role;
- GPU NVIDIA AMI family;
- queue existence, `VALID` status, enabled state, and exact compute-environment
  binding;
- submitted resource request exactly matches the selected fixed profile;
- required job and execution roles exist;
- S3 bucket/prefix access and Aim endpoint reachability.

Validation is fail-closed. `trainerctl run` contains no queue or compute
environment create/update/scale API calls.

## Image and Job Definitions

Users may reference a friendly ECR tag. At experiment creation, the controller:

1. Resolves the tag to a digest.
2. Retrieves and validates the script catalog bound to that digest.
3. Stores the digest in the resolved experiment.
4. Finds an existing job definition matching resource profile, digest, worker
   protocol version, IAM roles, and logging configuration.
5. Registers a new immutable revision only when no exact match exists.

AWS Batch does not permit image override at submission, so job-definition
versioning is required. It does not alter the queue.

Old job-definition revisions may be deregistered only when no active job or
retained experiment references them and the configured retention period has
expired. Image retention is managed separately.

## Permissions

The worker’s job role is scoped to:

- read the submitted job bundle and run configs;
- write completion markers, Aim buffers, checkpoints, and Rerun artifacts under
  the current experiment prefix;
- reach the local Aim service;
- access only algorithm-specific resources explicitly declared by deployment.

The worker cannot submit/cancel Batch jobs, access Optuna SQLite, or inspect the
Aim repository. The local controller owns Batch control APIs.

## Real Smoke Test

Actual submission requires a separate, explicit yes/no authorization that lists
the exact smoke experiment/configs according to repository execution rules.

The smoke test uses:

- Aim scratch, not the main Aim repository;
- a unique `smoke/<experiment-id>` S3 prefix;
- test-labelled job names, image tags, and log streams;
- both fixed CPU and GPU profiles;
- at least two parallel jobs;
- at least two serial runs per job where the selected profile and script permit;
- an experiment with multiple independent groups and automatic `2+2+1` HPO
  batches in one CLI invocation.

Required evidence:

- compute environment, queue, job definition, image digest, and resource
  identity;
- Batch `SUCCEEDED` state and child exit codes;
- GPU job confirms JAX sees the L4;
- CloudWatch timestamps demonstrate inter-job parallelism and intra-job
  serialism;
- Aim scratch contains the exact experiment name, group/script/run/trial/seed
  hparams, mandatory episode summaries, and finalized objectives;
- Rerun artifacts contain selected complete episodes and required metadata;
- S3 completion markers contain no copied metrics;
- retries preserve logical run/trial identity.

The smoke report retains resource identifiers and assertion results, not a
duplicate metrics dataset.

## Smoke Cleanup

Cleanup runs after evidence capture and before success is declared:

1. Delete smoke Aim runs.
2. Delete the smoke S3 prefix, including configs, buffers, checkpoints, and
   Rerun files.
3. Remove temporary image tags without deleting digest versions referenced by
   retained records.
4. Remove cleanable smoke log streams.
5. Verify the main Aim repository and formal experiment prefixes contain no
   smoke data.

AWS audit records that cannot be deleted must remain clearly test-labelled and
excluded from formal experiment queries.

## Queue Cutover

After the target queues pass all smoke and drift checks:

1. Stop submissions to the superseded queues.
2. Query `SUBMITTED`, `PENDING`, `RUNNABLE`, `STARTING`, and `RUNNING` states.
3. Refuse cutover while any nonterminal job exists.
4. Disable `rtrrl-cpu-queue` and `rtrrl-gpu-queue`.
5. Wait for stable disabled state.
6. Delete those two queues and verify absence.
7. Re-run target-queue validation.

This cutover does not delete old compute environments, job definitions, IAM
roles, ECR images, S3 buckets, or Aim data. Those require separate explicit
authorization.

## Failure Handling

- Target profile drift: fail before submission.
- Aim endpoint unavailable at preflight: fail before paid compute starts.
- S3 permission failure: fail before submission.
- Capacity or transient Batch infrastructure failure: retry the same run up to
  the configured infrastructure-attempt limit.
- Persistent smoke failure: retain test-labelled evidence long enough to
  diagnose, then clean data; do not delete old queues.

## Testing

- Fake AWS API tests for every preflight field and fail-closed behavior.
- Tests proving run execution never calls queue/compute-environment mutation.
- Job-definition idempotence and digest/version retention tests.
- S3 prefix least-privilege and completion-marker tests.
- Smoke cleanup scope tests that cannot remove formal prefixes.
- Queue-decommission refusal tests for every nonterminal state.
- Actual read-only preflight and explicitly authorized real smoke test.
