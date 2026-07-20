# Shared Dev/Run AWS Batch Queues Design

## Status and Scope

This specification replaces the fixed single-CPU/single-GPU queue topology in
`2026-07-20-aws-batch-migration-design.md` and the queue mapping in
`2026-07-20-batch-heavy-test-runner-design.md`. Image digest identity, immutable
job-definition revisions, observability, S3 exchange, and experiment semantics
remain unchanged.

The migration creates eight purpose-and-profile-specific queues while reusing
four validated compute environments. Development tests and formal experiments
share the same elastic capacity pools but use different queues and scheduling
priorities. Each CPU queue binds exactly one environment so a selected profile
guarantees its EC2 instance type. Fractional G6f instances are explicitly
outside this version.

## Region and Account

All resources are fixed to:

- AWS account `007122174918`;
- region `eu-north-1`;
- On-Demand managed EC2 compute environments;
- the existing VPC subnets, security group, ECS instance profile, Batch job
  role, and Batch execution role.

The deployment and runtime preflight fail closed if region, account, networking,
roles, AMI family, or any resource identity differs.

## Compute Environments

The following existing environments are retained and reused:

| Environment | Instance type | max vCPU | AMI family |
| --- | --- | ---: | --- |
| `rtrrl-cpu-c7am-ce` | `c7a.medium` | 16 | ECS AL2023 |
| `rtrrl-cpu-c7al-ce` | `c7a.large` | 32 | ECS AL2023 |
| `rtrrl-cpu-c7ax-ce` | `c7a.xlarge` | 16 | ECS AL2023 |
| `rtrrl-gpu-g6x-ce` | `g6.xlarge` | 32 | ECS AL2023 NVIDIA |

Every environment has `minvCpus=0` and normally returns to
`desiredvCpus=0`. `desiredvCpus` is dynamic runtime state and is not treated as
configuration drift. The three CPU pools expose a combined maximum of 64
vCPUs. The GPU pool exposes at most 32 vCPUs, corresponding to approximately
eight `g6.xlarge` instances with one full NVIDIA L4 each.

Multiple queues may share one compute environment. The environment is both the
instance-provisioning definition and the shared autoscaling capacity pool; it
is not merely a launch template. Capacity already occupied by one queue is not
reserved for or preempted by another queue.

## Job Queues

The migration creates exactly eight queues:

| Queue | Priority | Compute environment |
| --- | ---: | --- |
| `dev-cpu-c7am-queue` | 10 | c7am |
| `dev-cpu-c7al-queue` | 10 | c7al |
| `dev-cpu-c7ax-queue` | 10 | c7ax |
| `run-cpu-c7am-queue` | 100 | c7am |
| `run-cpu-c7al-queue` | 100 | c7al |
| `run-cpu-c7ax-queue` | 100 | c7ax |
| `dev-gpu-queue` | 10 | g6x |
| `run-gpu-queue` | 100 | g6x |

The complete compute-environment ARNs are stored and validated. Queue
separation is required because AWS Batch jobs can request vCPU and memory but
cannot select one compute environment from a multi-environment queue. Binding
one environment per queue prevents a c7am request from falling back to c7al or
c7ax and makes the resource profile an exact instance-type contract.

Queue priority applies only while jobs are waiting to be scheduled. A pending
run job is preferred over a pending dev job when they compete for a shared
environment, but AWS Batch does not preempt a running dev job and this design
does not reserve fixed run capacity.

## Resource Profiles and Routing

The facility exposes four resource profiles:

| Profile | Request |
| --- | --- |
| `c7am` | 1 vCPU, 1600 MiB, 0 GPU |
| `c7al` | 2 vCPU, 3200 MiB, 0 GPU |
| `c7ax` | 4 vCPU, 7168 MiB, 0 GPU |
| `g6x` | 4 vCPU, 12000 MiB, 1 GPU |

Users select a resource profile, not an AWS queue name.

- `trainer-heavy-test` routes c7am/c7al/c7ax to
  `dev-cpu-c7am-queue`/`dev-cpu-c7al-queue`/`dev-cpu-c7ax-queue`
  respectively, and g6x to `dev-gpu-queue`.
- Formal `trainerctl run` routes c7am/c7al/c7ax to
  `run-cpu-c7am-queue`/`run-cpu-c7al-queue`/`run-cpu-c7ax-queue`
  respectively, and g6x to `run-gpu-queue`.

Queue identity is not part of an AWS Batch job definition. A job-definition
revision remains keyed by image digest, resource profile, worker protocol,
roles, and logging configuration, and the same exact revision may be submitted
to the corresponding dev or run queue.

Runtime commands contain no queue or compute-environment create, update, scale,
disable, or delete APIs. They validate exact topology before submitting.

## G6f Exclusion

G6f is not used in this design. G6f exposes a fractional L4 and the standard
ECS/Batch integer `GPU=1` resource requirement reports no complete allocatable
GPU. Supporting it would require a separate no-GPU-resource job contract, a
custom launch template with NVIDIA as the default runtime, and full-instance
CPU/memory reservation to prevent accidental sharing. That is a separate future
design and must not be introduced as a fallback for `g6x`.

## Deployment and Cutover

The migration is an explicit deployment command, separate from normal
experiment submission:

1. Capture a read-only inventory of queues, compute environments, bindings, and
   all nonterminal job IDs.
2. Validate the four retained compute environments exactly.
3. Create each missing dev/run profile queue with its approved priority and
   single exact environment binding.
4. If a same-named queue exists, reuse it only when every field matches;
   otherwise stop without updating it.
5. Register digest-bound smoke job definitions and run the acceptance matrix.
6. Switch the heavy-test runner and formal controller to the new queue names
   only after every smoke assertion succeeds.
7. Re-inventory old resources immediately before each cleanup action.

Creation and validation are idempotent. Partial failure does not switch runtime
routing and does not delete pre-existing resources.

## Smoke Acceptance Matrix

Use a unique, test-labelled image tag resolved to a digest and submit eight
independent jobs:

| Queue | Profile |
| --- | --- |
| `dev-cpu-c7am-queue` | c7am |
| `dev-cpu-c7al-queue` | c7al |
| `dev-cpu-c7ax-queue` | c7ax |
| `run-cpu-c7am-queue` | c7am |
| `run-cpu-c7al-queue` | c7al |
| `run-cpu-c7ax-queue` | c7ax |
| `dev-gpu-queue` | g6x |
| `run-gpu-queue` | g6x |

Each job must prove:

- exact queue ARN;
- exact job-definition ARN and revision;
- exact image digest;
- exact resource requirements;
- successful container exit;
- actual EC2 instance type.

Both GPU jobs must also show a JAX CUDA device and `NVIDIA L4` in the job log.
Smoke jobs do not create Aim runs or write formal experiment S3 prefixes.

Successful acceptance additionally requires:

- all eight queues are `VALID` and `ENABLED`;
- queue priorities and single exact environment bindings match this
  specification;
- all four retained environments remain `VALID` and `ENABLED`;
- run priority is represented in AWS configuration without claiming
  preemption;
- every temporary artifact has an exact cleanup identity.

## Old Resource Cleanup

Cleanup operates on explicit resource names, never a broad prefix.

For every old queue:

1. Query `SUBMITTED`, `PENDING`, `RUNNABLE`, `STARTING`, and `RUNNING`.
2. If all five sets are empty, disable the queue, wait for stable state, delete
   it, and verify absence.
3. If any set is nonempty, keep the queue enabled so existing work can finish,
   stop all new facility submissions to it, and record the queue and job IDs as
   deferred cleanup.

After queue cleanup, consider these unneeded environments:

- `rtrrl-cpu-ce`;
- `rtrrl-cpu2-ce`;
- `rtrrl-gpu-ce`.

An environment may be disabled and deleted only when it has no nonterminal
jobs and no remaining queue reference. Any query error is treated as
not-safe-to-delete. The four retained c7am/c7al/c7ax/g6x environments are never
cleanup candidates in this migration.

Old job-definition revisions, formal ECR digests, S3 data, Aim data, IAM roles,
and networking are not deleted. Temporary smoke image tags, smoke
job-definition revisions, and exactly identified smoke log streams are removed
after evidence capture.

The controller role currently lacks `batch:TagResource`. Isolation therefore
uses deterministic `trainer-smoke-*` names rather than expanding IAM
permissions.

## Failure Handling

- Existing target resource drift: stop before paid compute.
- Partial queue creation: retain exact created-resource identities; remove only
  newly created resources with no job references.
- Smoke failure: retain diagnostic evidence, do not switch routing, and do not
  clean old resources.
- AWS read error during cleanup: skip deletion.
- Active old job: keep its queue and required environment until a later,
  separately authorized cleanup.
- Capacity shortage: keep the logical job identity and apply bounded
  infrastructure retries; do not change instance type or queue.

## Testing

Unit and fake-AWS tests cover:

- exact eight-queue topology and priorities;
- both queues sharing each approved environment;
- every queue having one exact environment binding;
- profile-to-dev/run routing;
- c7al profile support in both runner and controller;
- exact region/account/network/role/AMI/capacity validation;
- no runtime mutation APIs;
- existing-resource drift rejection;
- all five nonterminal states blocking deletion;
- exact-name cleanup scope;
- partial creation rollback boundaries;
- job-definition reuse across dev/run queues;
- smoke evidence identity and GPU L4 assertions.

Real validation uses the eight-job matrix and records queue, environment,
instance, digest, resource, log, and cleanup evidence.

## User and Operator Usage

User documentation must provide copyable examples for:

- selecting each of `c7am`, `c7al`, `c7ax`, and `g6x` in an experiment;
- running formal experiments without specifying queue names;
- running a heavy test on a dev resource profile;
- inspecting queue and job status;
- understanding that run priority is non-preemptive;
- understanding the shared 64-vCPU CPU and 32-vCPU GPU caps;
- diagnosing preflight drift and capacity errors;
- showing that G6f is unsupported in this version.

Operator documentation must provide:

- a read-only inventory command;
- the explicit deployment command;
- the eight-job smoke command;
- evidence collection;
- dry-run cleanup output;
- explicitly authorized cleanup execution;
- the deferred-cleanup report format for resources with active jobs.
