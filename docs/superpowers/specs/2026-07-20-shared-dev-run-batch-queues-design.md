# Simple Dev/Run AWS Batch Migration Design

## Scope

This revision replaces the earlier deployment, smoke-evidence, and artifact-
cleanup framework. The task is a one-time AWS Batch resource migration plus a
small routing change. It does not redesign the control plane, HPO,
observability SDK, training scripts, or their failure policies.

The migration:

1. creates or reuses four exact-instance compute environments;
2. creates or reuses eight dev/run queues;
3. updates the heavy-test profile mapping to dev queues;
4. removes idle superseded queues and their now-unreferenced environments.

No smoke jobs are submitted. Normal training and test commands never create,
update, or delete Batch infrastructure.

## Fixed AWS Resources

Account and region are fixed to `007122174918` and `eu-north-1`. Compute
environments use the existing VPC subnets, security group, ECS instance
profile, On-Demand managed EC2 provisioning, and approved ECS AL2023 image
families.

| Environment | Instance type | max vCPU | AMI family |
| --- | --- | ---: | --- |
| `rtrrl-cpu-c7am-ce` | `c7a.medium` | 16 | ECS AL2023 |
| `rtrrl-cpu-c7al-ce` | `c7a.large` | 32 | ECS AL2023 |
| `rtrrl-cpu-c7ax-ce` | `c7a.xlarge` | 16 | ECS AL2023 |
| `rtrrl-gpu-g6x-ce` | `g6.xlarge` | 32 | ECS AL2023 NVIDIA |

All environments have `minvCpus=0`. `desiredvCpus` is runtime state, not fixed
configuration. G6f is not supported.

| Queue | Priority | Environment |
| --- | ---: | --- |
| `dev-cpu-c7am-queue` | 10 | `rtrrl-cpu-c7am-ce` |
| `dev-cpu-c7al-queue` | 10 | `rtrrl-cpu-c7al-ce` |
| `dev-cpu-c7ax-queue` | 10 | `rtrrl-cpu-c7ax-ce` |
| `run-cpu-c7am-queue` | 100 | `rtrrl-cpu-c7am-ce` |
| `run-cpu-c7al-queue` | 100 | `rtrrl-cpu-c7al-ce` |
| `run-cpu-c7ax-queue` | 100 | `rtrrl-cpu-c7ax-ce` |
| `dev-gpu-queue` | 10 | `rtrrl-gpu-g6x-ce` |
| `run-gpu-queue` | 100 | `rtrrl-gpu-g6x-ce` |

Each queue binds exactly one environment. Dev and run queues share capacity;
priority affects waiting jobs but does not preempt running jobs.

## Resource Profiles and Routing

Profiles remain:

| Profile | Request |
| --- | --- |
| `c7am` | 1 vCPU, 1600 MiB, 0 GPU |
| `c7al` | 2 vCPU, 3200 MiB, 0 GPU |
| `c7ax` | 4 vCPU, 7168 MiB, 0 GPU |
| `g6x` | 4 vCPU, 12000 MiB, 1 GPU |

Users select a profile, never a queue name. `trainer-heavy-test` maps the four
profiles to the corresponding dev queues. The same mapping helper exposes run
queues for the formal Batch adapter when that separate control-plane task is
implemented; this migration does not implement that adapter.

## One-Time Migration Script

A short repository script contains the fixed resource tables and uses boto3
directly. It is not a reusable deployment framework.

The default mode is read-only and prints the actions it would take. `--execute`
performs them:

1. confirm the account and region;
2. describe each target compute environment;
3. reuse an existing exact match or create a missing environment;
4. wait for created environments to become `VALID/ENABLED`;
5. reuse an exact matching queue or create a missing queue;
6. wait for created queues to become `VALID/ENABLED`;
7. inspect the exact old-resource allowlist;
8. skip each old queue with a job in `SUBMITTED`, `PENDING`, `RUNNABLE`,
   `STARTING`, or `RUNNING`;
9. disable and delete each other old queue;
10. delete an old compute environment only after no remaining queue references
    it.

An existing target resource with different instance type, capacity, priority,
or binding is reported and left unchanged.

The script performs no automatic retries, rollback, error classification,
partial-artifact recovery, or cleanup outside the fixed allowlist. boto3/AWS
errors stop the command and remain visible. Creation and deletion require
short state waits because Batch changes are asynchronous; rerunning the script
continues from the resulting AWS state.

## Old Resource Scope

Only these exact superseded queue names may be removed:

- `rtrrl-cpu-c7am-queue`;
- `rtrrl-cpu-c7al-queue`;
- `rtrrl-cpu-c7ax-queue`;
- `rtrrl-gpu-g6x-queue`;
- `rtrrl-cpu-queue`;
- `rtrrl-cpu2-queue`;
- `rtrrl-gpu-queue`.

Only these exact obsolete environment names may be removed:

- `rtrrl-cpu-ce`;
- `rtrrl-cpu2-ce`;
- `rtrrl-gpu-ce`.

The four target environments are never cleanup candidates. Job definitions,
ECR repositories or images, CloudWatch logs, S3 data, Aim data, IAM resources,
and networking are not cleaned by this migration.

If one old queue has nonterminal work, that queue and any environment it still
references are skipped while cleanup continues for independent idle resources.

## Verification

Local tests use fake AWS clients and cover:

- the four environment and eight queue constants;
- profile-to-dev/run routing;
- c7al support in the heavy-test path;
- read-only default behavior;
- creation of missing resources;
- reuse of matching resources;
- refusal to update mismatched resources;
- all five nonterminal states blocking deletion;
- deletion only from the exact old-resource allowlist;
- preservation of referenced environments.

There is no Batch smoke-job matrix. After local tests, the operator first runs
the script without `--execute` and reviews its output. Real AWS mutation occurs
only after separate authorization. After execution, direct read-only AWS CLI
commands verify the four environments and eight queues are `VALID/ENABLED` and
have the specified instance types, capacities, priorities, and bindings.

## Removed Complexity

The final code does not retain the earlier:

- general inventory/report model;
- eight-job smoke runner or ECS/EC2 evidence chain;
- smoke job-definition, ECR tag, or log cleanup;
- cross-process definition locking;
- automatic rollback or partial-submission recovery;
- generalized retry and exception hierarchy.

These features are outside the migration requirement.
