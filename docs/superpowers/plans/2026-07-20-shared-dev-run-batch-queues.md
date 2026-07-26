# Simple Dev/Run AWS Batch Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the discarded Batch framework with one fixed, dry-run-first migration script and three heavy-test queue-name substitutions.

**Architecture:** Restore the repository to its pre-framework Batch code, then add a standalone boto3 script containing fixed environment, queue, and cleanup tables. The script creates or reuses resources, waits for asynchronous state changes, and deletes only idle allowlisted legacy resources; it has no smoke runner, rollback, retry, evidence, or generalized deployment layer.

**Tech Stack:** Python 3.10+, boto3, pytest, Ruff, AWS Batch.

## Global Constraints

- AWS account is `007122174918`; region is `eu-north-1`.
- Target environments are `rtrrl-cpu-c7am-ce`, `rtrrl-cpu-c7al-ce`, `rtrrl-cpu-c7ax-ce`, and `rtrrl-gpu-g6x-ce`.
- Target instance types are `c7a.medium`, `c7a.large`, `c7a.xlarge`, and `g6.xlarge`.
- Environment max-vCPU values are 16, 32, 16, and 32 respectively.
- Dev queue priority is 10; run queue priority is 100.
- Each of the eight queues binds exactly one matching environment.
- Default execution is read-only; `--execute` is required for mutation.
- Existing mismatched target resources are reported and never updated.
- Only the five nonterminal states block old-queue deletion.
- No smoke job, automatic retry, rollback, evidence collection, or artifact cleanup is implemented.
- Heavy-test changes are limited to c7am/c7ax/g6x queue names.
- Real AWS mutation requires separate user authorization after dry-run.

---

### Task 1: Remove the Discarded Batch Framework

**Files:**
- Delete: `rtrrl/infra/control-plane/src/trainer_infra/batch_admin.py`
- Delete: `rtrrl/infra/control-plane/src/trainer_infra/batch_admin_cli.py`
- Delete: `rtrrl/infra/control-plane/src/trainer_infra/batch_smoke.py`
- Delete: `rtrrl/infra/control-plane/src/trainer_infra/batch_topology.py`
- Delete: `rtrrl/infra/control-plane/tests/test_batch_admin.py`
- Delete: `rtrrl/infra/control-plane/tests/test_batch_smoke.py`
- Delete: `rtrrl/infra/control-plane/tests/test_batch_topology.py`
- Restore to `c8db009`: `infra/batch/heavy-tests/build-image.sh`
- Restore to `c8db009`: `rtrrl/infra/control-plane/pyproject.toml`
- Restore to `c8db009`: `rtrrl/infra/control-plane/src/trainer_infra/heavy_test_cli.py`
- Restore to `c8db009`: `rtrrl/infra/control-plane/src/trainer_infra/heavy_tests.py`
- Restore to `c8db009`: `rtrrl/infra/control-plane/src/trainer_infra/models.py`
- Restore to `c8db009`: `rtrrl/infra/control-plane/tests/test_heavy_tests.py`
- Restore to `c8db009`: `rtrrl/infra/control-plane/tests/test_materialize.py`
- Restore to `c8db009`: `rtrrl/infra/control-plane/tests/test_models.py`
- Restore to `c8db009`: `rtrrl/infra/control-plane/tests/test_resolve.py`
- Restore to `c8db009`: `rtrrl/infra/control-plane/tests/test_sampling.py`

**Interfaces:**
- Preserves the pre-framework heavy-test CLI and behavior.
- Removes `trainer-batch-admin` and all smoke/deployment package APIs.

- [ ] **Step 1: Reverse only the discarded implementation diff**

Run from the repository root:

```bash
git diff c8db009 -- \
  infra/batch/heavy-tests/build-image.sh \
  rtrrl/infra/control-plane/pyproject.toml \
  rtrrl/infra/control-plane/src/trainer_infra/batch_admin.py \
  rtrrl/infra/control-plane/src/trainer_infra/batch_admin_cli.py \
  rtrrl/infra/control-plane/src/trainer_infra/batch_smoke.py \
  rtrrl/infra/control-plane/src/trainer_infra/batch_topology.py \
  rtrrl/infra/control-plane/src/trainer_infra/heavy_test_cli.py \
  rtrrl/infra/control-plane/src/trainer_infra/heavy_tests.py \
  rtrrl/infra/control-plane/src/trainer_infra/models.py \
  rtrrl/infra/control-plane/tests/test_batch_admin.py \
  rtrrl/infra/control-plane/tests/test_batch_smoke.py \
  rtrrl/infra/control-plane/tests/test_batch_topology.py \
  rtrrl/infra/control-plane/tests/test_heavy_tests.py \
  rtrrl/infra/control-plane/tests/test_materialize.py \
  rtrrl/infra/control-plane/tests/test_models.py \
  rtrrl/infra/control-plane/tests/test_resolve.py \
  rtrrl/infra/control-plane/tests/test_sampling.py |
  git apply -R
```

- [ ] **Step 2: Verify framework symbols are gone**

Run:

```bash
rg "batch_admin|batch_smoke|batch_topology|trainer-batch-admin" \
  rtrrl/infra/control-plane
```

Expected: no matches.

- [ ] **Step 3: Run the restored baseline**

Run:

```bash
cd rtrrl/infra/control-plane
uv run pytest -q
uv run ruff check .
git diff --check
```

Expected: all tests pass, Ruff passes, and no whitespace errors are reported.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(infra): remove generalized Batch migration"
```

---

### Task 2: Add the One-Time Migration Script

**Files:**
- Create: `rtrrl/infra/control-plane/scripts/migrate_shared_queues.py`
- Create: `rtrrl/infra/control-plane/tests/test_migrate_shared_queues.py`
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/heavy_tests.py`

**Interfaces:**
- Produces: `migrate(*, batch: Any, sts: Any, execute: bool) -> tuple[str, ...]`.
- Produces CLI: `uv run python scripts/migrate_shared_queues.py [--execute]`.
- Preserves all heavy-test behavior except three queue-name values.

- [ ] **Step 1: Write failing fixed-resource tests**

```python
def test_fixed_resource_tables(migration):
    assert [item["name"] for item in migration.ENVIRONMENTS] == [
        "rtrrl-cpu-c7am-ce",
        "rtrrl-cpu-c7al-ce",
        "rtrrl-cpu-c7ax-ce",
        "rtrrl-gpu-g6x-ce",
    ]
    assert [item["name"] for item in migration.QUEUES] == [
        "dev-cpu-c7am-queue",
        "dev-cpu-c7al-queue",
        "dev-cpu-c7ax-queue",
        "run-cpu-c7am-queue",
        "run-cpu-c7al-queue",
        "run-cpu-c7ax-queue",
        "dev-gpu-queue",
        "run-gpu-queue",
    ]
    assert [item["priority"] for item in migration.QUEUES] == [
        10, 10, 10, 100, 100, 100, 10, 100
    ]
```

- [ ] **Step 2: Write failing behavior tests**

Use a small fake Batch client that records mutation calls and implements only
the Batch methods called by the script.

```python
def test_default_mode_only_reports_actions(migration, empty_aws):
    actions = migration.migrate(
        batch=empty_aws.batch,
        sts=empty_aws.sts,
        execute=False,
    )
    assert len([a for a in actions if a.startswith("create environment")]) == 4
    assert len([a for a in actions if a.startswith("create queue")]) == 8
    assert empty_aws.batch.mutations == []


def test_execute_creates_exact_resources(migration, empty_aws):
    migration.migrate(batch=empty_aws.batch, sts=empty_aws.sts, execute=True)
    assert empty_aws.batch.created_instance_types == [
        "c7a.medium", "c7a.large", "c7a.xlarge", "g6.xlarge"
    ]
    assert empty_aws.batch.created_queue_names == [
        "dev-cpu-c7am-queue",
        "dev-cpu-c7al-queue",
        "dev-cpu-c7ax-queue",
        "run-cpu-c7am-queue",
        "run-cpu-c7al-queue",
        "run-cpu-c7ax-queue",
        "dev-gpu-queue",
        "run-gpu-queue",
    ]


def test_active_old_queue_is_skipped(migration, matching_aws):
    matching_aws.batch.jobs[("rtrrl-cpu-queue", "RUNNING")] = ["job-1"]
    actions = migration.migrate(
        batch=matching_aws.batch,
        sts=matching_aws.sts,
        execute=True,
    )
    assert "skip active queue rtrrl-cpu-queue" in actions
    assert "rtrrl-cpu-queue" not in matching_aws.batch.deleted_queues


def test_idle_old_queue_and_unreferenced_environment_are_deleted(
    migration, matching_aws
):
    migration.migrate(
        batch=matching_aws.batch,
        sts=matching_aws.sts,
        execute=True,
    )
    assert "rtrrl-cpu2-queue" in matching_aws.batch.deleted_queues
    assert "rtrrl-cpu2-ce" in matching_aws.batch.deleted_environments
```

- [ ] **Step 3: Verify RED**

Run:

```bash
cd rtrrl/infra/control-plane
uv run pytest tests/test_migrate_shared_queues.py -v
```

Expected: FAIL because the script does not exist.

- [ ] **Step 4: Implement the fixed tables and direct migration flow**

The script contains literal tuples for:

```python
ACCOUNT_ID = "007122174918"
REGION = "eu-north-1"
ACTIVE_STATES = ("SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING")
OLD_QUEUES = (
    "rtrrl-cpu-c7am-queue",
    "rtrrl-cpu-c7al-queue",
    "rtrrl-cpu-c7ax-queue",
    "rtrrl-gpu-g6x-queue",
    "rtrrl-cpu-queue",
    "rtrrl-cpu2-queue",
    "rtrrl-gpu-queue",
)
OLD_ENVIRONMENTS = ("rtrrl-cpu-ce", "rtrrl-cpu2-ce", "rtrrl-gpu-ce")
SUBNETS = (
    "subnet-08127d1c5d4de6ac2",
    "subnet-0b8c68ea0a9784758",
    "subnet-01a2aa195678f8411",
)
SECURITY_GROUPS = ("sg-0c0ed6b927c5113dc",)
INSTANCE_ROLE = (
    "arn:aws:iam::007122174918:instance-profile/rtrrl-ecs-instance-role"
)
```

`migrate()` performs direct loops in this order:

1. compare `sts.get_caller_identity()["Account"]` and
   `batch.meta.region_name`;
2. describe each target environment by exact name;
3. compare type, state/status, instance type, min/max vCPU, subnets, security
   groups, instance role, and AMI image type; raise `ValueError` on mismatch;
4. append a readable action and, only with `execute=True`, call
   `create_compute_environment` for a missing target;
5. wait for each created environment by repeated describe calls; AWS errors
   propagate unchanged;
6. repeat the same reuse/create flow for all eight queues, comparing state,
   status, priority, and the single environment ARN;
7. list all five active states for each existing old queue; skip any nonempty
   queue, otherwise disable, wait, and delete it only in execute mode;
8. describe all remaining queues, collect referenced environment ARNs, and
   disable/delete only allowlisted old environments not in that set.

State waits use a fixed five-minute monotonic deadline. They do not catch or
retry AWS exceptions.

- [ ] **Step 5: Change only the three existing heavy-test queue values**

```python
# c7am
queue_name="dev-cpu-c7am-queue"

# c7ax
queue_name="dev-cpu-c7ax-queue"

# g6x
queue_name="dev-gpu-queue"
```

Do not add c7al, purpose arguments, STS validation, new identity formats, or
other runner behavior.

- [ ] **Step 6: Run local verification**

```bash
cd rtrrl/infra/control-plane
uv run pytest tests/test_migrate_shared_queues.py tests/test_heavy_tests.py -q
uv run pytest -q
uv run ruff check .
git diff --check
```

Expected: all tests pass, Ruff passes, and no whitespace errors are reported.

- [ ] **Step 7: Commit**

```bash
git add \
  scripts/migrate_shared_queues.py \
  src/trainer_infra/heavy_tests.py \
  tests/test_migrate_shared_queues.py
git commit -m "feat(infra): add simple Batch queue migration"
```

---

### Task 3: Read-Only AWS Handoff

**Files:** None.

**Interfaces:**
- Consumes the Task 2 script.
- Produces a reviewed dry-run action list; performs no AWS mutation.

- [ ] **Step 1: Run the migration without execute**

```bash
cd rtrrl/infra/control-plane
uv run python scripts/migrate_shared_queues.py
```

Expected: prints exact create/reuse/skip/delete actions and performs no
mutation.

- [ ] **Step 2: Verify target and old resources directly**

```bash
aws sts get-caller-identity
aws batch describe-compute-environments --region eu-north-1
aws batch describe-job-queues --region eu-north-1
```

Expected: account `007122174918`; output is used to confirm the dry-run plan.

- [ ] **Step 3: Stop before mutation**

Report the exact dry-run actions and old queues with nonterminal jobs. Request
explicit user authorization before running:

```bash
uv run python scripts/migrate_shared_queues.py --execute
```
