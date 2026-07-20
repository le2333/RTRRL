# AWS Batch Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strictly validate and reuse the approved c7a.medium and g6.xlarge Batch profiles, prove the complete facility with authorized real smoke tests, clean all smoke data, and safely delete only the superseded queues.

**Architecture:** AWS mutation is split from normal experiment execution. `trainerctl run` validates fixed profiles and may register image-specific job-definition revisions, but cannot create or modify queues or compute environments. Separate, narrowly scoped scripts perform smoke cleanup and allowlisted queue decommission.

**Tech Stack:** Python 3.10, boto3, AWS Batch/ECR/S3/CloudWatch, Aim 3.28, pytest.

## Global Constraints

- Target CPU queue is `rtrrl-cpu-c7am-queue` on `c7a.medium`, 1 vCPU, 1600 MiB.
- Target GPU queue is `rtrrl-gpu-g6x-queue` on `g6.xlarge`, 4 vCPU, 12000 MiB, 1 L4.
- Existing target queues are reused only after strict read-only validation.
- Normal execution has no queue/compute-environment mutation API.
- Real Batch submission requires a separate explicit yes/no authorization listing exact smoke configs.
- Smoke uses Aim scratch and a unique S3 `smoke/<experiment-id>` prefix.
- Formal-path smoke data must be removed before success is declared.
- Only `rtrrl-cpu-queue` and `rtrrl-gpu-queue` are in the initial deletion allowlist.
- Old compute environments and job definitions are not deleted by this plan.
- Commit commands require separate explicit user authorization.

---

### Task 1: Exact Resource Model and Read-Only Preflight

**Files:**
- Create: `rtrrl/infra/control-plane/config/control.example.yaml`
- Create: `rtrrl/infra/control-plane/src/trainer_infra/aws_profiles.py`
- Create: `rtrrl/infra/control-plane/tests/test_aws_profiles.py`

**Interfaces:**
- `expected_profiles() -> Mapping[str, AwsProfile]`.
- `ProfileValidator.validate_all() -> None`.

- [ ] **Step 1: Write failing exact-profile tests**

```python
def test_expected_profiles_are_fixed():
    profiles = expected_profiles()
    assert profiles["cpu"] == AwsProfile(
        queue="rtrrl-cpu-c7am-queue",
        compute_environment="rtrrl-cpu-c7am-ce",
        instance_type="c7a.medium",
        vcpus=1,
        memory_mib=1600,
        gpus=0,
        ami_family="ECS_AL2023",
    )
    assert profiles["gpu"].instance_type == "g6.xlarge"
    assert profiles["gpu"].gpus == 1
    assert profiles["gpu"].ami_family == "ECS_AL2023_NVIDIA"


@pytest.mark.parametrize(
    "mutation",
    ["instance_type", "ami_family", "queue_binding", "queue_state", "ce_status", "memory"],
)
def test_every_profile_drift_fails_closed(mutation, fake_aws):
    fake_aws.mutate(mutation)
    with pytest.raises(ProfileDriftError):
        ProfileValidator(fake_aws, expected_profiles()).validate_all()
    assert fake_aws.update_calls == []
```

- [ ] **Step 2: Verify RED**

Run:

```bash
cd rtrrl/infra/control-plane
uv run pytest tests/test_aws_profiles.py -v
```

Expected: missing `trainer_infra.aws_profiles`.

- [ ] **Step 3: Implement immutable profile definitions**

```python
@dataclass(frozen=True)
class AwsProfile:
    queue: str
    compute_environment: str
    instance_type: str
    vcpus: int
    memory_mib: int
    gpus: int
    ami_family: str


CPU = AwsProfile(
    queue="rtrrl-cpu-c7am-queue",
    compute_environment="rtrrl-cpu-c7am-ce",
    instance_type="c7a.medium",
    vcpus=1,
    memory_mib=1600,
    gpus=0,
    ami_family="ECS_AL2023",
)
GPU = AwsProfile(
    queue="rtrrl-gpu-g6x-queue",
    compute_environment="rtrrl-gpu-g6x-ce",
    instance_type="g6.xlarge",
    vcpus=4,
    memory_mib=12000,
    gpus=1,
    ami_family="ECS_AL2023_NVIDIA",
)
```

Validator methods call describe APIs only and compare every expected field.

- [ ] **Step 4: Verify GREEN**

Run targeted tests and `git diff --check`; expect zero failures.

---

### Task 2: Digest-Specific Job Definitions and Permissions

**Files:**
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/adapters/aws_batch.py`
- Test: `rtrrl/infra/control-plane/tests/test_job_definitions.py`
- Create: `rtrrl/infra/control-plane/docs/aws-permissions.md`

**Interfaces:**
- `ensure_job_definition(profile, image_digest, worker_protocol) -> str`.

- [ ] **Step 1: Write failing identity/idempotence tests**

```python
def test_exact_job_definition_is_reused(fake_batch, gpu_profile):
    adapter = AwsBatchAdapter(fake_batch)
    first = adapter.ensure_job_definition(gpu_profile, DIGEST, "v1")
    second = adapter.ensure_job_definition(gpu_profile, DIGEST, "v1")
    assert first == second
    assert fake_batch.register_calls == 1


def test_new_digest_registers_new_revision(fake_batch, gpu_profile):
    adapter = AwsBatchAdapter(fake_batch)
    old = adapter.ensure_job_definition(gpu_profile, "sha256:" + "a" * 64, "v1")
    new = adapter.ensure_job_definition(gpu_profile, "sha256:" + "b" * 64, "v1")
    assert old != new
```

- [ ] **Step 2: Verify RED**

Run the targeted test; expect missing behavior.

- [ ] **Step 3: Implement exact matching**

```python
def definition_name(profile: AwsProfile, digest: str, protocol: str) -> str:
    digest_prefix = digest.removeprefix("sha256:")[:12]
    return f"trainer-{profile.queue.removesuffix('-queue')}-{protocol}-{digest_prefix}"
```

Register a definition only after checking image digest, argv worker command,
resource requirements, job/execution roles, awslogs configuration, and protocol
environment. Document least-privilege S3-prefix and Aim-network permissions.

- [ ] **Step 4: Verify GREEN**

Run targeted tests, ruff, and `git diff --check`; expect zero failures.

---

### Task 3: Safe Smoke Harness and Evidence

**Files:**
- Create: `rtrrl/infra/control-plane/tests/aws/test_smoke_contract.py`
- Create: `rtrrl/infra/control-plane/scripts/verify_profiles.py`
- Create: `rtrrl/infra/control-plane/scripts/render_smoke_report.py`
- Create: `rtrrl/infra/control-plane/examples/experiment-smoke.yaml`

**Interfaces:**
- `verify_profiles` is read-only.
- `render_smoke_report` writes identifiers and assertions, not copied metrics.

- [ ] **Step 1: Write failing smoke-contract tests**

```python
def test_smoke_uses_only_scratch_namespaces(smoke_spec):
    assert smoke_spec.aim_repo.endswith(".aim-scratch")
    assert smoke_spec.s3_prefix.startswith("smoke/")
    assert smoke_spec.experiment.name.startswith("trainer-smoke-")


def test_report_contains_evidence_not_metric_dataset(smoke_report):
    assert smoke_report.cpu.instance_type == "c7a.medium"
    assert smoke_report.gpu.instance_type == "g6.xlarge"
    assert smoke_report.gpu.accelerator == "L4"
    assert smoke_report.metric_points is None
```

- [ ] **Step 2: Verify RED**

Run `uv run pytest tests/aws/test_smoke_contract.py -v`; expect missing harness.

- [ ] **Step 3: Implement read-only verification and report model**

```python
@dataclass(frozen=True)
class SmokeAssertion:
    name: str
    passed: bool
    evidence: str


@dataclass(frozen=True)
class SmokeReport:
    experiment_id: str
    job_ids: tuple[str, ...]
    image_digests: tuple[str, ...]
    assertions: tuple[SmokeAssertion, ...]

    @property
    def passed(self) -> bool:
        return all(assertion.passed for assertion in self.assertions)
```

The smoke YAML contains at least two independent groups and budgets that
exercise automatic `2+2+1` batches. It requires both profiles, at least two
parallel jobs, and two serial runs where resources permit.

- [ ] **Step 4: Verify GREEN locally**

Run contract tests, ruff, and `git diff --check`; expect zero failures.

- [ ] **Step 5: Obtain explicit run authorization**

Before any AWS submission, present only the exact smoke config list and required
selection evidence according to repository rules, then ask a yes/no run
question. Stop until the user answers yes.

- [ ] **Step 6: Execute authorized smoke and capture evidence**

Run the single `trainerctl run` command. Verify:

```text
Batch jobs terminal SUCCEEDED
CPU instance c7a.medium
GPU instance g6.xlarge
JAX sees NVIDIA L4
parallel job intervals overlap
run intervals inside each job do not overlap
every group completes 2+2+1 without another CLI call
Aim scratch has exact experiment and structured hparams
mandatory episode summaries and finalized objectives exist
Rerun artifacts contain complete selected episodes
S3 completion markers contain no metrics
```

Write `rtrrl/infra/control-plane/docs/smoke-report.md` with job/resource IDs and
assertion results.

---

### Task 4: Scoped Smoke Cleanup

**Files:**
- Create: `rtrrl/infra/control-plane/scripts/cleanup_smoke.py`
- Test: `rtrrl/infra/control-plane/tests/test_cleanup_smoke.py`

**Interfaces:**
- `cleanup_smoke(experiment_id: str, *, dry_run: bool = True) -> CleanupReport`.

- [ ] **Step 1: Write failing scope tests**

```python
def test_cleanup_refuses_formal_prefix(fake_services):
    with pytest.raises(UnsafeCleanupError):
        cleanup_smoke("formal-experiment", services=fake_services, dry_run=False)


def test_cleanup_only_deletes_matching_smoke_identity(fake_services):
    report = cleanup_smoke("trainer-smoke-123", services=fake_services, dry_run=False)
    assert set(report.deleted_s3_keys) == fake_services.keys_under("smoke/trainer-smoke-123/")
    assert fake_services.main_aim_runs_untouched
```

- [ ] **Step 2: Verify RED**

Run the targeted test; expect missing cleanup module.

- [ ] **Step 3: Implement allowlisted cleanup**

```python
def require_smoke_identity(experiment_id: str) -> None:
    if not experiment_id.startswith("trainer-smoke-"):
        raise UnsafeCleanupError("cleanup is restricted to trainer-smoke-* identities")
```

Delete only matching Aim scratch runs, exact S3 smoke prefix, temporary image
tags, and labelled cleanable log streams. Re-query every service and fail the
cleanup report when any formal-path test data remains.

- [ ] **Step 4: Verify GREEN and clean authorized smoke**

Run tests first. Then run dry-run, inspect its exact deletion set, run cleanup,
and verify empty smoke locations before declaring smoke success.

---

### Task 5: Allowlisted Old-Queue Cutover

**Files:**
- Create: `rtrrl/infra/control-plane/scripts/decommission_queues.py`
- Test: `rtrrl/infra/control-plane/tests/test_decommission_queues.py`

**Interfaces:**
- `decommission_queues(names: Sequence[str], *, dry_run: bool = True) -> DecommissionReport`.

- [ ] **Step 1: Write failing deletion-guard tests**

```python
@pytest.mark.parametrize("state", ["SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING"])
def test_nonterminal_job_blocks_deletion(fake_batch, state):
    fake_batch.add_job("rtrrl-cpu-queue", state)
    with pytest.raises(ActiveJobsError):
        decommission_queues(["rtrrl-cpu-queue"], batch=fake_batch, dry_run=False)


def test_only_two_old_queues_are_allowlisted(fake_batch):
    with pytest.raises(UnsafeQueueError):
        decommission_queues(["rtrrl-cpu-c7am-queue"], batch=fake_batch, dry_run=False)
```

- [ ] **Step 2: Verify RED**

Run the targeted test; expect missing script.

- [ ] **Step 3: Implement disable/wait/delete**

```python
ALLOWED_OLD_QUEUES = frozenset({"rtrrl-cpu-queue", "rtrrl-gpu-queue"})
NONTERMINAL = ("SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING")


def assert_safe_queue(queue: str, batch: BatchApi) -> None:
    if queue not in ALLOWED_OLD_QUEUES:
        raise UnsafeQueueError(queue)
    active = [job for state in NONTERMINAL for job in batch.list_jobs(queue, state)]
    if active:
        raise ActiveJobsError(queue, active)
```

For each allowlisted queue: assert safe, disable, wait for stable disabled
state, delete, and verify absence. Do not call compute-environment or
job-definition deletion APIs.

- [ ] **Step 4: Verify GREEN**

Run targeted tests and ruff; expect zero failures.

- [ ] **Step 5: Execute cutover only after acceptance gates**

Required sequence:

```text
target profile smoke passed
smoke cleanup passed
target drift recheck passed
old queues have zero nonterminal jobs
dry-run deletion set contains exactly two old queues
disable and delete
old queues absent
target profiles still valid
```

- [ ] **Step 6: Final verification**

Run:

```bash
cd rtrrl/infra/control-plane
uv run pytest -v
uv run ruff check src tests scripts
git diff --check
```

Expected: zero failures/diagnostics, no formal-path smoke data, target queues
healthy, old queues absent.

- [ ] **Step 7: Review checkpoint**

Review against
`docs/superpowers/specs/2026-07-20-aws-batch-migration-design.md`. Commit only
after explicit authorization, in task-sized commits.
