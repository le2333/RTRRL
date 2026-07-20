# Shared Dev/Run AWS Batch Queues Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create eight exact-profile dev/run queues that share three CPU capacity pools and one full-L4 GPU pool, then validate, cut over, and safely remove only idle superseded resources.

**Architecture:** A single immutable topology module is shared by experiment resolution, the heavy-test runner, deployment, smoke evidence, and cleanup. Normal dev/run submission is read-only with respect to queues and compute environments; a separate dry-run-first admin CLI creates queues, runs the eight-job acceptance matrix, and performs exact-name cleanup.

**Tech Stack:** Python 3.10+, boto3, AWS Batch/ECS/EC2/ECR/CloudWatch, Pydantic 2, pytest, Ruff, Bash.

**Supersedes:** AWS migration Tasks 1–5 in `2026-07-20-aws-batch-migration.md` and the queue/profile portions of `2026-07-20-batch-heavy-test-runner.md`. Completed test-image isolation and one-file-per-job behavior remain in force.

## Global Constraints

- AWS account is exactly `007122174918`; region is exactly `eu-north-1`.
- Retained compute environments are exactly `rtrrl-cpu-c7am-ce`, `rtrrl-cpu-c7al-ce`, `rtrrl-cpu-c7ax-ce`, and `rtrrl-gpu-g6x-ce`.
- CPU max-vCPU values are exactly 16, 32, and 16; GPU max vCPUs are exactly 32.
- New queues are exactly `dev-cpu-c7am-queue`, `dev-cpu-c7al-queue`, `dev-cpu-c7ax-queue`, `run-cpu-c7am-queue`, `run-cpu-c7al-queue`, `run-cpu-c7ax-queue`, `dev-gpu-queue`, and `run-gpu-queue`.
- Dev queue priority is 10; run queue priority is 100.
- Every queue binds exactly one compute environment; each dev/run pair shares the matching c7am, c7al, c7ax, or g6x environment.
- Profiles are exactly `c7am`, `c7al`, `c7ax`, and `g6x`.
- Profile requests are c7am=1/1600/0, c7al=2/3200/0, c7ax=4/7168/0, and g6x=4/12000/1 for vCPU/MiB/GPU.
- `trainer-heavy-test` always uses dev queues; formal experiment submission always uses run queues.
- G6f is not supported and must never be selected as a fallback.
- Runtime submission code has no queue or compute-environment mutation APIs.
- Deployment and cleanup default to dry-run and require `--execute` for AWS mutation.
- Existing same-named resources are reused only after exact fail-closed validation; they are never silently updated.
- No active job is cancelled, moved, or preempted.
- Old resources are deleted only by exact name after a fresh five-state nonterminal-job check.
- Real AWS submission and deletion require an explicit authorization that lists the eight smoke jobs and exact cleanup candidates.
- Test artifacts use deterministic `trainer-smoke-*` names because the controller lacks `batch:TagResource`.
- User and operator documentation must contain copyable commands and explain shared capacity and non-preemptive priority.

---

### Task 1: Single-Source Topology and Read-Only Validation

**Files:**
- Create: `rtrrl/infra/control-plane/src/trainer_infra/batch_topology.py`
- Create: `rtrrl/infra/control-plane/tests/test_batch_topology.py`
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/models.py:186-188`
- Modify: `rtrrl/infra/control-plane/tests/test_models.py`
- Modify: `rtrrl/infra/control-plane/tests/test_sampling.py`
- Modify: `rtrrl/infra/control-plane/tests/test_resolve.py`
- Modify: `rtrrl/infra/control-plane/tests/test_materialize.py`

**Interfaces:**
- Produces: `ExecutionPurpose`, `ResourceProfile`, `ComputeEnvironmentSpec`, `QueueSpec`, `BatchTopology`, `AwsNetworkSettings`, `ValidatedTopology`, `ProfileDriftError`.
- Produces: `expected_topology() -> BatchTopology`.
- Produces: `queue_for(purpose: ExecutionPurpose, profile: str) -> QueueSpec`.
- Produces: `BatchTopologyValidator.validate() -> ValidatedTopology`.
- Consumed later by every Batch submission and admin component.

- [ ] **Step 1: Write failing immutable-topology tests**

```python
from trainer_infra.batch_topology import (
    ExecutionPurpose,
    expected_topology,
    queue_for,
)


def test_expected_topology_is_exact_and_shared():
    topology = expected_topology()
    assert tuple(topology.profiles) == ("c7am", "c7al", "c7ax", "g6x")
    assert topology.profiles["c7al"].resource_requirements == (
        ("VCPU", "2"),
        ("MEMORY", "3200"),
    )
    assert topology.compute_environments["c7am"].max_vcpus == 16
    assert topology.compute_environments["c7al"].max_vcpus == 32
    assert topology.compute_environments["c7ax"].max_vcpus == 16
    assert topology.compute_environments["g6x"].max_vcpus == 32
    assert topology.queues["dev-c7am"].compute_environments == ("c7am",)
    assert topology.queues["run-c7am"].compute_environments == ("c7am",)
    assert topology.queues["dev-c7al"].compute_environments == ("c7al",)
    assert topology.queues["run-c7al"].compute_environments == ("c7al",)
    assert topology.queues["dev-c7ax"].compute_environments == ("c7ax",)
    assert topology.queues["run-c7ax"].compute_environments == ("c7ax",)
    assert queue_for(ExecutionPurpose.DEV, "g6x").name == "dev-gpu-queue"
    assert queue_for(ExecutionPurpose.RUN, "g6x").name == "run-gpu-queue"


def test_run_has_higher_nonpreemptive_queue_priority():
    topology = expected_topology()
    assert topology.queues["dev-c7ax"].priority == 10
    assert topology.queues["run-c7ax"].priority == 100
```

- [ ] **Step 2: Write failing Pydantic profile-contract tests**

```python
import pytest
from pydantic import ValidationError
from trainer_infra.models import ResourcesSpec


@pytest.mark.parametrize("profile", ["c7am", "c7al", "c7ax", "g6x"])
def test_resources_accepts_only_declared_profiles(profile):
    assert ResourcesSpec(profile=profile).profile == profile


@pytest.mark.parametrize("profile", ["cpu", "gpu", "g6f", "c7a.2xlarge"])
def test_resources_rejects_legacy_or_undeclared_profiles(profile):
    with pytest.raises(ValidationError):
        ResourcesSpec(profile=profile)
```

- [ ] **Step 3: Run the targeted tests to verify RED**

Run:

```bash
cd rtrrl/infra/control-plane
uv run pytest tests/test_batch_topology.py tests/test_models.py -v
```

Expected: collection fails because `trainer_infra.batch_topology` does not exist and legacy `gpu` remains accepted.

- [ ] **Step 4: Implement the immutable topology**

```python
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

ACCOUNT_ID = "007122174918"
REGION = "eu-north-1"


class ExecutionPurpose(StrEnum):
    DEV = "dev"
    RUN = "run"


@dataclass(frozen=True)
class ComputeEnvironmentSpec:
    name: str
    instance_type: str
    max_vcpus: int
    ami_family: str


@dataclass(frozen=True)
class ResourceProfile:
    name: str
    compute_environment: str
    vcpus: int
    memory_mib: int
    gpus: int
    gpu_model: str | None = None

    @property
    def resource_requirements(self) -> tuple[tuple[str, str], ...]:
        values = [("VCPU", str(self.vcpus)), ("MEMORY", str(self.memory_mib))]
        if self.gpus:
            values.append(("GPU", str(self.gpus)))
        return tuple(values)


@dataclass(frozen=True)
class QueueSpec:
    name: str
    priority: int
    compute_environments: tuple[str, ...]
    purpose: ExecutionPurpose


@dataclass(frozen=True)
class BatchTopology:
    compute_environments: Mapping[str, ComputeEnvironmentSpec]
    profiles: Mapping[str, ResourceProfile]
    queues: Mapping[str, QueueSpec]


_COMPUTE_ENVIRONMENTS = MappingProxyType({
    "c7am": ComputeEnvironmentSpec("rtrrl-cpu-c7am-ce", "c7a.medium", 16, "ECS_AL2023"),
    "c7al": ComputeEnvironmentSpec("rtrrl-cpu-c7al-ce", "c7a.large", 32, "ECS_AL2023"),
    "c7ax": ComputeEnvironmentSpec("rtrrl-cpu-c7ax-ce", "c7a.xlarge", 16, "ECS_AL2023"),
    "g6x": ComputeEnvironmentSpec("rtrrl-gpu-g6x-ce", "g6.xlarge", 32, "ECS_AL2023_NVIDIA"),
})
_PROFILES = MappingProxyType({
    "c7am": ResourceProfile("c7am", "c7am", 1, 1600, 0),
    "c7al": ResourceProfile("c7al", "c7al", 2, 3200, 0),
    "c7ax": ResourceProfile("c7ax", "c7ax", 4, 7168, 0),
    "g6x": ResourceProfile("g6x", "g6x", 4, 12000, 1, "NVIDIA L4"),
})
_QUEUES = MappingProxyType({
    "dev-c7am": QueueSpec("dev-cpu-c7am-queue", 10, ("c7am",), ExecutionPurpose.DEV),
    "dev-c7al": QueueSpec("dev-cpu-c7al-queue", 10, ("c7al",), ExecutionPurpose.DEV),
    "dev-c7ax": QueueSpec("dev-cpu-c7ax-queue", 10, ("c7ax",), ExecutionPurpose.DEV),
    "run-c7am": QueueSpec("run-cpu-c7am-queue", 100, ("c7am",), ExecutionPurpose.RUN),
    "run-c7al": QueueSpec("run-cpu-c7al-queue", 100, ("c7al",), ExecutionPurpose.RUN),
    "run-c7ax": QueueSpec("run-cpu-c7ax-queue", 100, ("c7ax",), ExecutionPurpose.RUN),
    "dev-g6x": QueueSpec("dev-gpu-queue", 10, ("g6x",), ExecutionPurpose.DEV),
    "run-g6x": QueueSpec("run-gpu-queue", 100, ("g6x",), ExecutionPurpose.RUN),
})
_TOPOLOGY = BatchTopology(
    compute_environments=_COMPUTE_ENVIRONMENTS,
    profiles=_PROFILES,
    queues=_QUEUES,
)


def expected_topology() -> BatchTopology:
    return _TOPOLOGY


def queue_for(purpose: ExecutionPurpose, profile: str) -> QueueSpec:
    if profile not in _PROFILES:
        raise ValueError(f"unknown Batch resource profile: {profile!r}")
    return _QUEUES[f"{purpose.value}-{profile}"]
```

`queue_for()` never accepts a queue name from user input.

- [ ] **Step 5: Make formal resource configuration use exact profile names**

```python
from typing import Literal


class ResourcesSpec(ContractModel):
    profile: Literal["c7am", "c7al", "c7ax", "g6x"]
```

Replace every test fixture containing `ResourcesSpec(profile="gpu")` or
`"resources": {"profile": "gpu"}` with `g6x`. Add one resolved CPU fixture using
`c7al` so materialization preserves all four exact names.

- [ ] **Step 6: Write and implement fail-closed topology validation**

```python
class ProfileDriftError(RuntimeError):
    """Raised when deployed Batch topology differs from the exact contract."""


@dataclass(frozen=True)
class AwsNetworkSettings:
    subnets: tuple[str, ...]
    security_group_ids: tuple[str, ...]
    instance_role: str


@dataclass(frozen=True)
class ValidatedTopology:
    compute_environment_arns: Mapping[str, str]
    queue_arns: Mapping[str, str]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("instanceTypes", ["c7a.2xlarge"]),
        ("maxvCpus", 64),
        ("state", "DISABLED"),
        ("status", "INVALID"),
        ("imageType", "ECS_AL2"),
    ],
)
def test_compute_environment_drift_fails_without_mutation(
    fake_services, field, value
):
    fake_services.batch.mutate_compute_environment("c7al", field, value)
    with pytest.raises(ProfileDriftError):
        BatchTopologyValidator(
            fake_services.batch,
            fake_services.sts,
            expected_topology(),
        ).validate()
    assert fake_services.batch.mutation_calls == []


def test_cpu_queue_rejects_wrong_single_environment(fake_services):
    fake_services.batch.queues[
        "run-cpu-c7al-queue"
    ]["computeEnvironmentOrder"] = [
        fake_services.batch.binding("c7ax", order=1)
    ]
    with pytest.raises(ProfileDriftError, match="computeEnvironmentOrder"):
        BatchTopologyValidator(
            fake_services.batch,
            fake_services.sts,
            expected_topology(),
        ).validate()
```

`BatchTopologyValidator` performs only `describe_compute_environments` and
`describe_job_queues`. It compares account/region, managed EC2 type,
state/status, EC2 provisioning, instance type, max vCPUs, AMI family, network,
instance role, queue priority, and the single exact environment ARN. It ignores
dynamic `desiredvCpus`. Its constructor is:

```python
class BatchTopologyValidator:
    def __init__(
        self,
        batch: Any,
        sts: Any,
        topology: BatchTopology = _TOPOLOGY,
        network: AwsNetworkSettings = DEFAULT_AWS_NETWORK_SETTINGS,
    ) -> None:
        self._batch = batch
        self._sts = sts
        self._topology = topology
        self._network = network

    def validate(self) -> ValidatedTopology:
        account = self._sts.get_caller_identity()["Account"]
        if account != ACCOUNT_ID or self._batch.meta.region_name != REGION:
            raise ProfileDriftError(
                f"expected {ACCOUNT_ID}/{REGION}, got "
                f"{account}/{self._batch.meta.region_name}"
            )
        environment_arns: dict[str, str] = {}
        for key, expected in self._topology.compute_environments.items():
            response = self._batch.describe_compute_environments(
                computeEnvironments=[expected.name]
            )
            actual = require_one(
                response["computeEnvironments"],
                kind="compute environment",
                name=expected.name,
            )
            validate_compute_environment(actual, expected, self._network)
            environment_arns[key] = actual["computeEnvironmentArn"]
        queue_arns: dict[str, str] = {}
        for key, expected in self._topology.queues.items():
            response = self._batch.describe_job_queues(
                jobQueues=[expected.name]
            )
            actual = require_one(
                response["jobQueues"],
                kind="job queue",
                name=expected.name,
            )
            validate_queue(actual, expected, environment_arns)
            queue_arns[key] = actual["jobQueueArn"]
        return ValidatedTopology(
            compute_environment_arns=MappingProxyType(environment_arns),
            queue_arns=MappingProxyType(queue_arns),
        )
```

Implement `require_one`, `validate_compute_environment`, and `validate_queue`
as pure comparison functions. Each raises `ProfileDriftError` with the exact
field path and expected/actual values; none receives an AWS client, which makes
mutation impossible inside validation.

- [ ] **Step 7: Verify and commit**

Run:

```bash
uv run pytest tests/test_batch_topology.py tests/test_models.py \
  tests/test_sampling.py tests/test_resolve.py tests/test_materialize.py -v
uv run ruff check src tests
git diff --check
```

Expected: all selected tests pass, Ruff prints `All checks passed!`, and
`git diff --check` prints nothing.

Commit:

```bash
git add rtrrl/infra/control-plane/src/trainer_infra/batch_topology.py \
  rtrrl/infra/control-plane/src/trainer_infra/models.py \
  rtrrl/infra/control-plane/tests
git commit -m "refactor(infra): single-source Batch topology"
```

---

### Task 2: Route Heavy Tests to Dev Queues and Add c7al

**Files:**
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/heavy_tests.py`
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/heavy_test_cli.py`
- Modify: `rtrrl/infra/control-plane/tests/test_heavy_tests.py`
- Modify: `infra/batch/heavy-tests/build-image.sh`

**Interfaces:**
- Consumes: `expected_topology()`, `queue_for(ExecutionPurpose.DEV, profile)`.
- Produces: `HeavyTestRunner(batch: Any, logs: Any, sts: Any, *, sleep: Callable[[float], None] = time.sleep, repository_root: Path | None = None)`.
- Produces: `HeavyTestRunner.submit(*, profile: str, image: str, tests: Sequence[str], purpose: ExecutionPurpose = ExecutionPurpose.DEV, name_prefix: str = "trainer-heavy-test") -> tuple[SubmittedTestJob, ...]`.
- Preserves: one exact pytest file per Batch job and digest-bound evidence.
- Adds: c7al image-builder and submission support.

- [ ] **Step 1: Write failing dev-routing tests**

```python
@pytest.mark.parametrize(
    ("profile", "queue"),
    [
        ("c7am", "dev-cpu-c7am-queue"),
        ("c7al", "dev-cpu-c7al-queue"),
        ("c7ax", "dev-cpu-c7ax-queue"),
        ("g6x", "dev-gpu-queue"),
    ],
)
def test_heavy_tests_route_only_to_dev_queues(runner, profile, queue):
    submitted = runner.submit(
        profile=profile,
        image=IMAGE,
        tests=["memo/tests/test_logging_compat.py"],
    )
    assert submitted[0].purpose == "dev"
    assert submitted[0].queue_name == queue


def test_heavy_test_cli_has_no_run_queue_switch(parser):
    with pytest.raises(SystemExit):
        parser.parse_args([
            "submit", "--purpose", "run", "--profile", "c7ax",
            "--image", IMAGE, "memo/tests/test_logging_compat.py",
        ])
```

- [ ] **Step 2: Write failing shared-queue identity tests**

```python
def test_wait_rejects_cpu_queue_bound_to_wrong_profile(fake_aws, runner):
    fake_aws.queue("dev-cpu-c7al-queue")["computeEnvironmentOrder"] = [
        fake_aws.binding("c7ax", order=1),
    ]
    with pytest.raises(ProfileDriftError, match="computeEnvironmentOrder"):
        runner.submit(
            profile="c7al",
            image=IMAGE,
            tests=["memo/tests/test_logging_compat.py"],
        )
```

- [ ] **Step 3: Run targeted tests to verify RED**

Run:

```bash
cd rtrrl/infra/control-plane
uv run pytest tests/test_heavy_tests.py -v
```

Expected: failures show old per-profile queue names and missing c7al.

- [ ] **Step 4: Remove duplicated topology and add purpose to job identity**

Delete `HeavyTestProfile`, `TEST_PROFILES`, and hard-coded c7am/c7ax/g6x queue
maps from `heavy_tests.py`. Import the shared `ResourceProfile` values.
Add the STS client to `HeavyTestRunner`; every submit and wait validates account,
region, all retained environments, and all eight queue bindings through the
shared validator before any paid work or evidence acceptance.

```python
@dataclass(frozen=True)
class SubmittedTestJob:
    job_id: str
    test_file: str
    purpose: ExecutionPurpose
    profile: str
    queue_name: str
    queue_arn: str
    image: str
    job_definition_arn: str
    job_definition_revision: int
    resource_requirements: tuple[ResourceRequirement, ...]
    command_text: str
```

Job names include purpose, while job-definition identity is purpose-neutral and
binds kind/profile/digest/resources. This lets the dev/run pair for one profile
reuse the same exact digest-bound definition ARN/revision; `wait()` verifies
purpose from the job name together with the exact queue:

```python
def job_name(purpose: ExecutionPurpose, profile: str, stem: str, suffix: str) -> str:
    return bounded_name(
        prefix=f"trainer-heavy-test-{purpose.value}-{profile}-",
        stem=stem,
        suffix=suffix,
        maximum=128,
    )


def definition_name(kind: str, profile: str, digest: str) -> str:
    return f"trainer-{kind}-{profile}-{digest}"
```

`wait()` parses purpose and profile, then validates the exact queue, complete
queue topology, job-definition revision, image digest, and resource
requirements. Existing pre-migration job IDs are evidence-only and need not be
accepted by the new parser.

- [ ] **Step 5: Keep CLI dev-only and add c7al builder support**

`heavy_test_cli.py` constructs Batch, Logs, and STS clients in `eu-north-1`,
passes `ExecutionPurpose.DEV` internally, and exposes only the four profile
choices. `build-image.sh` accepts `c7al` as a CPU profile and uses the same
pinned CPU base-image path as c7am/c7ax.

```bash
case "${PROFILE}" in
  c7am|c7al|c7ax) PLATFORM="cpu" ;;
  g6x) PLATFORM="gpu" ;;
  *) fail "profile must be one of: c7am, c7al, c7ax, g6x" ;;
esac
```

- [ ] **Step 6: Verify and commit**

Run:

```bash
cd rtrrl/infra/control-plane
uv run pytest tests/test_heavy_tests.py tests/test_batch_topology.py -v
uv run ruff check src tests
bash -n ../../../infra/batch/heavy-tests/build-image.sh
git diff --check
```

Expected: zero failures and diagnostics.

Commit:

```bash
git add rtrrl/infra/control-plane/src/trainer_infra/heavy_tests.py \
  rtrrl/infra/control-plane/src/trainer_infra/heavy_test_cli.py \
  rtrrl/infra/control-plane/tests/test_heavy_tests.py \
  infra/batch/heavy-tests/build-image.sh
git commit -m "feat(infra): route heavy tests through dev queues"
```

---

### Task 3: Dry-Run-First Queue Deployment

**Files:**
- Create: `rtrrl/infra/control-plane/src/trainer_infra/batch_admin.py`
- Create: `rtrrl/infra/control-plane/src/trainer_infra/batch_admin_cli.py`
- Create: `rtrrl/infra/control-plane/tests/test_batch_admin.py`
- Modify: `rtrrl/infra/control-plane/pyproject.toml`

**Interfaces:**
- Produces: `BatchAdminServices(batch: Any, sts: Any)`.
- Produces: `inventory(services: BatchAdminServices) -> BatchInventory`.
- Produces: `deploy_queues(services: BatchAdminServices, *, execute: bool = False) -> DeploymentReport`.
- Produces CLI: `trainer-batch-admin inventory`.
- Produces CLI: `trainer-batch-admin deploy [--execute]`.
- Does not expose compute-environment creation or update.

- [ ] **Step 1: Write failing deployment tests**

```python
def test_deploy_dry_run_has_no_mutation(fake_services):
    report = deploy_queues(fake_services, execute=False)
    assert report.create_queues == (
        "dev-cpu-c7am-queue",
        "dev-cpu-c7al-queue",
        "dev-cpu-c7ax-queue",
        "run-cpu-c7am-queue",
        "run-cpu-c7al-queue",
        "run-cpu-c7ax-queue",
        "dev-gpu-queue",
        "run-gpu-queue",
    )
    assert fake_services.batch.mutation_calls == []


def test_deploy_creates_exact_shared_bindings(fake_services):
    report = deploy_queues(fake_services, execute=True)
    assert report.created == (
        "dev-cpu-c7am-queue",
        "dev-cpu-c7al-queue",
        "dev-cpu-c7ax-queue",
        "run-cpu-c7am-queue",
        "run-cpu-c7al-queue",
        "run-cpu-c7ax-queue",
        "dev-gpu-queue",
        "run-gpu-queue",
    )
    assert fake_services.batch.create_calls["run-cpu-c7al-queue"] == {
        "jobQueueName": "run-cpu-c7al-queue",
        "state": "ENABLED",
        "priority": 100,
        "computeEnvironmentOrder": [
            fake_services.batch.binding("c7al", order=1),
        ],
    }


def test_existing_drift_is_never_updated(fake_services):
    fake_services.batch.add_queue("dev-cpu-c7am-queue", priority=99)
    with pytest.raises(ProfileDriftError, match="priority"):
        deploy_queues(fake_services, execute=True)
    assert fake_services.batch.update_calls == []


def test_partial_creation_removes_only_new_unreferenced_queues(fake_services):
    fake_services.batch.fail_create("run-gpu-queue")
    with pytest.raises(DeploymentError) as raised:
        deploy_queues(fake_services, execute=True)
    assert set(raised.value.report.rolled_back) == {
        "dev-cpu-c7am-queue",
        "dev-cpu-c7al-queue",
        "dev-cpu-c7ax-queue",
        "run-cpu-c7am-queue",
        "run-cpu-c7al-queue",
        "run-cpu-c7ax-queue",
        "dev-gpu-queue",
    }
    assert fake_services.batch.preexisting_resources_untouched
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
cd rtrrl/infra/control-plane
uv run pytest tests/test_batch_admin.py -v
```

Expected: missing `trainer_infra.batch_admin`.

- [ ] **Step 3: Implement immutable inventory and deployment reports**

```python
@dataclass(frozen=True)
class BatchInventory:
    captured_at: datetime
    queues: tuple[InventoryQueue, ...]
    compute_environments: tuple[InventoryComputeEnvironment, ...]
    nonterminal_jobs: tuple[InventoryJob, ...]


@dataclass(frozen=True)
class BatchAdminServices:
    batch: Any
    sts: Any


@dataclass(frozen=True)
class DeploymentReport:
    execute: bool
    reused: tuple[str, ...]
    created: tuple[str, ...]
    create_queues: tuple[str, ...]
    rolled_back: tuple[str, ...]
    topology_valid: bool
```

`inventory()` paginates every describe/list operation and records all five
nonterminal states. `deploy_queues()` first validates all four retained
compute environments through `BatchTopologyValidator(services.batch,
services.sts)`, then creates only missing queues. After each create, it
waits with a bounded timeout and re-runs the read-only validator. If a later
create fails before smoke submission, it disables and deletes only queues
created by that invocation after rechecking that they have no jobs; it never
alters pre-existing resources.

- [ ] **Step 4: Implement JSON CLI with explicit mutation**

```python
deploy = subparsers.add_parser("deploy")
deploy.add_argument("--execute", action="store_true")
```

`trainer-batch-admin deploy` prints planned creates and performs no mutation.
`--execute` performs the exact create calls. Both commands fix region to
`eu-north-1` and verify caller account before any Batch API.

Add:

```toml
trainer-batch-admin = "trainer_infra.batch_admin_cli:main"
```

- [ ] **Step 5: Verify and commit**

Run:

```bash
uv run pytest tests/test_batch_admin.py tests/test_batch_topology.py -v
uv run ruff check src tests
git diff --check
```

Commit:

```bash
git add rtrrl/infra/control-plane/src/trainer_infra/batch_admin.py \
  rtrrl/infra/control-plane/src/trainer_infra/batch_admin_cli.py \
  rtrrl/infra/control-plane/tests/test_batch_admin.py \
  rtrrl/infra/control-plane/pyproject.toml
git commit -m "feat(infra): deploy shared Batch queues safely"
```

---

### Task 4: Eight-Job Smoke Matrix and Instance Evidence

**Files:**
- Create: `rtrrl/infra/control-plane/src/trainer_infra/batch_smoke.py`
- Create: `rtrrl/infra/control-plane/tests/test_batch_smoke.py`
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/batch_admin_cli.py`
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/heavy_tests.py`

**Interfaces:**
- Produces: `SmokeServices(batch: Any, logs: Any, sts: Any, ecs: Any, ec2: Any)`.
- Produces: `SmokeCase`, `SmokeEvidence`, `SmokeReport`.
- Produces: `smoke_plan(cpu_image: str, gpu_image: str) -> tuple[SmokeCase, ...]`.
- Produces: `run_smoke(services: SmokeServices, *, cpu_image: str, gpu_image: str, execute: bool = False) -> SmokeReport`.
- Produces CLI: `trainer-batch-admin smoke --cpu-image IMAGE@DIGEST --gpu-image IMAGE@DIGEST [--execute]`.

- [ ] **Step 1: Write failing exact-matrix tests**

```python
def test_smoke_matrix_is_exact():
    cases = smoke_plan(CPU_IMAGE, GPU_IMAGE)
    assert [(case.purpose.value, case.profile) for case in cases] == [
        ("dev", "c7am"),
        ("dev", "c7al"),
        ("dev", "c7ax"),
        ("run", "c7am"),
        ("run", "c7al"),
        ("run", "c7ax"),
        ("dev", "g6x"),
        ("run", "g6x"),
    ]
    assert len({case.smoke_name for case in cases}) == 8
    assert all(case.smoke_name.startswith("trainer-smoke-") for case in cases)
```

- [ ] **Step 2: Write failing instance/evidence tests**

```python
@pytest.mark.parametrize(
    ("profile", "instance_type"),
    [
        ("c7am", "c7a.medium"),
        ("c7al", "c7a.large"),
        ("c7ax", "c7a.xlarge"),
        ("g6x", "g6.xlarge"),
    ],
)
def test_smoke_requires_actual_instance_type(fake_services, profile, instance_type):
    fake_services.set_job_instance(profile, instance_type)
    evidence = collect_smoke_evidence(fake_services, fake_services.job_id(profile))
    assert evidence.instance_type == instance_type


def test_wrong_instance_or_missing_l4_fails_report(fake_services):
    fake_services.set_job_instance("c7al", "c7a.xlarge")
    fake_services.set_gpu_log("run", [])
    report = collect_report(fake_services)
    assert not report.passed
    assert "instance type" in report.failure_text
    assert "NVIDIA L4" in report.failure_text


def test_dev_and_run_reuse_same_digest_bound_definition(fake_services):
    report = run_smoke(
        fake_services,
        cpu_image=CPU_IMAGE,
        gpu_image=GPU_IMAGE,
        execute=True,
    )
    by_case = {
        (item.purpose.value, item.profile): item
        for item in report.cases
    }
    for profile in ("c7am", "c7al", "c7ax", "g6x"):
        assert (
            by_case[("dev", profile)].job_definition_arn
            == by_case[("run", profile)].job_definition_arn
        )
```

- [ ] **Step 3: Run tests to verify RED**

Run:

```bash
cd rtrrl/infra/control-plane
uv run pytest tests/test_batch_smoke.py -v
```

Expected: missing `trainer_infra.batch_smoke`.

- [ ] **Step 4: Implement dry-run plan and internal run-purpose submission**

Extend `HeavyTestRunner.submit()` with an internal `purpose` argument. The
public heavy-test CLI always passes DEV. `batch_smoke.py` passes DEV or RUN from
the fixed matrix; it does not accept arbitrary queue names.

```python
@dataclass(frozen=True)
class SmokeServices:
    batch: Any
    logs: Any
    sts: Any
    ecs: Any
    ec2: Any


@dataclass(frozen=True)
class SmokeReport:
    smoke_id: str
    account_id: str
    region: str
    captured_at: datetime
    queue_deployment_observed_at: datetime
    passed: bool
    cases: tuple[SmokeEvidence, ...]
    job_definition_arns: tuple[str, ...]
    temporary_image_tags: tuple[tuple[str, str], ...]
    log_stream_names: tuple[str, ...]
```

The execute path atomically writes
`.trainer/smoke/trainer-smoke-shared-queues.json`; the dry-run path writes
nothing.

Each smoke command runs a lightweight pytest file and emits values built from
the fixed case:

```python
print(f"trainer_smoke_profile={case.profile}")
print(f"trainer_smoke_purpose={case.purpose.value}")
print(f"JAX devices: {jax.devices()}")
```

The GPU command additionally runs
`nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`, whose accepted
output must contain `NVIDIA L4`.

The GPU probe is required only for g6x but both g6x cases must contain it.

- [ ] **Step 5: Resolve actual EC2 instance type without container IMDS**

For each terminal job:

1. Read `containerInstanceArn` from the successful attempt.
2. Read each target compute environment's `ecsClusterArn`.
3. Call `ecs.describe_container_instances` in those exact clusters until the
   container instance is found.
4. Read `ec2InstanceId`.
5. Call `ec2.describe_instances` and validate `InstanceType`.

Any missing ARN, ambiguous match, wrong instance type, wrong queue, wrong
definition, wrong digest, resource mismatch, nonzero exit, missing RSS, missing
JAX CUDA device, or missing L4 makes `SmokeReport.passed` false.

- [ ] **Step 6: Add dry-run-first smoke CLI**

Without `--execute`, print the eight cases, exact queues, image digests, and
resources without registering definitions or submitting jobs. With
`--execute`, submit all eight, wait for all terminal states, collect evidence,
and return exit 1 unless every assertion passes.

- [ ] **Step 7: Verify and commit**

Run:

```bash
uv run pytest tests/test_batch_smoke.py tests/test_heavy_tests.py -v
uv run ruff check src tests
git diff --check
```

Commit:

```bash
git add rtrrl/infra/control-plane/src/trainer_infra/batch_smoke.py \
  rtrrl/infra/control-plane/src/trainer_infra/batch_admin_cli.py \
  rtrrl/infra/control-plane/src/trainer_infra/heavy_tests.py \
  rtrrl/infra/control-plane/tests/test_batch_smoke.py \
  rtrrl/infra/control-plane/tests/test_heavy_tests.py
git commit -m "feat(infra): verify shared queues with smoke matrix"
```

---

### Task 5: Exact Old-Queue and Compute-Environment Cleanup

**Files:**
- Create: `rtrrl/infra/control-plane/src/trainer_infra/batch_cleanup.py`
- Create: `rtrrl/infra/control-plane/tests/test_batch_cleanup.py`
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/batch_admin_cli.py`

**Interfaces:**
- Produces: `CleanupServices(batch: Any, sts: Any, ecr: Any, logs: Any)`.
- Produces: `cleanup_plan(services: CleanupServices, smoke_report: SmokeReport) -> CleanupPlan`.
- Produces: `cleanup_old_resources(services: CleanupServices, *, smoke_report: SmokeReport, execute: bool = False) -> CleanupReport`.
- Produces CLI: `trainer-batch-admin cleanup [--execute]`.

- [ ] **Step 1: Write failing exact-scope tests**

```python
OLD_QUEUES = frozenset({
    "rtrrl-cpu-c7am-queue",
    "rtrrl-cpu-c7al-queue",
    "rtrrl-cpu-c7ax-queue",
    "rtrrl-gpu-g6x-queue",
    "rtrrl-cpu-queue",
    "rtrrl-cpu2-queue",
    "rtrrl-gpu-queue",
})
UNNEEDED_ENVIRONMENTS = frozenset({
    "rtrrl-cpu-ce",
    "rtrrl-cpu2-ce",
    "rtrrl-gpu-ce",
})


def test_cleanup_dry_run_uses_only_exact_allowlists(fake_services, smoke_report):
    report = cleanup_old_resources(
        fake_services,
        smoke_report=smoke_report,
        execute=False,
    )
    assert set(report.queue_candidates) == OLD_QUEUES
    assert set(report.environment_candidates) == UNNEEDED_ENVIRONMENTS
    assert fake_services.batch.mutation_calls == []
    assert fake_services.ecr.mutation_calls == []
    assert fake_services.logs.mutation_calls == []
```

- [ ] **Step 2: Write all-state protection tests**

```python
@pytest.mark.parametrize(
    "state",
    ["SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING"],
)
def test_active_old_queue_is_deferred_not_disabled(
    fake_services, smoke_report, state
):
    fake_services.batch.add_job(
        "rtrrl-cpu2-queue", state, job_id=f"job-{state}"
    )
    report = cleanup_old_resources(
        fake_services,
        smoke_report=smoke_report,
        execute=True,
    )
    assert report.deferred_queues["rtrrl-cpu2-queue"] == (f"job-{state}",)
    assert "rtrrl-cpu2-queue" not in fake_services.batch.disable_calls


def test_referenced_or_active_environment_is_deferred(
    fake_services, smoke_report
):
    fake_services.batch.reference_environment("custom-queue", "rtrrl-cpu2-ce")
    report = cleanup_old_resources(
        fake_services,
        smoke_report=smoke_report,
        execute=True,
    )
    assert report.deferred_environments["rtrrl-cpu2-ce"] == (
        "referenced by custom-queue",
    )
    assert (
        "rtrrl-cpu2-ce"
        not in fake_services.batch.delete_environment_calls
    )


def test_smoke_cleanup_uses_only_reported_exact_identities(fake_services, smoke_report):
    report = cleanup_old_resources(
        fake_services,
        smoke_report=smoke_report,
        execute=True,
    )
    assert set(report.deregistered_definitions) == set(
        smoke_report.job_definition_arns
    )
    assert set(report.deleted_image_tags) == set(smoke_report.temporary_image_tags)
    assert set(report.deleted_log_streams) == set(smoke_report.log_stream_names)
    assert fake_services.formal_digests_untouched
    assert fake_services.unrelated_log_streams_untouched
```

- [ ] **Step 3: Run tests to verify RED**

Run:

```bash
cd rtrrl/infra/control-plane
uv run pytest tests/test_batch_cleanup.py -v
```

Expected: missing `trainer_infra.batch_cleanup`.

- [ ] **Step 4: Implement fresh-query queue cleanup**

For each exact old queue independently:

1. Describe it; absent is recorded as already clean.
2. Query all five nonterminal states with pagination.
3. If any jobs exist, record IDs and make no mutation.
4. Otherwise disable, poll boundedly for stable disabled state, delete, and
   verify absence.

An API error is recorded as deferred and prevents mutation for that resource.
The implementation never cancels jobs.

- [ ] **Step 5: Implement environment cleanup after queue cleanup**

For each exact unneeded environment:

1. Re-describe every remaining queue and collect references.
2. Re-query nonterminal jobs on every referencing queue.
3. Defer on any reference, active job, or read failure.
4. Otherwise disable the environment, wait until disabled/valid with
   `desiredvCpus=0`, delete, and verify absence.

Hard-code a retained-environment denylist and raise `UnsafeCleanupError` before
any mutation if a caller or report ever includes c7am, c7al, c7ax, or g6x.

- [ ] **Step 6: Implement exact smoke-artifact cleanup**

```python
@dataclass(frozen=True)
class CleanupServices:
    batch: Any
    sts: Any
    ecr: Any
    logs: Any
```

Read exact job-definition ARNs, temporary ECR repository/tag pairs, and
CloudWatch log-stream names from the passing smoke report. Before each
mutation, verify that:

- the job definition has a `trainer-smoke-` name and exact revision;
- the image tag has a `trainer-smoke-` name while its digest remains untouched;
- the log stream is listed by the report under `/aws/batch/job`;
- no artifact is referenced by a formal retained record.

Dry-run returns these exact identities. Execute deregisters only those
definition revisions, removes only those ECR tags, and deletes only those log
streams. It performs no Aim or S3 calls because smoke created neither.

- [ ] **Step 7: Add cleanup CLI and acceptance gate**

`trainer-batch-admin cleanup` is dry-run. `--execute` first validates all eight
new queues and all four retained environments, and requires a previously
written passing smoke-report path:

```bash
SMOKE_REPORT=".trainer/smoke/trainer-smoke-shared-queues.json"
trainer-batch-admin cleanup \
  --smoke-report "${SMOKE_REPORT}" \
  --execute
```

Reject a missing report, `passed=false`, wrong account/region, or a report older
than the current queue deployment.

- [ ] **Step 8: Verify and commit**

Run:

```bash
uv run pytest tests/test_batch_cleanup.py tests/test_batch_admin.py -v
uv run ruff check src tests
git diff --check
```

Commit:

```bash
git add rtrrl/infra/control-plane/src/trainer_infra/batch_cleanup.py \
  rtrrl/infra/control-plane/src/trainer_infra/batch_admin_cli.py \
  rtrrl/infra/control-plane/tests/test_batch_cleanup.py
git commit -m "feat(infra): clean idle legacy Batch resources"
```

---

### Task 6: User and Operator Documentation

**Files:**
- Create: `rtrrl/infra/control-plane/docs/batch-usage.md`
- Create: `rtrrl/infra/control-plane/examples/resources-cpu.yaml`
- Create: `rtrrl/infra/control-plane/examples/resources-gpu.yaml`
- Modify: `docs/superpowers/specs/2026-07-20-aws-batch-migration-design.md`
- Modify: `docs/superpowers/specs/2026-07-20-batch-heavy-test-runner-design.md`
- Test: `rtrrl/infra/control-plane/tests/test_batch_docs.py`

**Interfaces:**
- Documents all four profiles and both automatic routing purposes.
- Documents inventory, deploy, smoke, cleanup, status inspection, and deferred cleanup.
- Makes the new design the explicit source of truth from superseded specs.

- [ ] **Step 1: Write failing documentation-contract tests**

```python
def test_batch_usage_documents_every_profile_and_queue():
    text = Path("docs/batch-usage.md").read_text()
    for value in (
        "c7am", "c7al", "c7ax", "g6x",
        "dev-cpu-c7am-queue", "dev-cpu-c7al-queue", "dev-cpu-c7ax-queue",
        "run-cpu-c7am-queue", "run-cpu-c7al-queue", "run-cpu-c7ax-queue",
        "dev-gpu-queue", "run-gpu-queue",
        "64 vCPU", "32 vCPU", "non-preemptive", "G6f",
    ):
        assert value in text


def test_batch_usage_has_copyable_admin_sequence():
    text = Path("docs/batch-usage.md").read_text()
    assert "trainer-batch-admin inventory" in text
    assert "trainer-batch-admin deploy" in text
    assert "trainer-batch-admin smoke" in text
    assert "trainer-batch-admin cleanup" in text
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
cd rtrrl/infra/control-plane
uv run pytest tests/test_batch_docs.py -v
```

Expected: `docs/batch-usage.md` is missing.

- [ ] **Step 3: Write user examples**

`resources-cpu.yaml` contains three independent groups selecting c7am, c7al,
and c7ax. `resources-gpu.yaml` selects g6x. Neither file contains an AWS queue
name.

Document:

```bash
trainerctl run --config examples/resources-cpu.yaml
trainerctl run --config examples/resources-gpu.yaml
trainer-heavy-test submit --profile c7al --image "${IMAGE_DIGEST}" \
  memo/tests/test_logging_compat.py
```

Explain that the first two commands route to run queues and the third routes to
a dev queue, that run priority does not preempt running dev jobs, and that
users never choose queue names. Until control-plane Tasks 4–5 deliver the
formal submission adapter and `trainerctl run` implementation, label the first
two commands as the stable post-Task-5 interface rather than claiming they are
already executable. The admin CLI and heavy-test command delivered by this plan
must be documented as immediately executable.

- [ ] **Step 4: Write operator commands and supersession notes**

Document the exact dry-run-first sequence:

```bash
trainer-batch-admin inventory
trainer-batch-admin deploy
trainer-batch-admin deploy --execute
trainer-batch-admin smoke --cpu-image "${CPU_IMAGE}" --gpu-image "${GPU_IMAGE}"
trainer-batch-admin smoke --cpu-image "${CPU_IMAGE}" --gpu-image "${GPU_IMAGE}" --execute
trainer-batch-admin cleanup --smoke-report "${SMOKE_REPORT}"
trainer-batch-admin cleanup --smoke-report "${SMOKE_REPORT}" --execute
```

Show the JSON fields for created, reused, deleted, deferred, job IDs, queue
ARNs, instance types, and L4 evidence. State that G6f is unsupported.

Add a short `Superseded by
2026-07-20-shared-dev-run-batch-queues-design.md` notice to the old topology
sections without deleting their historical content.

- [ ] **Step 5: Verify and commit**

Run:

```bash
uv run pytest tests/test_batch_docs.py -v
uv run ruff check src tests
git diff --check
```

Commit:

```bash
git add rtrrl/infra/control-plane/docs \
  rtrrl/infra/control-plane/examples \
  rtrrl/infra/control-plane/tests/test_batch_docs.py \
  docs/superpowers/specs/2026-07-20-aws-batch-migration-design.md \
  docs/superpowers/specs/2026-07-20-batch-heavy-test-runner-design.md
git commit -m "docs(infra): explain shared Batch queue usage"
```

---

### Task 7: Authorized Real Deployment, Smoke, and Cleanup

**Files:**
- Create after execution: `rtrrl/infra/control-plane/docs/shared-queues-rollout-report.md`

**Interfaces:**
- Consumes the completed admin CLI and passing local suite.
- Produces an auditable rollout report; no reusable source code is added here.

- [ ] **Step 1: Run complete local verification**

Run:

```bash
cd rtrrl/infra/control-plane
uv run pytest -v
uv run ruff check src tests
bash -n ../../../infra/batch/heavy-tests/build-image.sh
git diff --check
```

Expected: zero test failures, Ruff diagnostics, shell syntax errors, or
whitespace errors.

- [ ] **Step 2: Generate read-only inventory and dry runs**

Run:

```bash
trainer-batch-admin inventory
trainer-batch-admin deploy
```

Record the exact eight creates/reuses, every old queue, every nonterminal job ID,
and the three environment cleanup candidates. Do not mutate AWS.

- [ ] **Step 3: Obtain explicit real-AWS authorization**

Present one yes/no authorization containing:

- eight queue create/reuse actions;
- eight exact smoke cases and their image digests;
- current old queues with zero jobs;
- old queues deferred because of exact job IDs;
- exact environment deletion candidates;
- confirmation that no jobs will be cancelled.

Stop until the user answers yes.

- [ ] **Step 4: Deploy and verify the eight queues**

Run:

```bash
trainer-batch-admin deploy --execute
trainer-batch-admin inventory
```

Expected: all eight queues are `VALID/ENABLED`; priorities are 10/100 and
bindings are exact.

- [ ] **Step 5: Build immutable smoke images and run eight jobs**

Run:

```bash
CPU_JSON="$(../../../infra/batch/heavy-tests/build-image.sh --profile c7ax)"
CPU_IMAGE="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["image"])' "${CPU_JSON}")"
GPU_JSON="$(../../../infra/batch/heavy-tests/build-image.sh --profile g6x)"
GPU_IMAGE="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["image"])' "${GPU_JSON}")"
trainer-batch-admin smoke \
  --cpu-image "${CPU_IMAGE}" \
  --gpu-image "${GPU_IMAGE}" \
  --execute
```

Expected: eight `SUCCEEDED` jobs, exact instance types for all profiles, and
JAX/L4 evidence for both GPU queues. Save the emitted report under
`.trainer/smoke/trainer-smoke-shared-queues.json`.

- [ ] **Step 6: Switch routing and run cleanup dry-run**

The routing switch is the already-deployed code from Tasks 1–4. Re-run one dev
submission and resolve one formal run profile without executing training to
prove queue selection:

```bash
python3 -c 'from trainer_infra.batch_topology import ExecutionPurpose, queue_for; print(queue_for(ExecutionPurpose.RUN, "c7al").name)'
```

Expected: `run-cpu-c7al-queue`.

Run:

```bash
SMOKE_REPORT=".trainer/smoke/trainer-smoke-shared-queues.json"
trainer-batch-admin cleanup --smoke-report "${SMOKE_REPORT}"
```

Inspect exact deletions and deferred resources. If the output contains a new
queue, a retained environment, an active-job queue, or an unlisted resource,
stop without `--execute`.

- [ ] **Step 7: Obtain cleanup authorization and execute**

Present the exact dry-run deletion/deferred sets for a second yes/no
authorization. After approval:

```bash
trainer-batch-admin cleanup \
  --smoke-report "${SMOKE_REPORT}" \
  --execute
trainer-batch-admin inventory
```

Expected: idle old queues and unreferenced unneeded environments are absent;
active-job resources remain with job IDs; all four new queues and retained
environments remain exact.

- [ ] **Step 8: Verify temporary smoke-artifact cleanup**

Read the cleanup report emitted in Step 7. Verify that every smoke
job-definition revision, temporary ECR tag, and exact CloudWatch log stream in
the smoke report appears in the deleted set, while the referenced digest
manifests remain. Verify no Aim run or formal S3 prefix was created.

- [ ] **Step 9: Write and commit the rollout report**

The report contains:

- account/region and timestamp;
- queue and compute-environment ARNs;
- priorities and bindings;
- eight job IDs, definitions, digests, instance types, exits, and L4 evidence;
- deleted old resources;
- deferred resources and blocking job IDs;
- smoke artifact cleanup results;
- assertion that no job was cancelled.

Run final read-only inventory and the complete local suite, then commit:

```bash
git add rtrrl/infra/control-plane/docs/shared-queues-rollout-report.md
git commit -m "docs(infra): record shared queue rollout"
```

Expected final state: eight valid shared dev/run queues, four retained exact
compute environments, no idle superseded queue/environment left, and a precise
deferred list for resources still serving old jobs.
