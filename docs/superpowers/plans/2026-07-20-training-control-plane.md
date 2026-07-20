# Training Control Plane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the independent local control plane and serial Batch worker that resolve explicit configuration groups, run independent Optuna loops, and complete a whole experiment from one CLI invocation.

**Architecture:** A standalone uv project under `rtrrl/infra/control-plane` owns configuration, image metadata, Optuna, Aim result collection, AWS adapters, and orchestration. A standard-library worker under `rtrrl/infra/worker` executes ordered run bundles in child processes and never imports Optuna or control-plane code.

**Tech Stack:** Python 3.10, uv, Pydantic 2, PyYAML, Optuna 4, Aim 3.28, boto3, pytest, ruff.

## Global Constraints

- Implement only in the dedicated `feature/trainer-infra` worktree.
- Do not import JAX, Brax, training scripts, or `rtrrl/logging_util.py` from the control plane.
- Every explicit `groups` entry is an independent study even when scripts and parameters match.
- Default parameter policy is `scan_unfixed`; `explicit_scan` is also supported.
- An experiment range replaces `default_search`; only genuine declared constraints limit it.
- One trial equals one logical run and one Aim run.
- A single `trainerctl run` advances all groups independently until completion.
- CPU profile is exactly `rtrrl-cpu-c7am-queue`, c7a.medium, 1 vCPU, 1600 MiB, 0 GPU.
- GPU profile is exactly `rtrrl-gpu-g6x-queue`, g6.xlarge, 4 vCPU, 12000 MiB, 1 GPU.
- `trainerctl run` never creates or mutates a queue or compute environment.
- There is no resume, next, history-import, shared-space, or multi-controller feature.
- Commit commands below require separate explicit user authorization before execution.

---

### Task 1: Standalone Package and Configuration Contracts

**Files:**
- Create: `rtrrl/infra/control-plane/pyproject.toml`
- Create: `rtrrl/infra/control-plane/src/trainer_infra/__init__.py`
- Create: `rtrrl/infra/control-plane/src/trainer_infra/models.py`
- Create: `rtrrl/infra/control-plane/src/trainer_infra/loaders.py`
- Create: `rtrrl/infra/control-plane/src/trainer_infra/resolve.py`
- Test: `rtrrl/infra/control-plane/tests/test_models.py`
- Test: `rtrrl/infra/control-plane/tests/test_resolve.py`

**Interfaces:**
- Produces `load_experiment(path: Path) -> ExperimentSpec`.
- Produces `load_script_catalog(path: Path) -> ScriptCatalog`.
- Produces `resolve_experiment(spec: ExperimentSpec, catalogs: Mapping[str, ScriptCatalog]) -> ResolvedExperiment`.
- Later tasks consume immutable `ResolvedGroup` objects.

- [ ] **Step 1: Write failing contract tests**

```python
def test_group_is_independent_and_defaults_are_resolved(catalog):
    spec = ExperimentSpec.model_validate({
        "experiment": {"name": "hopper"},
        "defaults": {
            "image": "repo/image:dev",
            "resources": {"profile": "gpu"},
            "hpo": {"total_trials": 5, "configs_per_batch": 2},
            "execution": {"runs_per_job": 2},
            "parameters": {"seed": {"values": [7]}},
        },
        "groups": {
            "shared": {"script": "rtrrl", "parameters": {"topology": {"values": ["shared"]}}},
            "dual": {"script": "rtrrl", "parameters": {"topology": {"values": ["dual"]}}},
        },
    })
    resolved = resolve_experiment(spec, {"repo/image@sha256:" + "a" * 64: catalog})
    assert [group.name for group in resolved.groups] == ["shared", "dual"]
    assert resolved.groups[0].study_key != resolved.groups[1].study_key
    assert resolved.groups[0].parameters["seed"].fixed_value == 7


def test_scan_unfixed_uses_default_search(catalog):
    group = resolve_one_group(catalog, policy="scan_unfixed", parameters={})
    assert group.parameters["learning_rate"].search_domain == ContinuousSearch(
        low=1e-5, high=1e-2, log=True, integer=False, step=None
    )


def test_experiment_domain_replaces_default_search(catalog):
    group = resolve_one_group(
        catalog,
        policy="scan_unfixed",
        parameters={"learning_rate": {"min": 1e-7, "max": 0.5, "scale": "log"}},
    )
    assert group.parameters["learning_rate"].search_domain == ContinuousSearch(
        low=1e-7, high=0.5, log=True, integer=False, step=None
    )
```

- [ ] **Step 2: Verify RED**

Run:

```bash
cd rtrrl/infra/control-plane
uv run pytest tests/test_models.py tests/test_resolve.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'trainer_infra'`.

- [ ] **Step 3: Add the package and exact model surface**

```toml
[project]
name = "trainer-infra"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "aim==3.28.0",
  "boto3>=1.35,<2",
  "optuna>=4,<5",
  "pydantic>=2.8,<3",
  "pyyaml>=6,<7",
]

[project.scripts]
trainerctl = "trainer_infra.cli:main"

[dependency-groups]
dev = ["pytest>=8,<9", "ruff>=0.9,<1"]

[tool.uv]
package = true

[tool.pytest.ini_options]
pythonpath = ["src"]

[tool.ruff]
line-length = 100
```

```python
JsonScalar: TypeAlias = str | int | float | bool | None


class ParameterPolicy(str, Enum):
    SCAN_UNFIXED = "scan_unfixed"
    EXPLICIT_SCAN = "explicit_scan"


class DiscreteDomain(BaseModel):
    values: list[JsonScalar]

    @model_validator(mode="after")
    def require_values(self) -> "DiscreteDomain":
        if not self.values:
            raise ValueError("values must not be empty")
        return self


class ContinuousDomain(BaseModel):
    min: float
    max: float
    scale: Literal["linear", "log"] = "linear"

    @model_validator(mode="after")
    def require_finite_ordered_bounds(self) -> "ContinuousDomain":
        if not math.isfinite(self.min) or not math.isfinite(self.max) or self.min >= self.max:
            raise ValueError("continuous bounds must be finite and min < max")
        if self.scale == "log" and self.min <= 0:
            raise ValueError("log domains require min > 0")
        return self


class HpoSpec(BaseModel):
    total_trials: PositiveInt
    configs_per_batch: PositiveInt
    parameter_policy: ParameterPolicy = ParameterPolicy.SCAN_UNFIXED

    @model_validator(mode="after")
    def batch_fits_budget(self) -> "HpoSpec":
        if self.configs_per_batch > self.total_trials:
            raise ValueError("configs_per_batch must not exceed total_trials")
        return self


class ExecutionSpec(BaseModel):
    runs_per_job: PositiveInt
    max_infra_retries: NonNegativeInt = 2
    max_algorithm_retries: NonNegativeInt = 0
    retry_backoff_seconds: PositiveInt = 30
    aim_result_timeout_seconds: PositiveInt = 600


@dataclass(frozen=True)
class DiscreteSearch:
    values: tuple[JsonScalar, ...]


@dataclass(frozen=True)
class ContinuousSearch:
    low: float
    high: float
    log: bool
    integer: bool
    step: int | float | None


SearchDomain: TypeAlias = DiscreteSearch | ContinuousSearch


@dataclass(frozen=True)
class ResolvedParameter:
    fixed_value: JsonScalar | None
    search_domain: SearchDomain | None
```

Implement the remaining Pydantic models with `extra="forbid"` and the exact fields from the specification. `ResolvedGroup` must be frozen and expose:

```python
def searchable_parameters(self) -> Mapping[str, SearchDomain]:
    return {
        name: parameter.search_domain
        for name, parameter in self.parameters.items()
        if parameter.search_domain is not None
    }
```

- [ ] **Step 4: Implement resolver precedence**

```python
def merge_mapping(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key == "parameters":
            params = merged.setdefault("parameters", {})
            for field_name, domain in value.items():
                params[field_name] = copy.deepcopy(domain)
        elif isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_mapping(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged
```

Resolve in order: descriptor defaults, experiment defaults, group fields, group
overrides. Under `scan_unfixed`, an omitted searchable field uses
`default_search`; under `explicit_scan`, it resolves to singleton `default`.
Reject unknown fields and constraints violations with group and field names in
the message.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
uv run pytest tests/test_models.py tests/test_resolve.py -v
uv run ruff check src tests
git diff --check
```

Expected: all Task 1 tests pass; ruff and diff checks return zero.

- [ ] **Step 6: Review checkpoint**

Review only Task 1 files. If commit authorization is later granted:

```bash
git add rtrrl/infra/control-plane
git commit -m "feat(infra): define experiment configuration contracts"
```

---

### Task 2: Digest-Bound Script Catalogs

**Files:**
- Create: `rtrrl/infra/control-plane/src/trainer_infra/image_catalog.py`
- Create: `rtrrl/infra/control-plane/tests/test_image_catalog.py`
- Create: `rtrrl/infra/scripts/index.yaml`
- Create: `rtrrl/infra/scripts/rtrrl.yaml`
- Create: `rtrrl/infra/scripts/ppo_baseline.yaml`
- Create: `rtrrl/infra/scripts/sac_baseline.yaml`
- Modify: `rtrrl/infra/docker/Dockerfile`
- Modify: `rtrrl/infra/docker/Dockerfile.gpu`

**Interfaces:**
- Consumes `ScriptCatalog`.
- Produces `resolve_image(reference: str) -> ResolvedImage`.
- Produces `EcrCatalogReader.fetch(image: ResolvedImage) -> ScriptCatalog`.

- [ ] **Step 1: Write failing label and ECR tests**

```python
def test_catalog_label_round_trip(catalog):
    encoded = encode_catalog(catalog)
    assert decode_catalog(encoded) == catalog


def test_tag_is_resolved_once(fake_ecr, catalog):
    fake_ecr.tag_digest = "sha256:" + "a" * 64
    fake_ecr.catalog = catalog
    image = EcrCatalogReader(fake_ecr).resolve_and_fetch("repo/image:dev")
    assert image.reference.endswith("@sha256:" + "a" * 64)
    assert image.catalog == catalog
```

- [ ] **Step 2: Verify RED**

Run `uv run pytest tests/test_image_catalog.py -v`.

Expected: import failure for `trainer_infra.image_catalog`.

- [ ] **Step 3: Implement deterministic label codec**

```python
LABEL = "org.rtrrl.trainer.scripts.v1"


def encode_catalog(catalog: ScriptCatalog) -> str:
    raw = catalog.model_dump_json(exclude_none=True).encode()
    return base64.b64encode(gzip.compress(raw, mtime=0)).decode()


def decode_catalog(value: str) -> ScriptCatalog:
    raw = gzip.decompress(base64.b64decode(value, validate=True))
    return ScriptCatalog.model_validate_json(raw)
```

Implement ECR behind a protocol with `resolve_tag`, `get_manifest`, and
`get_config_blob`. Reject mutable references after resolution, missing labels,
unsupported protocol versions, and catalog/script duplication.

- [ ] **Step 4: Bind descriptors during image build**

Add a build step that validates `infra/scripts/index.yaml`, encodes the complete
catalog, and passes it as:

```dockerfile
ARG TRAINER_SCRIPT_CATALOG
LABEL org.rtrrl.trainer.scripts.v1="${TRAINER_SCRIPT_CATALOG}"
COPY infra/scripts /opt/trainer/scripts
```

Descriptors declare argv arrays, fields, budgets, objectives, and SDK protocol.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
uv run pytest tests/test_image_catalog.py -v
uv run ruff check src tests
git diff --check
```

Expected: all Task 2 tests pass and checks return zero.

---

### Task 3: Sampling, Identity, and Concrete Runs

**Files:**
- Create: `rtrrl/infra/control-plane/src/trainer_infra/identities.py`
- Create: `rtrrl/infra/control-plane/src/trainer_infra/sampling.py`
- Create: `rtrrl/infra/control-plane/src/trainer_infra/materialize.py`
- Test: `rtrrl/infra/control-plane/tests/test_sampling.py`
- Test: `rtrrl/infra/control-plane/tests/test_materialize.py`

**Interfaces:**
- `study_name(experiment_id: str, group: str) -> str`
- `sample_parameters(trial: Trial, group: ResolvedGroup) -> dict[str, JsonScalar]`
- `materialize_run(group: ResolvedGroup, trial: Trial, sampled: Mapping[str, JsonScalar]) -> ConcreteRun`

- [ ] **Step 1: Write failing sampling tests**

```python
def test_fixed_parameters_are_not_suggested(fake_trial, resolved_group):
    values = sample_parameters(fake_trial, resolved_group)
    assert "seed" not in fake_trial.suggested_names
    assert values["seed"] == 7


def test_run_identity_is_group_local(resolved_experiment):
    first = materialize_run(resolved_experiment.groups[0], FakeTrial(0), {})
    second = materialize_run(resolved_experiment.groups[1], FakeTrial(0), {})
    assert first.run_name == "shared-rtrrl-0001"
    assert second.run_name == "dual-rtrrl-0001"
    assert first.study_key != second.study_key
```

- [ ] **Step 2: Verify RED**

Run `uv run pytest tests/test_sampling.py tests/test_materialize.py -v`.

Expected: missing-module failures.

- [ ] **Step 3: Implement public Optuna sampling**

```python
def sample_parameters(trial: Trial, group: ResolvedGroup) -> dict[str, JsonScalar]:
    values = dict(group.fixed_parameters)
    for name, domain in group.searchable_parameters().items():
        if isinstance(domain, DiscreteSearch):
            values[name] = trial.suggest_categorical(name, domain.values)
        elif domain.integer:
            values[name] = trial.suggest_int(name, int(domain.low), int(domain.high), step=domain.step)
        else:
            values[name] = trial.suggest_float(name, domain.low, domain.high, log=domain.log)
    return values
```

Create studies with `TPESampler(constant_liar=True)`. Track finite discrete
combinations and skip duplicates. Materialize canonical YAML/JSON and SHA-256.

- [ ] **Step 4: Verify GREEN**

Run targeted tests, ruff, and `git diff --check`; expect zero failures.

---

### Task 4: S3 Protocol, Serial Worker, and Batch Adapter

**Files:**
- Create: `rtrrl/infra/control-plane/src/trainer_infra/adapters/protocols.py`
- Create: `rtrrl/infra/control-plane/src/trainer_infra/adapters/s3.py`
- Create: `rtrrl/infra/control-plane/src/trainer_infra/adapters/aws_batch.py`
- Create: `rtrrl/infra/worker/worker.py`
- Test: `rtrrl/infra/control-plane/tests/test_s3.py`
- Test: `rtrrl/infra/control-plane/tests/test_aws_batch.py`
- Test: `rtrrl/infra/control-plane/tests/test_worker.py`

**Interfaces:**
- `ObjectStore.put_json/get_json/put_file`.
- `BatchAdapter.validate_profiles/ensure_job_definition/submit/poll`.
- Worker command: `python /opt/trainer/worker.py --bundle-s3-uri <uri>`.

- [ ] **Step 1: Write failing worker and AWS tests**

```python
def test_worker_runs_children_serially_and_continues(tmp_path, fake_store):
    bundle = bundle_with_commands(
        [python_exit_command(1), python_write_command(tmp_path / "second")]
    )
    code = run_bundle(bundle, fake_store)
    assert code == 1
    assert (tmp_path / "second").exists()
    assert [marker.exit_code for marker in fake_store.markers] == [1, 0]


def test_profile_drift_fails_without_update(fake_batch):
    fake_batch.cpu_instance_type = "c7a.xlarge"
    with pytest.raises(ProfileDriftError):
        AwsBatchAdapter(fake_batch, CONTROL_CONFIG).validate_profiles()
    assert fake_batch.update_calls == []
```

- [ ] **Step 2: Verify RED**

Run the three targeted test files; expect missing-module failures.

- [ ] **Step 3: Implement protocols and worker**

```python
@runtime_checkable
class ObjectStore(Protocol):
    def get_json(self, uri: str) -> Mapping[str, Any]: ...
    def put_json(self, uri: str, value: Mapping[str, Any]) -> None: ...
    def put_file(self, uri: str, path: Path) -> None: ...


def execute_run(run: RunBundle, store: ObjectStore) -> CompletionMarker:
    started = datetime.now(timezone.utc)
    completed = subprocess.run(run.argv, shell=False, env=run.environment, check=False)
    return CompletionMarker(
        run_id=run.run_id,
        attempt=run.attempt,
        exit_code=completed.returncode,
        started_at=started,
        finished_at=datetime.now(timezone.utc),
        artifacts=collect_registered_artifacts(run.artifact_directory),
    )
```

The top-level worker catches one child failure, writes its marker, and proceeds.
It returns nonzero after all children when any child failed.

- [ ] **Step 4: Implement strict profile validation**

Encode exact expected profile structures. Validation compares CE status/state,
instance type, AMI family, queue status/state/binding, and job resources.
`ensure_job_definition` keys identity by profile, image digest, and worker
protocol. No queue/CE create or update method exists.

- [ ] **Step 5: Verify GREEN**

Run targeted tests, ruff, and `git diff --check`; expect zero failures.

---

### Task 5: Aim Collection, Automatic Controller, and CLI

**Files:**
- Create: `rtrrl/infra/control-plane/src/trainer_infra/aim_reader.py`
- Create: `rtrrl/infra/control-plane/src/trainer_infra/controller.py`
- Create: `rtrrl/infra/control-plane/src/trainer_infra/cli.py`
- Create: `rtrrl/infra/control-plane/src/trainer_infra/__main__.py`
- Test: `rtrrl/infra/control-plane/tests/test_aim_reader.py`
- Test: `rtrrl/infra/control-plane/tests/test_controller.py`
- Test: `rtrrl/infra/control-plane/tests/test_cli.py`
- Create: `rtrrl/infra/control-plane/README.md`

**Interfaces:**
- `AimReader.completed_result(run_id, objective) -> CompletedResult | None`.
- `ExperimentController.run(spec) -> ExperimentReport`.
- CLI commands: `validate`, `run`, `status`.

- [ ] **Step 1: Write failing automatic-loop test**

```python
def test_one_run_command_completes_all_group_batches(fakes, experiment):
    report = ExperimentController(**fakes).run(experiment)
    assert report.group("shared").allocated_batches == [2, 2, 1]
    assert report.group("dual").allocated_batches == [2, 2, 1]
    assert report.complete
    assert fakes.optuna.tell_writers == {threading.get_ident()}
```

Add tests for Aim finalized markers, exact run ID lookup, finite objectives,
bounded Aim waits, two infrastructure retries, zero default algorithm retries,
and absence of `next`, `resume`, and history commands.

- [ ] **Step 2: Verify RED**

Run `uv run pytest tests/test_aim_reader.py tests/test_controller.py tests/test_cli.py -v`.

Expected: missing-module failures.

- [ ] **Step 3: Implement the event loop**

```python
def run(self, spec: ExperimentSpec) -> ExperimentReport:
    experiment = self.resolve(spec)
    self.batch.validate_profiles(experiment.required_profiles())
    groups = {group.name: GroupLoop.create(group, self.storage) for group in experiment.groups}
    while any(not loop.complete for loop in groups.values()):
        ready = [run for loop in groups.values() for run in loop.ask_ready_batch()]
        jobs = self.scheduler.pack(ready)
        submitted = self.submit_jobs(jobs)
        outcomes = self.wait_for_outcomes(submitted)
        for outcome in outcomes:
            groups[outcome.group].accept(outcome)
    return ExperimentReport.from_loops(experiment, groups)
```

`accept()` reads/replays Aim buffers, waits for finalized Aim, and calls
`study.tell()` on the controller thread only.

- [ ] **Step 4: Implement CLI and documentation**

```python
def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "validate":
        load_and_resolve(args.experiment)
        return 0
    if args.command == "run":
        report = build_controller(args.control).run(load_experiment(args.experiment))
        print(report.render())
        return 0 if report.complete else 1
    return render_status(args.experiment_id)
```

Document complete YAML, one-command lifecycle, identities, failure semantics,
and non-goals.

- [ ] **Step 5: Verify complete control-plane plan**

Run:

```bash
cd rtrrl/infra/control-plane
uv run pytest -v
uv run ruff check src tests
git diff --check
```

Expected: zero failures and diagnostics.

- [ ] **Step 6: Review checkpoint**

Review the full control-plane diff against
`docs/superpowers/specs/2026-07-20-training-control-plane-design.md`. If commit
authorization is granted, commit Task 2–5 changes in reviewable task commits
rather than one aggregate commit.
