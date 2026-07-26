# Memo-First Script Registration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register two observable memo algorithm-family launchers while keeping environment as experiment configuration and removing the unsupported legacy RTRRL registrations.

**Architecture:** Generalize the control-plane environment snapshot, move the facility SDK into one repository-level package consumed by both projects, and extend memo evaluation outputs with fixed-shape traces without changing training updates. Two memo launchers map validated environment names to existing builders; the memo image alone embeds their digest-bound descriptors.

**Tech Stack:** Python 3.12, uv, Pydantic 2, JAX 0.10, Brax 0.14, Aim 3.28, Rerun 0.34, pytest, Ruff, Docker, GitHub Actions.

**Supersedes:** Task 5 of
`2026-07-20-training-observability-sdk.md` and the legacy-RTRRL image binding
portion of Task 2 in `2026-07-20-training-control-plane.md`. The generic image
catalog codec and digest reader from that control-plane task remain in force.

## Global Constraints

- The first release registers exactly `memo_stream_ac` and `memo_rtrrl`.
- `memo_stream_ac` accepts only the `rtu_rtrl` topology.
- `memo_rtrrl` accepts only the `shared` topology.
- Environment is an immutable group configuration, never part of script identity.
- Every explicit group remains an independent study.
- Training summaries come only from completed `RecordEpisodeStatistics` entries and use `state.step`.
- Evaluation metrics are never relabeled as training metrics.
- Rerun receives complete evaluation episodes with N+1 observations and N transitions.
- Termination and truncation remain separate; a combined `done` is not guessed to be termination.
- SDK calls and NumPy conversion remain outside JIT.
- Training equations, training PRNG splits, optimizer state, and training return values do not change.
- The SDK has exactly one source tree at repository-level `training-sdk/`.
- Only the memo image embeds the initial script catalog.
- QRC, TBPTT, independent RTRRL, examples, random policy, and legacy `rtrrl` scripts are absent from the catalog.
- Commit commands require the existing explicit user authorization; never push.

---

### Task 1: Generic Environment and Descriptor Contracts

**Files:**
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/models.py`
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/resolve.py`
- Modify: `rtrrl/infra/control-plane/tests/test_resolve.py`
- Modify: `rtrrl/infra/control-plane/tests/test_materialize.py`
- Modify: `rtrrl/infra/control-plane/tests/test_image_catalog.py`

**Interfaces:**
- Produces: `EnvironmentSpec(name: str, options: Mapping[str, JsonValue])`.
- Produces: `ScriptDescriptor.environments: tuple[str, ...]`.
- Produces: `FieldDescriptor.choices: tuple[JsonScalar, ...] | None`.
- Preserves: `resolve_experiment()` and `materialize_run()` public signatures.

- [ ] **Step 1: Write failing generic-environment tests**

```python
def test_environment_is_generic_and_immutable():
    spec = EnvironmentSpec(
        name="memory_chain",
        options={"length": 75, "nested": {"observe": ["query"]}},
    )
    with pytest.raises(TypeError):
        spec.options["nested"]["observe"][0] = "answer"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), Path("x"), {1, 2}])
def test_environment_options_reject_non_json_values(value):
    with pytest.raises((TypeError, ValueError), match="JSON|finite"):
        EnvironmentSpec(name="memory_chain", options={"bad": value})
```

- [ ] **Step 2: Write failing descriptor restriction tests**

```python
def test_descriptor_rejects_unlisted_environment(resolved_defaults, stream_catalog):
    group = group_spec(script="memo_stream_ac", environment={"name": "unknown", "options": {}})
    with pytest.raises(ValueError, match="memo_stream_ac.*unknown"):
        resolve_group(group, resolved_defaults, stream_catalog)


def test_single_choice_topology_rejects_other_value(stream_group):
    stream_group.parameters["agent_type"] = {"values": ["rtu_tbptt"]}
    with pytest.raises(ValueError, match="agent_type.*rtu_rtrl"):
        resolve_experiment(stream_group)
```

- [ ] **Step 3: Run the tests and verify RED**

Run:

```bash
cd rtrrl/infra/control-plane
uv run pytest tests/test_resolve.py tests/test_materialize.py tests/test_image_catalog.py -v
```

Expected: failures because `EnvironmentSpec` is legacy-specific and descriptors have no environment/choice restrictions.

- [ ] **Step 4: Implement recursively frozen generic environment values**

```python
JsonValue: TypeAlias = (
    JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
)


def thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_json(item) for item in value]
    return value


class EnvironmentSpec(ContractModel):
    name: str
    options: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("options")
    @classmethod
    def require_finite_json(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        json.dumps(value, allow_nan=False)
        return value

    def model_post_init(self, __context: Any) -> None:
        object.__setattr__(self, "options", freeze_json(dict(self.options)))
```

Materialization must thaw the frozen value through the existing JSON helper:

```python
config = {
    "environment": {
        "name": group.environment.name,
        "options": thaw_json(group.environment.options),
    },
    "logging": group.logging.model_dump(mode="json"),
    "parameters": final,
    "training_budget": group.training_budget.model_dump(mode="json"),
}
```

- [ ] **Step 5: Add explicit descriptor restrictions**

```python
class FieldDescriptor(ContractModel):
    path: str
    type: Literal["bool", "int", "float", "str"]
    default: JsonScalar
    searchable: bool = False
    constraints: FieldConstraints = FieldConstraints()
    default_search: ParameterDomain | None = None
    choices: tuple[JsonScalar, ...] | None = None


class ScriptDescriptor(ContractModel):
    name: str
    argv: tuple[str, ...]
    sdk_protocol_version: str
    defaults: DescriptorDefaults
    objective: ObjectiveSpec
    environments: tuple[str, ...]
    fields: dict[str, FieldDescriptor]
```

Resolver checks happen before parameter domains are accepted:

```python
if configuration.environment.name not in descriptor.environments:
    raise ValueError(
        f"group '{group.name}' script '{descriptor.name}' does not support "
        f"environment '{configuration.environment.name}'"
    )
if field.choices is not None and value not in field.choices:
    raise ValueError(f"field '{name}' must be one of {list(field.choices)!r}")
```

- [ ] **Step 6: Verify GREEN**

Run:

```bash
cd rtrrl/infra/control-plane
uv run pytest -v
uv run ruff check src tests
git diff --check
```

Expected: all control-plane tests pass.

- [ ] **Step 7: Commit**

```bash
git add rtrrl/infra/control-plane
git commit -m "feat(infra): generalize script environment contracts"
```

---

### Task 2: Single-Source SDK Package and Child Bootstrap

**Files:**
- Create: `training-sdk/pyproject.toml`
- Move: `rtrrl/training_sdk/*.py` → `training-sdk/src/training_sdk/*.py`
- Move: `rtrrl/tests/training_sdk/*.py` → `training-sdk/tests/*.py`
- Create: `training-sdk/src/training_sdk/bootstrap.py`
- Create: `training-sdk/tests/test_bootstrap.py`
- Modify: `rtrrl/pyproject.toml`
- Modify: `rtrrl/uv.lock`
- Modify: `memo/pyproject.toml`
- Modify: `memo/uv.lock`
- Modify: `memo/logging_util.py`
- Test: `memo/tests/test_logging_compat.py`

**Interfaces:**
- Produces: `bootstrap_from_environment(environ: Mapping[str, str] = os.environ) -> TrainingRun | None`.
- Produces: repository package `training-sdk` imported as `training_sdk`.
- Preserves: `current_run()`, `maybe_current_run()`, `set_current_run()`, and all logger compatibility.

- [ ] **Step 1: Write failing bootstrap tests**

```python
def test_missing_context_is_local_noop(monkeypatch):
    monkeypatch.delenv("TRAINER_RUN_CONTEXT_PATH", raising=False)
    assert bootstrap_from_environment() is None
    assert maybe_current_run() is None


def test_bootstrap_is_idempotent_for_one_context(valid_context_path, fake_backends):
    env = {"TRAINER_RUN_CONTEXT_PATH": str(valid_context_path)}
    first = bootstrap_from_environment(env)
    second = bootstrap_from_environment(env)
    assert second is first


def test_bootstrap_rejects_conflicting_contexts(first_context, second_context):
    bootstrap_from_environment({"TRAINER_RUN_CONTEXT_PATH": str(first_context)})
    with pytest.raises(RuntimeError, match="different run context"):
        bootstrap_from_environment({"TRAINER_RUN_CONTEXT_PATH": str(second_context)})
```

- [ ] **Step 2: Verify RED**

Run:

```bash
cd training-sdk
uv run --with pytest pytest tests/test_bootstrap.py -v
```

Expected: import or missing-function failure.

- [ ] **Step 3: Create the standalone package**

```toml
[project]
name = "training-sdk"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "aim==3.28.*",
  "numpy>=2",
  "rerun-sdk>=0.34,<0.35",
]

[dependency-groups]
dev = ["pytest", "ruff"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

Both project manifests use the same path source:

```toml
dependencies = [
  "training-sdk",
]

[tool.uv.sources]
training-sdk = { path = "../training-sdk" }
```

- [ ] **Step 4: Implement bootstrap with injectable factories**

```python
RUN_CONTEXT_ENV = "TRAINER_RUN_CONTEXT_PATH"


def bootstrap_from_environment(
    environ: Mapping[str, str] = os.environ,
    *,
    aim_factory: Callable[[RunContext, Mapping[str, str]], AimSink] | None = None,
    rerun_factory: Callable[[RunContext], RerunSink] | None = None,
) -> TrainingRun | None:
    value = environ.get(RUN_CONTEXT_ENV)
    if value is None:
        return None
    path = Path(value).resolve()
    current = maybe_current_run()
    if current is not None:
        if current.context_path != path:
            raise RuntimeError("training SDK already uses a different run context")
        return current
    context = RunContext.from_path(path)
    if aim_factory is None:
        aim_factory = lambda context, env: AimAdapter(repo=env.get("AIM_REPO"))
    if rerun_factory is None:
        rerun_factory = lambda context: RerunAdapter(
            context,
            every_episodes=int(context.logging["rerun_every_episodes"]),
            root=context.artifact_directory,
        )
    spool = EventSpool(context.artifact_directory / "aim-buffer" / "events.jsonl")
    run = TrainingRun(
        context,
        aim_factory(context, environ),
        rerun_factory(context),
        spool,
        context_path=path,
    )
    run.start()
    set_current_run(run)
    return run
```

The concrete factory reads only facility environment variables such as `AIM_REPO`; scripts never import Aim or Rerun.

- [ ] **Step 5: Port memo logger compatibility**

```python
class AimLogger(DummyLogger):
    def __init__(self, name, repo=None, hparams=None, run_name="", *, training_run=None):
        self.training_run = training_run
        if training_run is not None:
            self._final_metrics = {}
            return
        self._init_legacy_aim(name, repo, hparams, run_name)


def _facility_aim_logger(
    project_name: str,
    aim_repo: str | None,
    hparams: object,
    run_name: str,
) -> AimLogger:
    facility_run = maybe_current_run()
    return AimLogger(
        project_name,
        repo=aim_repo,
        hparams=hparams,
        run_name=run_name,
        training_run=facility_run,
    )
```

Copy behavior, not source duplication: `memo/logging_util.py` imports the standalone SDK and keeps its existing W&B/local behavior.

- [ ] **Step 6: Verify GREEN**

Run:

```bash
cd training-sdk
uv sync
uv run pytest -v
uv run ruff check src tests
cd ../memo
uv lock --check
uv run pytest tests/test_logging_compat.py -v
cd ../rtrrl
uv lock --check
git diff --check
```

Expected: SDK and both consumers resolve the same local package; all tests pass.

- [ ] **Step 7: Commit**

```bash
git add training-sdk rtrrl/pyproject.toml rtrrl/uv.lock memo/pyproject.toml memo/uv.lock memo/logging_util.py memo/tests
git rm -r rtrrl/training_sdk rtrrl/tests/training_sdk
git commit -m "refactor(sdk): share one training observability package"
```

---

### Task 3: Fixed-Shape Evaluation Traces and Termination Semantics

**Files:**
- Modify: `memo/memorax/online_ac/types.py`
- Modify: `memo/memorax/online_ac/build.py`
- Modify: `memo/memorax/online_ac/standard.py`
- Modify: `memo/memorax/online_ac/meta.py`
- Modify: `memo/memorax/environments/brax.py`
- Modify: `memo/memorax/environments/memory_chain.py`
- Modify: `memo/memorax/environments/kmemory_chain.py`
- Modify: `memo/memorax/environments/wrappers/record_episode_statistics.py`
- Create: `memo/tests/online_ac/test_eval_trace.py`
- Modify: `memo/tests/online_ac/test_evaluation_parity.py`
- Modify: `memo/tests/online_ac/test_jit_contract.py`

**Interfaces:**
- Produces: `EvalTrace` in `EvalSummary.trace`.
- Produces: `JAXEnvAdapter.trace_step_fn`.
- Preserves: training `step_fn`, training scans, and existing scalar evaluation result.

- [ ] **Step 1: Write failing trace-shape and ending tests**

```python
def test_eval_trace_has_n_plus_one_observations_and_n_transitions(trace):
    assert trace.observations.shape[0] == trace.actions.shape[0] + 1
    assert trace.rewards.shape[0] == trace.actions.shape[0]
    assert trace.terminals.shape == trace.truncations.shape == trace.rewards.shape


def test_first_truncation_sets_valid_transition_count(truncating_adapter):
    summary = evaluate_with_trace(truncating_adapter, num_steps=5)
    assert int(summary.trace.valid_transitions[0]) == 3
    assert not bool(summary.trace.terminals[2, 0])
    assert bool(summary.trace.truncations[2, 0])
```

- [ ] **Step 2: Write a failing JAXPR boundary test**

```python
def test_evaluation_trace_is_jittable_without_host_callbacks(program, key, state):
    jaxpr = str(jax.make_jaxpr(program.evaluate)(key, state, 8))
    assert "pure_callback" not in jaxpr
    assert "io_callback" not in jaxpr
```

- [ ] **Step 3: Verify RED**

Run:

```bash
cd memo
uv run pytest tests/online_ac/test_eval_trace.py tests/online_ac/test_jit_contract.py -v
```

Expected: missing `EvalTrace` and `trace_step_fn`.

- [ ] **Step 4: Add immutable trace types**

```python
@struct.dataclass
class EvalTrace:
    observations: Any
    actions: Any
    rewards: Any
    terminals: Any
    truncations: Any
    valid_transitions: Any
    environment_states: Any = None


@struct.dataclass
class EvalSummary:
    info: Any = None
    normalization: Any = None
    trace: EvalTrace | None = None
```

`JAXEnvAdapter` gets a separate evaluation-only callable:

```python
trace_step_fn: Callable[
    [Any, Any, Any, Any],
    tuple[Any, Any, Any, Any, Any, Any],
]
```

- [ ] **Step 5: Preserve native termination and truncation**

Brax rejects unavailable truncation instead of guessing:

```python
if "truncation" not in next_state.info:
    raise ValueError("Brax environment does not expose truncation")
truncated = next_state.info["truncation"].astype(bool)
terminated = jnp.logical_and(next_state.done.astype(bool), jnp.logical_not(truncated))
```

Memory-chain tasks return `terminated=done, truncated=False`. POPJym is omitted
from the initial launcher and descriptor because its current adapter does not
provide a tested truncation signal.

- [ ] **Step 6: Extend evaluation scans only**

```python
def eval_step(carry, _):
    key, env_state, observation = carry
    key, action_key = jax.random.split(key)
    action = policy(action_key, observation)
    next_observation, next_state, reward, terminated, truncated, info = (
        adapter.trace_step_fn(env_state, action, adapter.env_params, adapter.build_context)
    )
    transition = (action, reward, terminated, truncated, next_state)
    return (key, next_state, next_observation), (next_observation, transition)


(_, _, _), (next_observations, transitions) = jax.lax.scan(
    eval_step, initial, xs=None, length=num_steps
)
observations = jnp.concatenate([initial_observation[None], next_observations], axis=0)
finished = jnp.logical_or(terminals, truncations)
valid = first_true_index_plus_one_or_zero(finished)
```

Do not edit the policy-update or train-epoch scans.

- [ ] **Step 7: Verify parity and GREEN**

Run:

```bash
cd memo
uv run pytest tests/online_ac/test_eval_trace.py -v
uv run pytest tests/online_ac/test_evaluation_parity.py tests/online_ac/test_jit_contract.py -v
uv run pytest tests/online_ac/test_legacy_characterization.py -v
uv run pytest tests/online_ac/test_standard_parity.py tests/online_ac/test_meta_parity.py -v
uv run ruff check memorax/online_ac memorax/environments tests/online_ac
git diff --check
```

Expected: traces pass and all pre-existing golden/parity results remain unchanged.

- [ ] **Step 8: Commit**

```bash
git add memo/memorax memo/tests/online_ac
git commit -m "feat(memo): retain complete evaluation traces"
```

---

### Task 4: Two Facility Launchers and Host Observability

**Files:**
- Create: `memo/experiments/base/facility.py`
- Modify: `memo/experiments/base/experiment.py`
- Create: `memo/experiments/memo_stream_ac/run.py`
- Create: `memo/experiments/memo_rtrrl/run.py`
- Modify: `memo/experiments/stream_ac_memorychain/run.py`
- Modify: `memo/experiments/stream_ac_kmemorychain/run.py`
- Modify: `memo/experiments/stream_ac_mujoco_masked/run.py`
- Modify: `memo/experiments/rtrrl_hopper/run.py`
- Create: `memo/tests/test_facility_launchers.py`
- Create: `memo/tests/test_experiment_observability.py`
- Create: `memo/tests/fixtures/facility_stream_ac.yml`
- Create: `memo/tests/fixtures/facility_rtrrl.yml`
- Create: `memo/tests/fixtures/facility_run_context.json`

**Interfaces:**
- Produces: `load_facility_input(path: Path) -> FacilityInput`.
- Produces: `build_stream_ac_config(value: FacilityInput) -> ExperimentConfig`.
- Produces: `build_rtrrl_config(value: FacilityInput) -> RTRRLHopperConfig`.
- Consumes: `EvalSummary.trace` and standalone `training_sdk`.

- [ ] **Step 1: Write failing launcher tests**

```python
def test_environment_selects_builder_not_script_identity(tmp_path):
    memory = build_stream_ac_config(
        load_facility_input(write_config(tmp_path, environment="memory_chain"))
    )
    mujoco = build_stream_ac_config(
        load_facility_input(write_config(tmp_path, environment="mujoco_masked"))
    )
    assert isinstance(memory, StreamACMemoryChainConfig)
    assert isinstance(mujoco, StreamACMujocoMaskedConfig)
    assert STREAM_AC_SCRIPT_NAME == "memo_stream_ac"


@pytest.mark.parametrize("agent_type", ["rtu_tbptt", "lru_rtrl"])
def test_stream_launcher_rejects_unregistered_topology(agent_type, tmp_path):
    path = write_stream_config(tmp_path, parameters={"agent_type": agent_type})
    with pytest.raises(ValueError, match="rtu_rtrl"):
        build_stream_ac_config(load_facility_input(path))
```

- [ ] **Step 2: Write failing observability source tests**

```python
def test_completed_training_episodes_use_state_step(fake_logger):
    info = {
        "returned_episode": np.array([False, True, True]),
        "returned_episode_returns": np.array([0.0, 2.0, 3.0]),
        "returned_episode_lengths": np.array([0, 7, 9]),
    }
    emit_training_summaries(fake_logger, info, state_step=123)
    assert fake_logger.summaries == [
        {"env_steps": 123, "episode_return": 2.0, "episode_length": 7},
        {"env_steps": 123, "episode_return": 3.0, "episode_length": 9},
    ]


def test_incomplete_eval_trace_is_not_submitted(fake_logger, incomplete_trace):
    assert emit_eval_episode(fake_logger, incomplete_trace, episode_number=4) is None
    assert fake_logger.episodes == []
```

- [ ] **Step 3: Verify RED**

Run:

```bash
cd memo
uv run pytest tests/test_facility_launchers.py tests/test_experiment_observability.py -v
```

Expected: missing facility loader and host emitters.

- [ ] **Step 4: Implement strict nested facility input**

```python
@dataclass(frozen=True)
class FacilityInput:
    environment: Mapping[str, JsonValue]
    logging: Mapping[str, JsonValue]
    parameters: Mapping[str, JsonValue]
    training_budget: Mapping[str, JsonValue]


def load_facility_input(path: Path) -> FacilityInput:
    payload = yaml.safe_load(path.read_text())
    allowed = {"environment", "logging", "parameters", "training_budget"}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown facility config fields: {sorted(unknown)}")
    environment = payload["environment"]
    if set(environment) != {"name", "options"}:
        raise ValueError("environment requires exactly name and options")
    return FacilityInput(**payload)
```

The launchers use static maps, never import strings:

```python
STREAM_AC_ENVIRONMENTS = {
    "memory_chain": (StreamACMemoryChainConfig, make_memory_chain),
    "kmemory_chain": (StreamACKMemoryChainConfig, make_kmemory_chain),
    "mujoco_masked": (StreamACMujocoMaskedConfig, make_mujoco_masked),
}
RTRRL_ENVIRONMENTS = {"hopper": (RTRRLHopperConfig, make_hopper)}
```

Environment option names map explicitly to existing dataclass fields:

```python
STREAM_AC_OPTION_FIELDS = {
    "memory_chain": {
        "length": "chain_length",
        "max_episode_steps": "chain_length",
    },
    "kmemory_chain": {
        "length": "chain_length",
        "num_bits": "num_bits",
        "max_episode_steps": "chain_length",
    },
    "mujoco_masked": {
        "env_name": "env_name",
        "mode": "mode",
        "backend": "backend",
        "max_episode_steps": "max_episode_steps",
    },
}
RTRRL_OPTION_FIELDS = {
    "hopper": {
        "mode": "mode",
        "backend": "backend",
        "max_episode_steps": "max_episode_steps",
        "normalize_obs": "normalize_obs",
        "normalize_reward": "normalize_reward",
    }
}
```

For both memory-chain environments, `max_episode_steps` must equal `length`;
the launcher rejects unequal values rather than silently choosing one.

- [ ] **Step 5: Emit truthful host records**

```python
def emit_training_summaries(logger, info: Mapping[str, Any], *, state_step: int) -> None:
    completed = np.asarray(info["returned_episode"], dtype=bool).reshape(-1)
    returns = np.asarray(info["returned_episode_returns"]).reshape(-1)
    lengths = np.asarray(info["returned_episode_lengths"]).reshape(-1)
    for index in np.flatnonzero(completed):
        logger.log_episode_summary(
            env_steps=int(state_step),
            episode_return=float(returns[index]),
            episode_length=int(lengths[index]),
        )


def episode_from_trace(
    trace: EvalTrace,
    *,
    episode_number: int,
    start_env_steps: int,
    phase: str = "eval",
) -> Episode | None:
    counts = np.asarray(trace.valid_transitions).reshape(-1)
    completed_environments = np.flatnonzero(counts > 0)
    if completed_environments.size == 0:
        return None
    environment_index = int(completed_environments[0])
    count = int(counts[environment_index])
    return Episode(
        number=episode_number,
        phase=phase,
        observations=np.asarray(trace.observations)[: count + 1, environment_index],
        actions=np.asarray(trace.actions)[:count, environment_index],
        rewards=np.asarray(trace.rewards)[:count, environment_index],
        terminals=np.asarray(trace.terminals)[:count, environment_index],
        truncations=np.asarray(trace.truncations)[:count, environment_index],
        start_env_steps=start_env_steps,
        end_env_steps=start_env_steps + count,
    )
```

`train_loop()` calls `bootstrap_from_environment()` before creating loggers, uses
`int(state.step)` for both metric step and summaries, and never emits an eval
value through `log_episode_summary`. It maintains a host integer
`evaluation_episode_number`, incremented only when `episode_from_trace()` returns
a complete episode, and passes that value to `logger.log_episode()`.

- [ ] **Step 6: Run short CLI and regression tests**

Run:

```bash
cd memo
uv run pytest tests/test_facility_launchers.py tests/test_experiment_observability.py -v
uv run python experiments/memo_stream_ac/run.py --config_path tests/fixtures/facility_stream_ac.yml
uv run python experiments/memo_rtrrl/run.py --config_path tests/fixtures/facility_rtrrl.yml
uv run pytest tests/online_ac -v
uv run ruff check experiments tests/test_facility_launchers.py tests/test_experiment_observability.py
git diff --check
```

Expected: both short commands exit zero, produce facility records, and online-AC regression tests pass.

- [ ] **Step 7: Commit**

```bash
git add memo/experiments memo/tests
git commit -m "feat(memo): add observable facility launchers"
```

---

### Task 5: Memo Catalog and Lock-Consistent Images

**Files:**
- Create: `memo/infra/scripts/index.yaml`
- Create: `memo/infra/scripts/memo_stream_ac.yaml`
- Create: `memo/infra/scripts/memo_rtrrl.yaml`
- Modify: `memo/infra/docker/Dockerfile`
- Modify: `memo/infra/docker/Dockerfile.gpu`
- Create: `.dockerignore`
- Modify: `memo/project.env`
- Modify: `memo/pyproject.toml`
- Modify: `memo/uv.lock`
- Modify: `.github/workflows/build-memo-image.yml`
- Modify: `infra/build-and-push.sh`
- Modify: `rtrrl/infra/control-plane/tests/test_image_catalog.py`

**Interfaces:**
- Produces: memo image label `org.rtrrl.trainer.scripts.v1`.
- Produces: catalog containing exactly `memo_stream_ac` and `memo_rtrrl`.
- Consumes: `trainer-image-catalog` CLI and standalone `training-sdk`.

- [ ] **Step 1: Write failing repository-catalog tests**

```python
def test_memo_catalog_contains_only_initial_launchers(repo_root):
    catalog = load_catalog_file(repo_root / "memo/infra/scripts/index.yaml")
    assert set(catalog.scripts) == {"memo_stream_ac", "memo_rtrrl"}
    assert catalog.scripts["memo_stream_ac"].environments
    assert catalog.scripts["memo_stream_ac"].fields["agent_type"].choices == ("rtu_rtrl",)
    assert catalog.scripts["memo_rtrrl"].fields["rtrrl_topology"].choices == ("shared",)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
cd rtrrl/infra/control-plane
uv run pytest tests/test_image_catalog.py -v
```

Expected: memo catalog is missing.

- [ ] **Step 3: Add the exact descriptors**

```yaml
# memo/infra/scripts/index.yaml
protocol_version: "1"
scripts:
  memo_stream_ac: memo_stream_ac.yaml
  memo_rtrrl: memo_rtrrl.yaml
```

```yaml
# memo/infra/scripts/memo_stream_ac.yaml
name: memo_stream_ac
argv: [python, experiments/memo_stream_ac/run.py, --config_path, "{config_path}"]
sdk_protocol_version: "1"
environments: [memory_chain, kmemory_chain, mujoco_masked]
defaults:
  environment:
    name: memory_chain
    options: {length: 16, max_episode_steps: 16}
  training_budget: {env_steps: 500000}
  logging: {aim_every_env_steps: 10000, rerun_every_episodes: 100}
objective: {metric: eval/rewards, direction: maximize, reduction: last}
fields:
  agent_type:
    path: agent_type
    type: str
    default: rtu_rtrl
    choices: [rtu_rtrl]
  seed:
    path: seed
    type: int
    default: 0
    constraints: {ge: 0}
  hidden_dim:
    path: hidden_dim
    type: int
    default: 192
    searchable: true
    constraints: {gt: 0}
    default_search: {values: [64, 128, 192]}
  encoder_dim:
    path: encoder_dim
    type: int
    default: 64
    searchable: true
    constraints: {gt: 0}
    default_search: {values: [32, 64, 128]}
  gamma:
    path: gamma
    type: float
    default: 0.99
    searchable: true
    constraints: {gt: 0, le: 1}
    default_search: {min: 0.9, max: 1.0, scale: linear}
  trace_lambda:
    path: trace_lambda
    type: float
    default: 0.9
    searchable: true
    constraints: {ge: 0, le: 1}
    default_search: {min: 0.7, max: 1.0, scale: linear}
  actor_lr:
    path: actor_lr
    type: float
    default: 1.0
    searchable: true
    constraints: {gt: 0}
    default_search: {min: 0.01, max: 2.0, scale: log}
  critic_lr:
    path: critic_lr
    type: float
    default: 1.0
    searchable: true
    constraints: {gt: 0}
    default_search: {min: 0.01, max: 2.0, scale: log}
  entropy_coefficient:
    path: entropy_coefficient
    type: float
    default: 0.01
    searchable: true
    constraints: {ge: 0}
    default_search: {min: 0.00001, max: 0.1, scale: log}
  num_envs:
    path: num_envs
    type: int
    default: 16
    constraints: {gt: 0}
```

```yaml
# memo/infra/scripts/memo_rtrrl.yaml
name: memo_rtrrl
argv: [python, experiments/memo_rtrrl/run.py, --config_path, "{config_path}"]
sdk_protocol_version: "1"
environments: [hopper]
defaults:
  environment:
    name: hopper
    options:
      {mode: F, backend: spring, max_episode_steps: 1000,
       normalize_obs: true, normalize_reward: true}
  training_budget: {env_steps: 1000000}
  logging: {aim_every_env_steps: 50000, rerun_every_episodes: 20}
objective: {metric: eval/rewards, direction: maximize, reduction: last}
fields:
  rtrrl_topology:
    path: rtrrl_topology
    type: str
    default: shared
    choices: [shared]
  seed:
    path: seed
    type: int
    default: 0
    constraints: {ge: 0}
  backbone:
    path: backbone
    type: str
    default: lru
    choices: [lru, rtu]
  hidden_dim:
    path: hidden_dim
    type: int
    default: 32
    searchable: true
    constraints: {gt: 0}
    default_search: {values: [16, 32, 64]}
  gamma:
    path: gamma
    type: float
    default: 0.95
    searchable: true
    constraints: {gt: 0, le: 1}
    default_search: {min: 0.9, max: 1.0, scale: linear}
  lambda_pi:
    path: lambda_pi
    type: float
    default: 0.97
    searchable: true
    constraints: {ge: 0, le: 1}
    default_search: {min: 0.8, max: 1.0, scale: linear}
  lambda_v:
    path: lambda_v
    type: float
    default: 0.9
    searchable: true
    constraints: {ge: 0, le: 1}
    default_search: {min: 0.8, max: 1.0, scale: linear}
  lambda_rnn:
    path: lambda_rnn
    type: float
    default: 0.945
    searchable: true
    constraints: {ge: 0, le: 1}
    default_search: {min: 0.8, max: 1.0, scale: linear}
  td_lr:
    path: td_lr
    type: float
    default: 0.00003
    searchable: true
    constraints: {gt: 0}
    default_search: {min: 0.000001, max: 0.001, scale: log}
  rnn_lr:
    path: rnn_lr
    type: float
    default: 0.000002
    searchable: true
    constraints: {gt: 0}
    default_search: {min: 0.0000001, max: 0.0001, scale: log}
  eta_pi:
    path: eta_pi
    type: float
    default: 0.38
    searchable: true
    constraints: {gt: 0}
    default_search: {min: 0.05, max: 1.0, scale: log}
  eta_f:
    path: eta_f
    type: float
    default: 0.5
    searchable: true
    constraints: {gt: 0}
    default_search: {min: 0.05, max: 1.0, scale: log}
  entropy_rate:
    path: entropy_rate
    type: float
    default: 0.00003
    searchable: true
    constraints: {ge: 0}
    default_search: {min: 0.000001, max: 0.001, scale: log}
```

- [ ] **Step 4: Make memo images lock-consistent and catalog-bound**

Both Dockerfiles use repository-root context:

```dockerfile
ARG TRAINER_SCRIPT_CATALOG
RUN test -n "${TRAINER_SCRIPT_CATALOG}"
LABEL org.rtrrl.trainer.scripts.v1="${TRAINER_SCRIPT_CATALOG}"

COPY training-sdk /workspace/training-sdk
COPY memo/pyproject.toml memo/uv.lock /app/
RUN uv sync --frozen --no-dev --no-editable
COPY memo /app
COPY memo/infra/scripts /opt/trainer/scripts
```

The repository-root `.dockerignore` admits only the two required source trees:

```dockerignore
**
!memo/
!memo/**
!training-sdk/
!training-sdk/**
memo/.venv
memo/.pytest_cache
memo/**/__pycache__
training-sdk/.venv
training-sdk/.pytest_cache
training-sdk/**/__pycache__
```

GPU uses the lock-defined extra:

```dockerfile
RUN uv sync --frozen --no-dev --no-editable --extra brax --extra cuda12
```

The memo manifest declares the exact lock-compatible extra:

```toml
[project.optional-dependencies]
cuda12 = ["jax[cuda12]==0.10.0"]
```

Delete the independent `uv pip install "jax[cuda12]==0.10.2"` command. The
pyproject extra resolves the same JAX/JAXLIB version as `uv.lock`.

- [ ] **Step 5: Update local and GitHub builders**

`memo/project.env` opts in explicitly:

```bash
export TRAINER_CATALOG_INDEX="infra/scripts/index.yaml"
export DOCKER_BUILD_CONTEXT="repository"
```

Local build script:

```bash
if [ -n "${TRAINER_CATALOG_INDEX:-}" ]; then
  CATALOG_PATH="${PROJECT_DIR}/${TRAINER_CATALOG_INDEX}"
  [ -f "${CATALOG_PATH}" ] || {
    echo "ERROR: catalog index not found: ${CATALOG_PATH}" >&2
    exit 1
  }
  TRAINER_SCRIPT_CATALOG="$(
    uv run --project "${REPOSITORY_ROOT}/rtrrl/infra/control-plane" \
      trainer-image-catalog "${CATALOG_PATH}"
  )"
  BUILD_ARGS+=(--build-arg "TRAINER_SCRIPT_CATALOG=${TRAINER_SCRIPT_CATALOG}")
fi
```

GitHub workflow adds `astral-sh/setup-uv`, encodes the memo catalog, passes one-line output through `build-args`, and triggers on `memo/**`, `training-sdk/**`, and `rtrrl/infra/control-plane/**`.

- [ ] **Step 6: Verify catalog and CPU image**

Run:

```bash
CATALOG="$(
  uv run --project rtrrl/infra/control-plane \
    trainer-image-catalog memo/infra/scripts/index.yaml
)"
docker build --platform linux/amd64 \
  -f memo/infra/docker/Dockerfile \
  --build-arg "TRAINER_SCRIPT_CATALOG=${CATALOG}" \
  -t trainer-memo-smoke .
docker inspect trainer-memo-smoke \
  --format '{{ index .Config.Labels "org.rtrrl.trainer.scripts.v1" }}'
docker run --rm trainer-memo-smoke \
  python experiments/memo_stream_ac/run.py \
  --config_path tests/fixtures/facility_stream_ac.yml

ARTIFACTS="$(mktemp -d)"
docker run --rm \
  -e TRAINER_RUN_CONTEXT_PATH=/app/tests/fixtures/facility_run_context.json \
  -e AIM_REPO=/tmp/aim \
  -v "${ARTIFACTS}:/artifacts" \
  trainer-memo-smoke \
  python experiments/memo_stream_ac/run.py \
  --config_path tests/fixtures/facility_stream_ac.yml
test -s "${ARTIFACTS}/aim-buffer/events.jsonl"
```

The context fixture uses `"artifact_directory": "/artifacts"` and an objective
of `eval/rewards`. Expected: non-empty label, both local and facility commands
exit zero, and the mounted spool is non-empty. Do not push.

- [ ] **Step 7: Verify tests and commit**

Run:

```bash
uv run --project rtrrl/infra/control-plane pytest rtrrl/infra/control-plane/tests/test_image_catalog.py -v
cd memo
uv lock --check
cd ..
git diff --check
```

Then:

```bash
git add memo/infra memo/project.env memo/pyproject.toml memo/uv.lock \
  .dockerignore .github/workflows/build-memo-image.yml infra/build-and-push.sh \
  rtrrl/infra/control-plane/tests/test_image_catalog.py
git commit -m "feat(infra): bind memo launchers to training images"
```

---

### Task 6: Remove Legacy Registrations and Prove the Narrow Scope

**Files:**
- Delete: `rtrrl/infra/scripts/index.yaml`
- Delete: `rtrrl/infra/scripts/rtrrl.yaml`
- Delete: `rtrrl/infra/scripts/ppo_baseline.yaml`
- Delete: `rtrrl/infra/scripts/sac_baseline.yaml`
- Modify: `rtrrl/infra/docker/Dockerfile`
- Modify: `rtrrl/infra/docker/Dockerfile.gpu`
- Modify: `.github/workflows/build-rtrrl-image.yml`
- Modify: `rtrrl/infra/control-plane/tests/test_image_catalog.py`
- Create: `memo/docs/trainer-facility.md`

**Interfaces:**
- Guarantees: legacy RTRRL images have no facility catalog label.
- Documents: only two memo script identities and environment-as-group configuration.
- Preserves: generic catalog codec and ECR digest reader.

- [ ] **Step 1: Write the failing scope test**

```python
def test_only_memo_image_has_initial_registration(repo_root):
    assert not (repo_root / "rtrrl/infra/scripts/index.yaml").exists()
    memo = load_catalog_file(repo_root / "memo/infra/scripts/index.yaml")
    assert set(memo.scripts) == {"memo_stream_ac", "memo_rtrrl"}
    for forbidden in ("qrc", "tbptt", "independent", "ppo_baseline", "sac_baseline"):
        assert forbidden not in json.dumps(memo.model_dump(mode="json"))
```

- [ ] **Step 2: Verify RED**

Run:

```bash
cd rtrrl/infra/control-plane
uv run pytest tests/test_image_catalog.py::test_only_memo_image_has_initial_registration -v
```

Expected: legacy descriptor files still exist.

- [ ] **Step 3: Remove only legacy registration wiring**

Delete legacy descriptors. Remove these lines from both legacy Dockerfiles:

```dockerfile
ARG TRAINER_SCRIPT_CATALOG
RUN test -n "${TRAINER_SCRIPT_CATALOG}"
LABEL org.rtrrl.trainer.scripts.v1="${TRAINER_SCRIPT_CATALOG}"
COPY infra/scripts /opt/trainer/scripts
```

Remove the setup-uv catalog step and `TRAINER_SCRIPT_CATALOG` build argument from `build-rtrrl-image.yml`. Keep unrelated image build behavior unchanged.

- [ ] **Step 4: Add exact user documentation**

`memo/docs/trainer-facility.md` includes:

```yaml
experiment:
  name: memory-comparison
defaults:
  image: <memo-image-tag-or-digest>
groups:
  stream-memory:
    script: memo_stream_ac
    environment:
      name: memory_chain
      options:
        length: 75
        max_episode_steps: 75
    parameters:
      agent_type: {values: [rtu_rtrl]}
  stream-mujoco:
    script: memo_stream_ac
    environment:
      name: mujoco_masked
      options:
        env_name: hopper
        mode: velocity
        backend: spring
        max_episode_steps: 1000
    parameters:
      agent_type: {values: [rtu_rtrl]}
```

The document states that the two groups are independent studies even though they share one script, and lists every deferred algorithm/topology.

- [ ] **Step 5: Run complete local regression**

Run:

```bash
uv run --project rtrrl/infra/control-plane pytest -v
uv run --project training-sdk pytest -v
cd memo
uv run pytest tests/online_ac -v
uv run pytest tests/test_facility_launchers.py tests/test_experiment_observability.py -v
uv run ruff check \
  memorax/online_ac memorax/environments experiments/base \
  experiments/memo_stream_ac experiments/memo_rtrrl \
  tests/online_ac tests/test_facility_launchers.py \
  tests/test_experiment_observability.py
uv lock --check
cd ../rtrrl
uv lock --check
cd ..
git diff --check
```

Expected: all commands return zero.

- [ ] **Step 6: Commit**

```bash
git rm -r rtrrl/infra/scripts
git add rtrrl/infra/docker .github/workflows/build-rtrrl-image.yml \
  rtrrl/infra/control-plane/tests/test_image_catalog.py memo/docs/trainer-facility.md
git commit -m "chore(infra): remove unsupported legacy registrations"
```

---

## Execution Order

Execute Tasks 1 and 2 first. Task 3 may then proceed independently. Task 4
requires Tasks 2 and 3. Task 5 requires Tasks 1, 2, and 4. Task 6 runs only
after the memo catalog and image smoke in Task 5 succeed.

The real AWS Batch smoke, smoke-resource cleanup, and old-queue deletion remain
in the separate AWS migration plan and still require explicit execution
authorization.
