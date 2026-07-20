# Training Observability SDK Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide the facility-owned SDK and script integrations that record structured Aim runs, mandatory episode summaries, selected complete Rerun episodes, checkpoints, and recoverable local metric buffers.

**Architecture:** The worker injects an immutable run-context file. Training scripts call a narrow `TrainingRun` interface; Aim, Rerun, and spool adapters remain behind the SDK. Existing logger injection stays compatible, and trajectory collection remains outside JIT update kernels.

**Tech Stack:** Python 3.10, Aim 3.28, Rerun SDK, NumPy, pytest, existing JAX/Brax test environment.

## Global Constraints

- The exact Aim experiment is the user experiment name.
- Run names are `<group>-<four-digit-number>`.
- Filterable identity belongs in nested `hparams`, never only in the run name.
- Training episode summaries are mandatory and cannot be disabled.
- Rerun receives complete episodes only.
- SDK methods do not expose Aim hashes, S3 keys, Batch objects, or Optuna.
- SDK logging and array conversion remain outside JIT kernels.
- Existing learning updates and numerical behavior must not change.
- Commit commands require separate explicit user authorization.

---

### Task 1: Run Context and SDK Protocol

**Files:**
- Create: `rtrrl/training_sdk/__init__.py`
- Create: `rtrrl/training_sdk/context.py`
- Create: `rtrrl/training_sdk/types.py`
- Test: `rtrrl/tests/training_sdk/test_context.py`
- Test: `rtrrl/tests/training_sdk/test_types.py`

**Interfaces:**
- `RunContext.from_path(path: Path) -> RunContext`
- `current_run() -> TrainingRun`
- `Episode` complete trajectory value object.

- [ ] **Step 1: Write failing context tests**

```python
def test_context_preserves_user_experiment_and_structured_identity(tmp_path):
    path = write_context(tmp_path, experiment_name="hopper", group="dual", run_number=12)
    context = RunContext.from_path(path)
    assert context.experiment_name == "hopper"
    assert context.run_name == "dual-0012"
    assert context.hparams["identity"]["group"] == "dual"
    assert context.hparams["identity"]["run_number"] == 12


def test_episode_must_be_complete():
    with pytest.raises(ValueError, match="complete"):
        Episode(
            number=4,
            phase="eval",
            start_env_steps=100,
            end_env_steps=120,
            observations=[1, 2],
            actions=[0],
            rewards=[1],
            terminals=[False],
            truncations=[False],
        )
```

- [ ] **Step 2: Verify RED**

Run:

```bash
cd rtrrl
uv run pytest tests/training_sdk/test_context.py tests/training_sdk/test_types.py -v
```

Expected: import failure for `training_sdk`.

- [ ] **Step 3: Implement immutable context and episode types**

```python
JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class RunContext:
    experiment_name: str
    experiment_id: str
    group: str
    script: str
    run_id: str
    run_number: int
    trial_number: int
    seed: int
    metadata: Mapping[str, JsonValue]
    environment: Mapping[str, JsonValue]
    training_budget: Mapping[str, JsonValue]
    fixed_parameters: Mapping[str, JsonValue]
    sampled_parameters: Mapping[str, JsonValue]
    final_parameters: Mapping[str, JsonValue]
    image_digest: str
    resource_profile: str
    artifact_directory: Path

    @property
    def run_name(self) -> str:
        return f"{self.group}-{self.run_number:04d}"

    @property
    def hparams(self) -> dict[str, JsonValue]:
        return {
            "identity": {
                "group": self.group,
                "script": self.script,
                "run_number": self.run_number,
                "trial_number": self.trial_number,
                "seed": self.seed,
                "run_id": self.run_id,
            },
            "metadata": dict(self.metadata),
            "environment": dict(self.environment),
            "training_budget": dict(self.training_budget),
            "parameters": {
                "fixed": dict(self.fixed_parameters),
                "sampled": dict(self.sampled_parameters),
                "final": dict(self.final_parameters),
            },
            "infrastructure": {
                "image_digest": self.image_digest,
                "resource_profile": self.resource_profile,
            },
        }
```

`Episode.__post_init__` verifies equal transition lengths, terminal/truncated
completion, and `end_env_steps >= start_env_steps`.

- [ ] **Step 4: Verify GREEN**

Run targeted tests and `git diff --check`; expect zero failures.

---

### Task 2: Aim Adapter, Mandatory Episode Summaries, and Spool

**Files:**
- Create: `rtrrl/training_sdk/aim_adapter.py`
- Create: `rtrrl/training_sdk/spool.py`
- Create: `rtrrl/training_sdk/run.py`
- Test: `rtrrl/tests/training_sdk/test_aim_adapter.py`
- Test: `rtrrl/tests/training_sdk/test_spool.py`

**Interfaces:**
- `TrainingRun.log_metrics(env_steps, metrics)`.
- `TrainingRun.log_episode_summary(...)`.
- `TrainingRun.finish(final_metrics)`.
- `EventSpool.replay(sink)`.

- [ ] **Step 1: Write failing Aim and spool tests**

```python
def test_aim_uses_exact_experiment_and_nested_hparams(context, fake_aim):
    run = TrainingRun(context, aim=fake_aim, rerun=NullRerun(), spool=MemorySpool())
    run.start()
    assert fake_aim.experiment == context.experiment_name
    assert fake_aim.name == context.run_name
    assert fake_aim.hparams["identity"]["group"] == context.group


def test_episode_summary_is_never_throttled(training_run):
    training_run.log_episode_summary(env_steps=10, episode_return=2.5, episode_length=10)
    training_run.log_episode_summary(env_steps=11, episode_return=3.5, episode_length=1)
    assert training_run.aim.metric_names == [
        "train/episode_return",
        "train/episode_length",
        "train/env_steps",
    ] * 2


def test_spool_replay_is_idempotent(context, failing_then_healthy_aim, tmp_path):
    spool = EventSpool(tmp_path / "events.jsonl")
    run = TrainingRun(context, failing_then_healthy_aim, NullRerun(), spool)
    run.log_metrics(100, {"eval/reward": 4.0})
    spool.replay(failing_then_healthy_aim)
    spool.replay(failing_then_healthy_aim)
    assert failing_then_healthy_aim.event_ids == [spool.events[0].event_id]
```

- [ ] **Step 2: Verify RED**

Run the two targeted test files; expect missing-module failures.

- [ ] **Step 3: Implement append-before-send events**

```python
def _emit(self, event: MetricEvent) -> None:
    self.spool.append(event)
    try:
        self.aim.send(event)
    except AimUnavailable:
        return
    self.spool.mark_sent(event.event_id)


def log_episode_summary(
    self,
    *,
    env_steps: int,
    episode_return: float,
    episode_length: int,
) -> None:
    self._emit(MetricEvent.episode_summary(
        env_steps=env_steps,
        episode_return=episode_return,
        episode_length=episode_length,
    ))
```

General metrics use `aim_every_env_steps`; summaries bypass throttling.
`finish()` emits descriptor-selected final metrics and finalized state only
after all required values are finite.

- [ ] **Step 4: Verify GREEN**

Run targeted tests and `git diff --check`; expect zero failures.

---

### Task 3: Complete-Episode Rerun Adapter and Artifacts

**Files:**
- Create: `rtrrl/training_sdk/rerun_adapter.py`
- Test: `rtrrl/tests/training_sdk/test_rerun_adapter.py`
- Modify: `rtrrl/pyproject.toml`
- Modify: `rtrrl/uv.lock`

**Interfaces:**
- `TrainingRun.log_episode(episode: Episode)`.
- RRD path `<experiment>/<run>/episode-<six digits>.rrd`.

- [ ] **Step 1: Write failing Rerun tests**

```python
def test_only_selected_complete_episode_is_recorded(context, tmp_path):
    adapter = RerunAdapter(context, every_episodes=100, root=tmp_path)
    adapter.log_episode(complete_episode(number=99))
    adapter.log_episode(complete_episode(number=100))
    assert list(tmp_path.rglob("*.rrd")) == [
        tmp_path / "hopper" / "dual-0012" / "episode-000100.rrd"
    ]


def test_rerun_metadata_contains_search_keys(context, fake_rerun):
    RerunAdapter(context, every_episodes=1, sink=fake_rerun).log_episode(
        complete_episode(number=7, start=100, end=120)
    )
    assert fake_rerun.properties["group"] == "dual"
    assert fake_rerun.properties["episode_number"] == 7
    assert fake_rerun.properties["start_env_steps"] == 100
    assert fake_rerun.properties["end_env_steps"] == 120
```

- [ ] **Step 2: Verify RED**

Run `uv run pytest tests/training_sdk/test_rerun_adapter.py -v`.

Expected: missing adapter.

- [ ] **Step 3: Implement complete-episode recording**

```python
def log_episode(self, episode: Episode) -> Path | None:
    if episode.number % self.every_episodes:
        return None
    path = (
        self.root
        / safe_component(self.context.experiment_name)
        / self.context.run_name
        / f"episode-{episode.number:06d}.rrd"
    )
    recording = self.factory(path, self.context, episode)
    recording.log_observations(np.asarray(episode.observations))
    recording.log_actions(np.asarray(episode.actions))
    recording.log_rewards(np.asarray(episode.rewards))
    recording.log_terminals(np.asarray(episode.terminals))
    recording.log_truncations(np.asarray(episode.truncations))
    recording.save()
    return path
```

Add and lock a Rerun version proven by the tests.

- [ ] **Step 4: Verify GREEN**

Run targeted tests plus `uv lock --check`; expect zero failures.

---

### Task 4: Existing Logger Compatibility

**Files:**
- Modify: `rtrrl/logging_util.py`
- Test: `rtrrl/tests/training_sdk/test_logging_compat.py`

**Interfaces:**
- Existing `DummyLogger`, `AimLogger`, and `MultiLogger` preserve old methods.
- New `log_episode_summary` and `log_episode` methods delegate to the SDK.

- [ ] **Step 1: Write failing compatibility tests**

```python
def test_multilogger_fans_out_episode_calls():
    first, second = Recorder(), Recorder()
    logger = MultiLogger([first, second])
    logger.log_episode_summary(env_steps=10, episode_return=2.0, episode_length=10)
    logger.log_episode(complete_episode(number=1))
    assert first.calls == second.calls


def test_existing_log_api_is_unchanged():
    logger = DummyLogger()
    logger.log({"eval/reward": 1.0}, step=10)
    logger.log_params({"seed": 7})
    logger.finalize()
```

- [ ] **Step 2: Verify RED**

Run the compatibility test; expect missing episode methods.

- [ ] **Step 3: Add backward-compatible methods**

```python
class DummyLogger:
    def log_episode_summary(self, **summary) -> None:
        return None

    def log_episode(self, episode: Episode) -> None:
        return None


class MultiLogger:
    def log_episode_summary(self, **summary) -> None:
        for logger in self.loggers:
            logger.log_episode_summary(**summary)

    def log_episode(self, episode: Episode) -> None:
        for logger in self.loggers:
            logger.log_episode(episode)
```

`AimLogger` delegates identity/finalization to `TrainingRun` rather than
duplicating the nested hparams contract.

- [ ] **Step 4: Verify GREEN**

Run existing logger tests, new compatibility tests, and `git diff --check`.

---

### Task 5: Instrument Registered Training Scripts

**Files:**
- Modify: `rtrrl/rtrrl.py`
- Modify: `rtrrl/rtrrl_lru.py`
- Modify: `rtrrl/ppo_baseline.py`
- Modify: `rtrrl/sac_baseline.py`
- Test: `rtrrl/tests/training_sdk/test_script_contracts.py`
- Test: focused existing parity/JIT tests affected by evaluation output changes

**Interfaces:**
- Every registered script emits episode summaries.
- Every registered script can emit selected complete evaluation episodes.

- [ ] **Step 1: Write failing script-contract tests**

```python
@pytest.mark.parametrize("script", ["rtrrl", "rtrrl_lru", "ppo_baseline", "sac_baseline"])
def test_registered_script_emits_required_records(script, smoke_runner):
    records = smoke_runner.run(script)
    assert records.episode_summaries
    assert all("train/env_steps" in item for item in records.episode_summaries)
    assert records.complete_episodes
    assert all(episode.complete for episode in records.complete_episodes)
```

- [ ] **Step 2: Verify RED**

Run the contract test; expect missing episode records.

- [ ] **Step 3: Retain complete evaluation transitions**

For scan-based RTRRL evaluation, return transition fields from the scan:

```python
transition = {
    "observation": timestep.observation,
    "action": action,
    "reward": next_timestep.reward,
    "terminated": next_timestep.terminated,
    "truncated": next_timestep.truncated,
    "state": next_timestep.state,
}
```

After JIT returns, truncate at the first terminal/truncation and construct
`Episode`. For PPO/SAC, use the trained policy returned by the existing train
function to run one explicit evaluation rollout at selected recording points.
Do not alter update equations or place SDK calls inside JIT.

- [ ] **Step 4: Emit training summaries**

At existing episode-return points, call:

```python
logger.log_episode_summary(
    env_steps=int(env_steps),
    episode_return=float(episode_return),
    episode_length=int(episode_length),
)
```

Call `logger.log_episode(episode)` only after complete episode construction.

- [ ] **Step 5: Verify GREEN and numerical safety**

Run:

```bash
cd rtrrl
uv run pytest tests/training_sdk -v
uv run pytest tests -k "parity or jit or smoke" -v
git diff --check
```

Expected: new SDK/contract tests and all selected existing parity/JIT tests pass.

---

### Task 6: SDK User Documentation

**Files:**
- Create: `rtrrl/docs/training-sdk/quickstart.md`
- Create: `rtrrl/docs/training-sdk/api.md`
- Create: `rtrrl/docs/training-sdk/aim-rerun.md`
- Create: `rtrrl/examples/training_sdk_minimal.py`

**Interfaces:**
- Documentation is the supported user entry point.

- [ ] **Step 1: Add a documentation test**

```python
def test_documented_minimal_example_runs(tmp_path):
    result = runpy.run_path(
        "examples/training_sdk_minimal.py",
        init_globals={"OUTPUT_DIRECTORY": tmp_path},
    )
    assert result["RUN_COMPLETED"] is True
```

- [ ] **Step 2: Verify RED**

Run the documentation test; expect missing example failure.

- [ ] **Step 3: Add the exact minimal example**

```python
from training_sdk import Episode, current_run

observations = [[0.0], [1.0]]
actions = [0]
rewards = [1.0]
terminals = [True]
truncations = [False]

run = current_run()
run.log_episode_summary(env_steps=10, episode_return=1.0, episode_length=10)
run.log_episode(Episode.from_transitions(
    number=1,
    phase="eval",
    start_env_steps=0,
    end_env_steps=10,
    observations=observations,
    actions=actions,
    rewards=rewards,
    terminals=terminals,
    truncations=truncations,
))
run.finish({"eval/reward": 1.0})
RUN_COMPLETED = True
```

The quick start explains context injection, lifecycle, required summaries,
complete episodes, checkpoints, Aim hparams filters, Rerun paths, local
development, and buffering.

- [ ] **Step 4: Verify full SDK plan**

Run:

```bash
cd rtrrl
uv run pytest tests/training_sdk -v
uv run pytest tests -k "parity or jit or smoke" -v
git diff --check
```

Expected: zero failures.

- [ ] **Step 5: Review checkpoint**

Review against
`docs/superpowers/specs/2026-07-20-training-observability-sdk-design.md`.
Commit only after explicit authorization, in task-sized commits.
